"""Chat endpoint: POST /api/chat with {"message": "..."}, answered by a Bedrock model.

API Gateway has already validated the Cognito token before this runs. The model is MODEL_ID, and
MODEL_API says how to call it:
  converse  Converse on bedrock-runtime (on LocalStack an `ollama.<model>` ID, run on CPU).
  openai    OpenAI-compatible Chat Completions on bedrock-mantle, which has its own quotas.
There is no retrieval yet: the RAG pipeline (retrieve from the knowledge base, then generate) will
extend answer() later.
"""

import base64
import json
import os
import urllib.error
import urllib.request

from openinference.instrumentation import OITracer, TraceConfig
from opentelemetry import trace

MAX_MESSAGE_CHARS = 4000

SYSTEM_PROMPT = (
    "You are the assistant of a university course. Answer the student's question briefly and "
    "clearly. If you don't know the answer, say so instead of guessing."
)

# Bedrock errors that mean "try again later" rather than "this request is broken".
BUSY_ERRORS = {"ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException"}
BUSY_STATUSES = {429, 503}

_client = None
_mantle = None

# On LocalStack the spans go to the local Phoenix (PHOENIX_ENDPOINT); without it the tracer records
# nothing. Spans are sent as each one ends, before the Lambda is frozen.
if os.environ.get("PHOENIX_ENDPOINT"):
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    _provider = TracerProvider(resource=Resource({"openinference.project.name": "rag-in-production"}))
    _provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(os.environ["PHOENIX_ENDPOINT"], timeout=2)))
    trace.set_tracer_provider(_provider)
tracer = OITracer(trace.get_tracer(__name__), config=TraceConfig())


class ModelError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def bedrock():
    """The bedrock-runtime client, created on first use so the unit tests need no boto3."""
    global _client
    if _client is None:
        import boto3
        from botocore.config import Config

        # Stay inside API Gateway's 30-second limit; the caller can simply ask again.
        config = Config(connect_timeout=5, read_timeout=25, retries={"max_attempts": 1})
        _client = boto3.client("bedrock-runtime", config=config)
    return _client


class Mantle:
    """Chat Completions on bedrock-mantle, signed with the Lambda role's credentials.

    boto3 has no bedrock-mantle client, so botocore signs the request and urllib sends it.
    """

    def __init__(self, region: str):
        import botocore.session

        self.region = region
        self.url = f"https://bedrock-mantle.{region}.api.aws/v1/chat/completions"
        self.credentials = botocore.session.get_session().get_credentials()

    def chat_completions(self, **body) -> dict:
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        data = json.dumps(body).encode()
        signed = AWSRequest(method="POST", url=self.url, data=data, headers={"content-type": "application/json"})
        SigV4Auth(self.credentials.get_frozen_credentials(), "bedrock-mantle", self.region).add_auth(signed)
        request = urllib.request.Request(self.url, data=data, headers=dict(signed.headers.items()), method="POST")
        # Same budget as the bedrock-runtime client: inside API Gateway's 30-second limit.
        with urllib.request.urlopen(request, timeout=25) as res:
            return json.load(res)


def mantle():
    global _mantle
    if _mantle is None:
        _mantle = Mantle(os.environ["AWS_REGION"])
    return _mantle


@tracer.llm
def converse(message: str, max_tokens: int) -> str:
    res = bedrock().converse(
        modelId=os.environ["MODEL_ID"],
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": message}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.2},
    )
    return "".join(block.get("text", "") for block in res["output"]["message"]["content"])


@tracer.llm
def chat_completions(message: str, max_tokens: int) -> str:
    res = mantle().chat_completions(
        model=os.environ["MODEL_ID"],
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": message}],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    # A reasoning model that runs out of tokens while thinking returns content null.
    choices = res.get("choices") or [{}]
    return choices[0].get("message", {}).get("content") or ""


@tracer.chain
def answer(message: str, user_id: str) -> str:
    call = chat_completions if os.environ.get("MODEL_API") == "openai" else converse
    try:
        text = call(message, int(os.environ.get("MAX_TOKENS", "512"))).strip()
    except urllib.error.HTTPError as err:
        # bedrock-mantle answers errors with an HTTP status and a JSON body.
        print(f"Bedrock call failed for user {user_id}: HTTP {err.code}: {err.read(500).decode(errors='replace')}")
        if err.code in BUSY_STATUSES:
            raise ModelError(429, "The assistant is busy. Try again in a moment.") from err
        raise ModelError(502, "The assistant couldn't answer. Try again.") from err
    except Exception as err:
        # botocore's ClientError carries the AWS error code; timeouts and connection errors don't.
        code = (getattr(err, "response", None) or {}).get("Error", {}).get("Code", "")
        print(f"Bedrock call failed for user {user_id}: {code or type(err).__name__}: {err}")
        if code in BUSY_ERRORS:
            raise ModelError(429, "The assistant is busy. Try again in a moment.") from err
        if code:
            raise ModelError(502, "The assistant couldn't answer. Try again.") from err
        raise ModelError(504, "The assistant took too long to answer. Try again.") from err

    if not text:
        raise ModelError(502, "The assistant returned an empty answer. Try again.")
    return text


def response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json", "cache-control": "no-store"},
        "body": json.dumps(body),
    }


def handler(event, context):
    claims = event.get("requestContext", {}).get("authorizer", {}).get("jwt", {}).get("claims", {})
    user_id = claims.get("sub")
    if not user_id:
        return response(401, {"error": "Not signed in."})

    raw = event.get("body") or ""
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode("utf-8", errors="replace")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return response(400, {"error": "Body must be JSON."})

    message = payload.get("message") if isinstance(payload, dict) else None
    if not isinstance(message, str) or not message.strip():
        return response(400, {"error": "Send a non-empty 'message'."})
    if len(message) > MAX_MESSAGE_CHARS:
        return response(400, {"error": f"Messages are limited to {MAX_MESSAGE_CHARS} characters."})

    try:
        text = answer(message.strip(), user_id)
    except ModelError as err:
        return response(err.status, {"error": str(err)})
    return response(200, {"answer": text})
