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

from openinference.instrumentation import (
    OITracer,
    TraceConfig,
    get_input_attributes,
    get_llm_attributes,
    get_output_attributes,
)
from opentelemetry import trace

MAX_MESSAGE_CHARS = 4000
TEMPERATURE = 0.2

SYSTEM_PROMPT = (
    "You are the assistant of a university course. Answer the student's question briefly and "
    "clearly. If you don't know the answer, say so instead of guessing."
)

# Bedrock errors that mean "try again later" rather than "this request is broken".
BUSY_ERRORS = {"ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException"}
BUSY_STATUSES = {429, 503}

_client = None
_mantle = None

# The spans go to TRACING_ENDPOINT: the local Phoenix on LocalStack, Arize AX on AWS, which also needs
# TRACING_SPACE_ID and the API key in the SSM parameter TRACING_API_KEY_PARAMETER. Without an endpoint
# the tracer records nothing. Each request's spans are sent before the handler returns, since the
# Lambda is frozen afterwards.
_provider = None


def tracing_headers(ssm=None) -> dict[str, str] | None:
    """Arize's auth headers (none for Phoenix), or None if the API key can't be read."""
    headers = {}
    if space_id := os.environ.get("TRACING_SPACE_ID"):
        headers["arize-space-id"] = space_id
    if parameter := os.environ.get("TRACING_API_KEY_PARAMETER"):
        try:
            if ssm is None:
                import boto3
                from botocore.config import Config

                ssm = boto3.client("ssm", config=Config(connect_timeout=2, read_timeout=2, retries={"max_attempts": 2}))
            headers["authorization"] = ssm.get_parameter(Name=parameter, WithDecryption=True)["Parameter"]["Value"]
        except Exception as err:
            # Chat keeps working without traces.
            print(f"Tracing off: could not read {parameter}: {type(err).__name__}: {err}")
            return None
    return headers


def setup_tracing():
    global _provider
    endpoint = os.environ.get("TRACING_ENDPOINT")
    if not endpoint or (headers := tracing_headers()) is None:
        return
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    project = os.environ.get("TRACING_PROJECT", "rag-in-production")
    _provider = TracerProvider(resource=Resource({"openinference.project.name": project}))
    _provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint, headers=headers or None, timeout=3)))
    trace.set_tracer_provider(_provider)


setup_tracing()
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


def request_attributes(message: str, max_tokens: int, **llm) -> dict:
    """The LLM span's request side: model, parameters and the chat sent to it."""
    return {
        **get_llm_attributes(
            provider="aws",
            model_name=os.environ["MODEL_ID"],
            invocation_parameters={"max_tokens": max_tokens, "temperature": TEMPERATURE},
            input_messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": message}],
            **llm,
        ),
        **get_input_attributes(message),
    }


def reply_attributes(text: str, prompt=None, completion=None, total=None) -> dict:
    """The LLM span's reply side, with the token counts the model reported."""
    counts = {"prompt": prompt, "completion": completion, "total": total}
    return {
        **get_llm_attributes(
            output_messages=[{"role": "assistant", "content": text}],
            token_count={key: count for key, count in counts.items() if count is not None} or None,
        ),
        **get_output_attributes(text),
    }


def converse(message: str, max_tokens: int) -> str:
    with tracer.start_as_current_span("bedrock.converse", openinference_span_kind="llm") as span:
        span.set_attributes(request_attributes(message, max_tokens))
        res = bedrock().converse(
            modelId=os.environ["MODEL_ID"],
            system=[{"text": SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": message}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": TEMPERATURE},
        )
        text = "".join(block.get("text", "") for block in res["output"]["message"]["content"])
        usage = res.get("usage") or {}
        span.set_attributes(
            reply_attributes(text, usage.get("inputTokens"), usage.get("outputTokens"), usage.get("totalTokens"))
        )
        return text


def chat_completions(message: str, max_tokens: int) -> str:
    with tracer.start_as_current_span("bedrock-mantle.chat_completions", openinference_span_kind="llm") as span:
        span.set_attributes(request_attributes(message, max_tokens, system="openai"))
        res = mantle().chat_completions(
            model=os.environ["MODEL_ID"],
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": message}],
            max_tokens=max_tokens,
            temperature=TEMPERATURE,
        )
        # A reasoning model that runs out of tokens while thinking returns content null.
        choices = res.get("choices") or [{}]
        text = choices[0].get("message", {}).get("content") or ""
        usage = res.get("usage") or {}
        span.set_attributes(
            reply_attributes(
                text, usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens")
            )
        )
        return text


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
    try:
        return chat(event)
    finally:
        if _provider:
            # One export per request; a slow collector costs at most a few seconds, never the answer.
            _provider.force_flush(timeout_millis=3000)


def chat(event) -> dict:
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
