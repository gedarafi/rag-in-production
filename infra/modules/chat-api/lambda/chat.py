"""Chat endpoint: POST /api/chat with {"message": "..."}, answered by a Bedrock model.

API Gateway has already validated the Cognito token before this runs. The model is MODEL_ID
(on LocalStack an `ollama.<model>` ID, which LocalStack runs on CPU). There is no retrieval yet:
the RAG pipeline (retrieve from the knowledge base, then generate) will extend answer() later.
"""

import base64
import json
import os

MAX_MESSAGE_CHARS = 4000

SYSTEM_PROMPT = (
    "You are the assistant of a university course. Answer the student's question briefly and "
    "clearly. If you don't know the answer, say so instead of guessing."
)

# Bedrock errors that mean "try again later" rather than "this request is broken".
BUSY_ERRORS = {"ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException"}

_client = None


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


def answer(message: str, user_id: str) -> str:
    try:
        res = bedrock().converse(
            modelId=os.environ["MODEL_ID"],
            system=[{"text": SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": message}]}],
            inferenceConfig={"maxTokens": int(os.environ.get("MAX_TOKENS", "512")), "temperature": 0.2},
        )
    except Exception as err:
        # botocore's ClientError carries the AWS error code; timeouts and connection errors don't.
        code = (getattr(err, "response", None) or {}).get("Error", {}).get("Code", "")
        print(f"Bedrock call failed for user {user_id}: {code or type(err).__name__}: {err}")
        if code in BUSY_ERRORS:
            raise ModelError(429, "The assistant is busy. Try again in a moment.") from err
        if code:
            raise ModelError(502, "The assistant couldn't answer. Try again.") from err
        raise ModelError(504, "The assistant took too long to answer. Try again.") from err

    text = "".join(block.get("text", "") for block in res["output"]["message"]["content"]).strip()
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
