"""Chat API: the anime recommender answers asynchronously, because an answer can outlast API Gateway's 30 s.

POST /api/chat {"message": "...", "conversation_id"?: "..."}
    Stores a pending job, queues it for the recommender worker (src/recommender/worker.py) and returns
    202 {"job_id", "conversation_id", "status": "pending"}. Without conversation_id a new conversation
    starts; with one, the worker answers in the context of its earlier turns.
GET /api/chat/{job_id}
    The job: {"status": "pending" | "running" | "done" | "failed", "answer"?, "error"?, "trace_id"}.

API Gateway has already validated the Cognito token before this runs; a user only sees their own jobs
and conversations. The trace started here continues in the worker (traceparent in the SQS message), so
one Phoenix or Arize trace covers the whole question.
"""

import base64
import json
import os

from src.chat.store import ChatStore, new_id, valid_id
from src.tracing import annotate, flush, inject, setup_tracing, trace_id, tracer

MAX_MESSAGE_CHARS = 4000

setup_tracing()

_store = None
_sqs = None


def store() -> ChatStore:
    global _store
    if _store is None:
        _store = ChatStore.from_env()
    return _store


def sqs():
    """The SQS client, created on first use so the unit tests need no boto3."""
    global _sqs
    if _sqs is None:
        import boto3
        from botocore.config import Config

        _sqs = boto3.client("sqs", config=Config(connect_timeout=2, read_timeout=5, retries={"max_attempts": 3}))
    return _sqs


def response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json", "cache-control": "no-store"},
        "body": json.dumps(body),
    }


def handler(event, context):
    try:
        return route(event)
    finally:
        flush()


def route(event) -> dict:
    claims = event.get("requestContext", {}).get("authorizer", {}).get("jwt", {}).get("claims", {})
    user_id = claims.get("sub")
    if not user_id:
        return response(401, {"error": "Not signed in."})
    if event.get("requestContext", {}).get("http", {}).get("method", "POST") == "GET":
        return get_job(user_id, (event.get("pathParameters") or {}).get("job_id"))
    return ask(event, user_id)


@tracer.chain(name="api.ask")
def ask(event, user_id: str) -> dict:
    raw = event.get("body") or ""
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode("utf-8", errors="replace")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return response(400, {"error": "Body must be JSON."})
    if not isinstance(payload, dict):
        return response(400, {"error": "Send a non-empty 'message'."})

    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        return response(400, {"error": "Send a non-empty 'message'."})
    if len(message) > MAX_MESSAGE_CHARS:
        return response(400, {"error": f"Messages are limited to {MAX_MESSAGE_CHARS} characters."})
    conversation_id = payload.get("conversation_id")
    if conversation_id is None:
        conversation_id = new_id()
    elif not valid_id(conversation_id):
        return response(400, {"error": "'conversation_id' must be an ID returned by this API."})

    job_id = new_id()
    annotate(**{"session.id": conversation_id, "user.id": user_id, "recommender.job_id": job_id,
                "input.value": message.strip()})
    store().create_job(job_id, user_id, conversation_id, message.strip(), trace_id())
    sqs().send_message(
        QueueUrl=os.environ["QUEUE_URL"],
        MessageBody=json.dumps({"job_id": job_id}),
        MessageAttributes={key: {"DataType": "String", "StringValue": value} for key, value in inject().items()},
    )
    return response(202, {"job_id": job_id, "conversation_id": conversation_id, "status": "pending"})


def get_job(user_id: str, job_id) -> dict:
    job = store().get_job(job_id) if valid_id(job_id) else None
    # Someone else's job is reported exactly like a missing one.
    if job is None or job["user_id"] != user_id:
        return response(404, {"error": "No such question."})
    body = {key: job[key] for key in ("job_id", "conversation_id", "status", "answer", "error", "trace_id") if key in job}
    return response(200, body)
