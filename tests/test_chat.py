import json

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.chat import handler as chat
from src.chat.handler import MAX_MESSAGE_CHARS, handler
from src import tracing
from src.chat.store import ChatStore
from tests.fakes import FakeDynamo, FakeSqs, event, get_event

# The handler's tracer records nothing without a provider; this one keeps the spans for the span tests.
spans = InMemorySpanExporter()
tracing._sdk_provider("test").add_span_processor(SimpleSpanProcessor(spans))


@pytest.fixture(autouse=True)
def aws(monkeypatch):
    """In-memory DynamoDB and SQS behind the handler."""
    dynamo, sqs = FakeDynamo(), FakeSqs()
    monkeypatch.setattr(chat, "_store", ChatStore("jobs", "conversations", client=dynamo))
    monkeypatch.setattr(chat, "_sqs", sqs)
    monkeypatch.setenv("QUEUE_URL", "https://sqs.test/queue")
    spans.clear()
    return dynamo, sqs


def call(evt) -> tuple[int, dict]:
    result = handler(evt, None)
    return result["statusCode"], json.loads(result["body"])


# --- POST /api/chat: store a pending job and queue it ---


def test_queues_a_job(aws):
    dynamo, sqs = aws
    status, body = call(event({"message": "  Recommend something like Frieren  "}))
    assert status == 202
    assert body["status"] == "pending"
    job = chat.store().get_job(body["job_id"])
    assert job["user_id"] == "user-123"
    assert job["message"] == "Recommend something like Frieren"
    assert job["conversation_id"] == body["conversation_id"]
    (message,) = sqs.messages
    assert message["QueueUrl"] == "https://sqs.test/queue"
    assert json.loads(message["MessageBody"]) == {"job_id": body["job_id"]}


def test_the_trace_continues_in_the_worker(aws):
    _, sqs = aws
    status, body = call(event({"message": "hi"}))
    (span,) = [s for s in spans.get_finished_spans() if s.name == "api.ask"]
    traceparent = sqs.messages[0]["MessageAttributes"]["traceparent"]["StringValue"]
    assert format(span.context.trace_id, "032x") in traceparent
    assert chat.store().get_job(body["job_id"])["trace_id"] == format(span.context.trace_id, "032x")
    assert span.attributes["session.id"] == body["conversation_id"]
    assert span.attributes["user.id"] == "user-123"


def test_continues_a_conversation(aws):
    _, first = call(event({"message": "Recommend a mystery anime"}))
    status, second = call(event({"message": "Something shorter?", "conversation_id": first["conversation_id"]}))
    assert status == 202
    assert second["conversation_id"] == first["conversation_id"]
    assert second["job_id"] != first["job_id"]


@pytest.mark.parametrize("conversation_id", ["../other", 42, "a" * 31, "G" * 32])
def test_rejects_foreign_conversation_ids(conversation_id):
    status, body = call(event({"message": "hi", "conversation_id": conversation_id}))
    assert status == 400


def test_accepts_base64_body(aws):
    status, _ = call(event({"message": "hi"}, base64_encoded=True))
    assert status == 202


def test_rejects_missing_claims(aws):
    status, body = call(event({"message": "hi"}, sub=None))
    assert status == 401
    assert aws[1].messages == []


@pytest.mark.parametrize("body", ["not json", [], {}, {"message": ""}, {"message": "   "}, {"message": 42}])
def test_rejects_bad_input(aws, body):
    status, _ = call(event(body))
    assert status == 400
    assert aws[1].messages == []


def test_rejects_too_long_message(aws):
    status, body = call(event({"message": "x" * (MAX_MESSAGE_CHARS + 1)}))
    assert status == 400
    assert str(MAX_MESSAGE_CHARS) in body["error"]


def test_never_cached():
    result = handler(event({"message": "hi"}), None)
    assert result["headers"]["cache-control"] == "no-store"


# --- GET /api/chat/{job_id}: only the caller's own jobs ---


def test_returns_the_job():
    _, created = call(event({"message": "hi"}))
    chat.store().update_job(created["job_id"], "done", answer="Try Monster.")
    status, body = call(get_event(created["job_id"]))
    assert status == 200
    assert body["status"] == "done"
    assert body["answer"] == "Try Monster."
    assert body["conversation_id"] == created["conversation_id"]
    assert "message" not in body and "user_id" not in body


def test_hides_other_users_jobs():
    _, created = call(event({"message": "hi"}))
    assert call(get_event(created["job_id"], sub="someone-else"))[0] == 404


@pytest.mark.parametrize("job_id", [None, "missing", "0" * 32])
def test_unknown_jobs_are_not_found(job_id):
    assert call(get_event(job_id))[0] == 404


def test_get_requires_claims():
    assert call(get_event("0" * 32, sub=None))[0] == 401
