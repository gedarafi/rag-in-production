"""The recommender worker Lambda: answers the chat jobs that the api Lambda queues in SQS.

Each SQS message names one job (src/chat/store.py). The worker loads the conversation's earlier
turns, answers with the recommender over the active artifact bundle, and stores the answer and the new
turn. The page polls the job until it is done or failed.

Failures:
  - Anything the user can act on (a busy or failing model, a question the pipeline can't ground): the
    job fails with that message at once. The Bedrock clients retry throttling themselves, and an SQS
    retry would only come after the visibility timeout, long after the page stopped waiting.
  - Anything else is a bug or an outage: the message is retried and, after the last attempt
    (maxReceiveCount), the job is marked failed and the message lands in the dead-letter queue.

The trace continues the api Lambda's (traceparent from the message attributes).
"""
from __future__ import annotations

import json
import os

from src.chat.store import ChatStore
from src.tracing import annotate, extract, fail, flush, setup_tracing, trace_id, tracer

from .bedrock import ModelError

setup_tracing()

# Errors the user can act on; anything else is retried.
USER_ERRORS = (ModelError, RuntimeError)
MAX_RECEIVES = int(os.environ.get("MAX_RECEIVES", "2"))

_store = None
_artifacts = None
_recommender = None
_version = None


def store() -> ChatStore:
    global _store
    if _store is None:
        _store = ChatStore.from_env()
    return _store


def recommender():
    """The recommender over the active bundle, reloaded when a new version is activated."""
    global _artifacts, _recommender, _version
    from .artifacts import ArtifactStore
    from .service import Recommender

    if _artifacts is None:
        _artifacts = ArtifactStore.from_env()
    version, folder, _ = _artifacts.current()
    if _recommender is None or version != _version:
        _recommender, _version = Recommender(folder).warm(), version
    return _recommender


def handler(event, context):
    failures = []
    try:
        for record in event.get("Records", []):
            try:
                process(record)
            except Exception:
                # Partial batch response: only this message goes back to the queue.
                failures.append({"itemIdentifier": record["messageId"]})
    finally:
        flush()
    return {"batchItemFailures": failures}


def process(record: dict) -> None:
    carrier = {key: value["stringValue"] for key, value in (record.get("messageAttributes") or {}).items()
               if "stringValue" in value}
    job_id = json.loads(record["body"])["job_id"]
    receives = int((record.get("attributes") or {}).get("ApproximateReceiveCount", "1"))
    with tracer.start_as_current_span("recommender.job", context=extract(carrier),
                                      openinference_span_kind="chain") as span:
        job = store().get_job(job_id)
        if job is None:
            print(f"Job {job_id} not found; dropping the message.")
            return
        if job["status"] in ("done", "failed"):
            return  # A duplicate delivery.
        annotate(**{"session.id": job["conversation_id"], "user.id": job["user_id"], "recommender.job_id": job_id,
                    "recommender.attempt": receives, "input.value": job["message"]})
        store().update_job(job_id, "running", trace_id=trace_id())
        try:
            history = store().history(job["user_id"], job["conversation_id"])
            answer = recommender().answer(job["message"], history)
            annotate(**{"output.value": answer, "recommender.artifact_version": _version})
            store().add_turn(job["user_id"], job["conversation_id"], job_id, job["message"], answer)
            store().update_job(job_id, "done", answer=answer)
        except USER_ERRORS as err:
            fail(span, err)
            store().update_job(job_id, "failed", error=str(err))
        except Exception as err:
            fail(span, err)
            print(f"Job {job_id} attempt {receives} failed: {type(err).__name__}: {err}")
            if receives >= MAX_RECEIVES:
                store().update_job(job_id, "failed", error="The assistant couldn't answer. Try again later.")
            else:
                store().update_job(job_id, "pending")
            raise
