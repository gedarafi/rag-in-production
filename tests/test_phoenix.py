"""A question's trace arrives in a running Phoenix (`just phoenix-up`, then `just test-phoenix`).

Checks the pinned Phoenix image against the real exporter: the api Lambda queues the question, the
worker answers it with the stand-in model over the fixture catalog, and Phoenix stores one trace with
both Lambdas' spans, down to the LLM and tool spans, which come back from its REST API. The handlers set
up their exporter when they are imported, so it runs in a child process with TRACING_ENDPOINT set.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.phoenix

PHOENIX_URL = os.environ.get("PHOENIX_URL", "http://localhost:6006")
PROJECT = "rag-in-production"  # TRACING_PROJECT's default
ROOT = Path(__file__).resolve().parent.parent

CHILD = """
import json, sys, tempfile
from pathlib import Path
from src.chat import handler as api
from src.chat.store import ChatStore
from src.recommender import worker
from src.recommender.cli import main as cli
from src.recommender.service import Recommender
from tests.fakes import FakeDynamo, FakeSqs, event, sqs_record

workspace = Path(tempfile.mkdtemp())
cli(["build", "--source", "tests/fixtures/anilist-mini.csv", "--workspace", str(workspace),
     "--out", str(workspace / "bundles")])
store, sqs = ChatStore("jobs", "conversations", client=FakeDynamo()), FakeSqs()
api._store = worker._store = store
api._sqs = sqs
worker.recommender = lambda: Recommender(workspace)
result = api.handler(event({"message": sys.argv[1]}), None)
assert result["statusCode"] == 202, result
assert worker.handler({"Records": [sqs_record(sqs.messages[0])]}, None) == {"batchItemFailures": []}
job = store.get_job(json.loads(result["body"])["job_id"])
assert job["status"] == "done", job
"""


def get(path: str):
    with urllib.request.urlopen(f"{PHOENIX_URL}{path}", timeout=5) as response:
        return json.load(response)


@pytest.fixture(scope="module", autouse=True)
def phoenix_is_up():
    try:
        urllib.request.urlopen(f"{PHOENIX_URL}/healthz", timeout=5).close()
    except OSError as error:
        pytest.fail(f"Phoenix is not reachable at {PHOENIX_URL} (run `just phoenix-up`): {error}")


def spans_with_question(question: str, want: int, timeout: float = 20) -> list[dict]:
    """The spans of the trace that asked question, once Phoenix has stored all of them."""
    deadline = time.monotonic() + timeout
    while True:
        spans = get(f"/v1/projects/{PROJECT}/spans?limit=500")["data"]
        trace_ids = {s["context"]["trace_id"] for s in spans if s["attributes"].get("input.value") == question}
        found = [s for s in spans if s["context"]["trace_id"] in trace_ids]
        if len(found) >= want or time.monotonic() > deadline:
            return found
        time.sleep(0.5)


def test_question_trace_reaches_phoenix():
    question = f"Recommend anime like Clockwork Lighthouse. ({uuid.uuid4().hex[:8]})"  # finds this run's trace
    env = {**os.environ, "TRACING_ENDPOINT": f"{PHOENIX_URL}/v1/traces", "MODEL_ID": "stub", "MODEL_API": "stub",
           "QUEUE_URL": "https://sqs.test/queue"}
    env.pop("TRACING_PROJECT", None)
    subprocess.run([sys.executable, "-c", CHILD, question], cwd=ROOT, env=env, check=True, timeout=300)

    spans = spans_with_question(question, want=10)
    assert len({s["context"]["trace_id"] for s in spans}) == 1, "the api and worker spans must share one trace"
    by_name = {s["name"]: s for s in spans}
    assert {"api.ask", "recommender.job", "recommender.ask", "plan_tools", "recommend_anime"} <= set(by_name)

    api, job = by_name["api.ask"], by_name["recommender.job"]
    assert job["parent_id"] == api["context"]["span_id"]
    assert job["attributes"]["session.id"] == api["attributes"]["session.id"]
    assert by_name["plan_tools"]["span_kind"] == "LLM"
    assert by_name["plan_tools"]["attributes"]["llm.model_name"] == "stub"
    assert by_name["recommend_anime"]["span_kind"] == "TOOL"
    assert "Clockwork Lighthouse" in job["attributes"]["output.value"]
