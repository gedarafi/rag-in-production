"""The chat Lambda's traces arrive in a running Phoenix (`just phoenix-up`, then `just test-phoenix`).

Checks the pinned Phoenix image against the handler's real exporter: the spans must be stored and
come back from the REST API with the chat and token counts. The handler sets up its exporter when it
is imported, so it runs in a child process with TRACING_ENDPOINT set.
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
PROJECT = "rag-in-production"  # set by the handler's tracer resource
ROOT = Path(__file__).resolve().parent.parent

CHILD = """
import json, sys
from src.chat import handler as chat
from tests.fakes import FakeBedrock, event

chat._client = FakeBedrock(usage={"inputTokens": 12, "outputTokens": 5, "totalTokens": 17})
result = chat.handler(event({"message": sys.argv[1]}), None)
assert result["statusCode"] == 200, result
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


def spans_with_question(question: str, want: int = 2, timeout: float = 15) -> list[dict]:
    """The spans of the trace that asked question, once Phoenix has stored all of them."""
    deadline = time.monotonic() + timeout
    while True:
        spans = get(f"/v1/projects/{PROJECT}/spans?limit=100")["data"]
        trace_ids = {s["context"]["trace_id"] for s in spans if s["attributes"].get("input.value") == question}
        found = [s for s in spans if s["context"]["trace_id"] in trace_ids]
        if len(found) >= want or time.monotonic() > deadline:
            return found
        time.sleep(0.5)


def test_handler_trace_reaches_phoenix():
    question = f"What is in week 3? ({uuid.uuid4().hex[:8]})"  # finds this run's trace among the others
    env = {**os.environ, "TRACING_ENDPOINT": f"{PHOENIX_URL}/v1/traces", "MODEL_ID": "test-model"}
    env.pop("MODEL_API", None)
    subprocess.run([sys.executable, "-c", CHILD, question], cwd=ROOT, env=env, check=True, timeout=60)

    spans = {s["name"]: s for s in spans_with_question(question)}
    assert set(spans) == {"answer", "bedrock.converse"}

    chain, llm = spans["answer"], spans["bedrock.converse"]
    assert chain["span_kind"] == "CHAIN"
    assert llm["span_kind"] == "LLM"
    assert llm["parent_id"] == chain["context"]["span_id"]
    assert llm["attributes"]["llm.model_name"] == "test-model"
    assert llm["attributes"]["output.value"] == "Week 3 covers recursion."
    counts = [llm["attributes"][f"llm.token_count.{key}"] for key in ("prompt", "completion", "total")]
    assert counts == [12, 5, 17]
