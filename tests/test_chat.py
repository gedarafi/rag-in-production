import json
import urllib.error

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from src.chat import handler as chat
from src.chat.handler import MAX_MESSAGE_CHARS, SYSTEM_PROMPT, handler
from tests.fakes import AwsError, FakeBedrock, FakeMantle, event, http_error

# The handler's tracer records nothing without a provider; this one keeps the spans for the span tests.
spans = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(spans))
trace.set_tracer_provider(_provider)


def call(evt) -> tuple[int, dict]:
    result = handler(evt, None)
    return result["statusCode"], json.loads(result["body"])


@pytest.fixture(autouse=True)
def model_env(monkeypatch):
    """Converse with test-model by default; each test sets the client it needs."""
    monkeypatch.setenv("MODEL_ID", "test-model")
    monkeypatch.delenv("MODEL_API", raising=False)
    monkeypatch.delenv("MAX_TOKENS", raising=False)
    monkeypatch.setattr(chat, "_client", None)
    monkeypatch.setattr(chat, "_mantle", None)


@pytest.fixture
def bedrock(monkeypatch):
    fake = FakeBedrock()
    monkeypatch.setattr(chat, "_client", fake)
    return fake


@pytest.fixture
def mantle(monkeypatch):
    """MODEL_API=openai: Chat Completions on bedrock-mantle."""
    fake = FakeMantle()
    monkeypatch.setattr(chat, "_mantle", fake)
    monkeypatch.setenv("MODEL_ID", "google.gemma-3-4b-it")
    monkeypatch.setenv("MODEL_API", "openai")
    return fake


# --- Converse ---


def test_answers_with_the_model(bedrock):
    status, body = call(event({"message": "  What is in week 3?  "}))
    assert status == 200
    assert body == {"answer": "Week 3 covers recursion."}

    (request,) = bedrock.calls
    assert request["modelId"] == "test-model"
    assert request["system"] == [{"text": SYSTEM_PROMPT}]
    assert request["messages"] == [{"role": "user", "content": [{"text": "What is in week 3?"}]}]
    assert request["inferenceConfig"]["maxTokens"] == 512


def test_max_tokens_is_configurable(bedrock, monkeypatch):
    monkeypatch.setenv("MAX_TOKENS", "64")
    call(event({"message": "hi"}))
    assert bedrock.calls[0]["inferenceConfig"]["maxTokens"] == 64


def test_accepts_base64_body(bedrock):
    status, _ = call(event({"message": "hi"}, base64_encoded=True))
    assert status == 200


def test_rejects_missing_claims(bedrock):
    status, _ = call(event({"message": "hi"}, sub=None))
    assert status == 401
    assert bedrock.calls == []


@pytest.mark.parametrize("body", ["not json", {"message": ""}, {"message": "   "}, {"message": 5}, {}, [1, 2]])
def test_rejects_bad_input(bedrock, body):
    status, payload = call(event(body))
    assert status == 400
    assert "error" in payload
    assert bedrock.calls == []


def test_rejects_too_long_message(bedrock):
    status, _ = call(event({"message": "x" * (MAX_MESSAGE_CHARS + 1)}))
    assert status == 400
    assert bedrock.calls == []


@pytest.mark.parametrize(("error", "expected"), [
    (AwsError("ThrottlingException"), 429),
    (AwsError("ValidationException"), 502),
    (AwsError("AccessDeniedException"), 502),
    (TimeoutError("read timeout"), 504),
], ids=str)
def test_model_errors(monkeypatch, error, expected):
    monkeypatch.setattr(chat, "_client", FakeBedrock(error=error))
    status, body = call(event({"message": "hi"}))
    assert status == expected
    assert "error" in body
    assert "Exception" not in body["error"]  # no AWS internals in the reply


def test_daily_quota_says_so(monkeypatch):
    error = AwsError("ThrottlingException", "Too many tokens per day, please wait before trying again.")
    monkeypatch.setattr(chat, "_client", FakeBedrock(error=error))
    status, body = call(event({"message": "hi"}))
    assert status == 429
    assert "today's model quota" in body["error"]


def test_empty_model_answer_is_an_error(monkeypatch):
    monkeypatch.setattr(chat, "_client", FakeBedrock(reply="   "))
    status, _ = call(event({"message": "hi"}))
    assert status == 502


def test_never_cached(bedrock):
    result = handler(event({"message": "hi"}), None)
    assert result["headers"]["cache-control"] == "no-store"


# --- Chat Completions on bedrock-mantle ---


def test_mantle_answers_with_the_model(mantle):
    status, body = call(event({"message": "  What is in week 3?  "}))
    assert status == 200
    assert body == {"answer": "Week 3 covers recursion."}

    (request,) = mantle.calls
    assert request["model"] == "google.gemma-3-4b-it"
    assert request["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is in week 3?"},
    ]
    assert request["max_tokens"] == 512


def test_mantle_does_not_call_converse(mantle, bedrock):
    call(event({"message": "hi"}))
    assert bedrock.calls == []


@pytest.mark.parametrize(("error", "expected"), [
    (http_error(429), 429),
    (http_error(503), 429),
    (http_error(400), 502),
    (http_error(403), 502),
    (urllib.error.URLError("connection refused"), 504),
    (TimeoutError("read timeout"), 504),
], ids=str)
def test_mantle_model_errors(mantle, monkeypatch, error, expected):
    monkeypatch.setattr(chat, "_mantle", FakeMantle(error=error))
    status, body = call(event({"message": "hi"}))
    assert status == expected
    assert "error" in body
    assert "HTTP" not in body["error"]  # no endpoint internals in the reply


def test_mantle_daily_quota_says_so(mantle, monkeypatch):
    monkeypatch.setattr(chat, "_mantle", FakeMantle(error=http_error(429, "Too many tokens per day")))
    status, body = call(event({"message": "hi"}))
    assert status == 429
    assert "today's model quota" in body["error"]


@pytest.mark.parametrize("reply", ["   ", None])  # None: a reasoning model ran out of tokens while thinking
def test_mantle_empty_model_answer_is_an_error(mantle, monkeypatch, reply):
    monkeypatch.setattr(chat, "_mantle", FakeMantle(reply=reply))
    status, _ = call(event({"message": "hi"}))
    assert status == 502


# --- Spans: each model call is an LLM span under the answer chain, with the chat and token counts ---


@pytest.fixture
def finished_spans():
    spans.clear()
    return spans.get_finished_spans


def llm_span(finished_spans, name):
    (span,) = [s for s in finished_spans() if s.attributes.get("openinference.span.kind") == "LLM"]
    (parent,) = [s for s in finished_spans() if s.name == "answer"]
    assert span.name == name
    assert span.parent.span_id == parent.context.span_id
    return span


def assert_chat(span):
    attributes = span.attributes
    assert attributes["llm.provider"] == "aws"
    assert attributes["llm.model_name"] == "test-model"
    assert json.loads(attributes["llm.invocation_parameters"]) == {"max_tokens": 512, "temperature": 0.2}
    assert attributes["llm.input_messages.0.message.content"] == SYSTEM_PROMPT
    assert attributes["llm.input_messages.1.message.content"] == "What is in week 3?"
    assert attributes["llm.output_messages.0.message.content"] == "Week 3 covers recursion."
    assert attributes["input.value"] == "What is in week 3?"
    assert attributes["output.value"] == "Week 3 covers recursion."
    counts = [attributes[f"llm.token_count.{key}"] for key in ("prompt", "completion", "total")]
    assert counts == [12, 5, 17]


def test_converse_span(finished_spans, monkeypatch):
    monkeypatch.setattr(chat, "_client", FakeBedrock(usage={"inputTokens": 12, "outputTokens": 5, "totalTokens": 17}))
    call(event({"message": "What is in week 3?"}))
    assert_chat(llm_span(finished_spans, "bedrock.converse"))


def test_chat_completions_span(finished_spans, monkeypatch):
    monkeypatch.setattr(chat, "_mantle", FakeMantle(usage={"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}))
    monkeypatch.setenv("MODEL_API", "openai")
    call(event({"message": "What is in week 3?"}))
    span = llm_span(finished_spans, "bedrock-mantle.chat_completions")
    assert_chat(span)
    assert span.attributes["llm.system"] == "openai"


def test_no_usage_means_no_token_counts(finished_spans, bedrock):
    call(event({"message": "hi"}))
    assert "llm.token_count.total" not in llm_span(finished_spans, "bedrock.converse").attributes


def test_failed_call_is_an_error_span(finished_spans, monkeypatch):
    monkeypatch.setattr(chat, "_client", FakeBedrock(error=AwsError("ThrottlingException")))
    call(event({"message": "hi"}))
    assert llm_span(finished_spans, "bedrock.converse").status.status_code == StatusCode.ERROR


# --- Exporter headers: none for Phoenix, the space ID and the API key from SSM for Arize ---


class FakeSsm:
    def __init__(self, error=None):
        self.error, self.calls = error, []

    def get_parameter(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return {"Parameter": {"Value": "secret-key"}}


@pytest.fixture
def tracing_env(monkeypatch):
    monkeypatch.delenv("TRACING_SPACE_ID", raising=False)
    monkeypatch.delenv("TRACING_API_KEY_PARAMETER", raising=False)
    return monkeypatch


def test_phoenix_needs_no_headers(tracing_env):
    ssm = FakeSsm()
    assert chat.tracing_headers(ssm) == {}
    assert ssm.calls == []


def test_arize_headers_come_from_the_env_and_ssm(tracing_env):
    tracing_env.setenv("TRACING_SPACE_ID", "space-123")
    tracing_env.setenv("TRACING_API_KEY_PARAMETER", "/rag-aws/arize-api-key")
    ssm = FakeSsm()
    assert chat.tracing_headers(ssm) == {"arize-space-id": "space-123", "authorization": "secret-key"}
    assert ssm.calls == [{"Name": "/rag-aws/arize-api-key", "WithDecryption": True}]


def test_unreadable_api_key_turns_tracing_off(tracing_env, capsys):
    tracing_env.setenv("TRACING_SPACE_ID", "space-123")
    tracing_env.setenv("TRACING_API_KEY_PARAMETER", "/rag-aws/arize-api-key")
    assert chat.tracing_headers(FakeSsm(error=AwsError("AccessDeniedException"))) is None
    assert "Tracing off" in capsys.readouterr().out
