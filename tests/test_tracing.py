"""Exporter headers (none for Phoenix; the space ID and the API key from SSM for Arize) and span helpers."""

from opentelemetry import trace

from src import tracing
from tests.fakes import AwsError


class FakeSsm:
    def __init__(self, error=None):
        self.error, self.calls = error, []

    def get_parameter(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return {"Parameter": {"Value": "secret-key"}}


def test_phoenix_needs_no_headers(monkeypatch):
    monkeypatch.delenv("TRACING_SPACE_ID", raising=False)
    monkeypatch.delenv("TRACING_API_KEY_PARAMETER", raising=False)
    ssm = FakeSsm()
    assert tracing.tracing_headers(ssm) == {}
    assert ssm.calls == []


def test_arize_headers_come_from_the_env_and_ssm(monkeypatch):
    monkeypatch.setenv("TRACING_SPACE_ID", "space-123")
    monkeypatch.setenv("TRACING_API_KEY_PARAMETER", "/rag-aws/arize-api-key")
    ssm = FakeSsm()
    assert tracing.tracing_headers(ssm) == {"arize-space-id": "space-123", "authorization": "secret-key"}
    assert ssm.calls == [{"Name": "/rag-aws/arize-api-key", "WithDecryption": True}]


def test_unreadable_api_key_turns_tracing_off(monkeypatch, capsys):
    monkeypatch.setenv("TRACING_SPACE_ID", "space-123")
    monkeypatch.setenv("TRACING_API_KEY_PARAMETER", "/rag-aws/arize-api-key")
    assert tracing.tracing_headers(FakeSsm(error=AwsError("AccessDeniedException"))) is None
    assert "Tracing off" in capsys.readouterr().out


def test_no_endpoint_means_no_export(monkeypatch):
    monkeypatch.delenv("TRACING_ENDPOINT", raising=False)
    assert tracing.setup_tracing() is False


def test_stages_events_and_attributes_are_captured():
    with tracing.capture_spans() as spans:
        with tracing.stage("outer", plan={"steps": [1, 2]}):
            tracing.event("decision", scope="in_domain", count=2)
            tracing.annotate(**{"recommender.calls": 3})
            try:
                with tracing.stage("inner"):
                    raise ValueError("boom")
            except ValueError:
                pass
    inner, outer = spans
    assert inner.parent.span_id == outer.context.span_id
    assert inner.status.status_code.name == "ERROR"
    assert outer.attributes["plan"] == '{"steps": [1, 2]}'
    assert outer.attributes["recommender.calls"] == 3
    (decision,) = outer.events
    assert dict(decision.attributes) == {"scope": "in_domain", "count": 2}


def test_long_attributes_are_cut():
    assert len(tracing.jsonable("x" * 20000)) == tracing.MAX_ATTRIBUTE_CHARS + 1


def test_context_travels_through_a_carrier():
    with tracing.capture_spans() as spans:
        with tracing.stage("sender"):
            carrier = tracing.inject()
            sender = tracing.trace_id()
        with trace.get_tracer("test").start_as_current_span("receiver", context=tracing.extract(carrier)):
            receiver = tracing.trace_id()
    assert "traceparent" in carrier
    assert sender == receiver and len(sender) == 32
    assert len(spans) == 2


def test_open_and_close_span_make_it_current():
    with tracing.capture_spans() as spans:
        span, token = tracing.open_span("tool", "tool", **{"tool.name": "get_anime"})
        assert trace.get_current_span() is span
        tracing.close_span(span, token)
    assert spans[0].attributes["openinference.span.kind"] == "TOOL"
    assert trace.get_current_span() is not span
