"""The recommender's span helpers and the Phoenix exporter setup."""

from opentelemetry import trace

from src import tracing


def test_no_endpoint_means_no_export():
    assert tracing.setup_tracing("") is False


class FakeSsm:
    def __init__(self, error=None):
        self.error, self.calls = error, []

    def get_parameter(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return {"Parameter": {"Value": "secret-key"}}


def test_phoenix_needs_no_headers(monkeypatch):
    monkeypatch.delenv("RAG_TRACING_SPACE_ID", raising=False)
    monkeypatch.delenv("RAG_TRACING_API_KEY_PARAMETER", raising=False)
    ssm = FakeSsm()
    assert tracing.export_headers(ssm) == {}
    assert ssm.calls == []


def test_arize_headers_come_from_the_env_and_ssm(monkeypatch):
    monkeypatch.setenv("RAG_TRACING_SPACE_ID", "space-123")
    monkeypatch.setenv("RAG_TRACING_API_KEY_PARAMETER", "/rag-aws/arize-api-key")
    ssm = FakeSsm()
    assert tracing.export_headers(ssm) == {"arize-space-id": "space-123", "authorization": "secret-key"}
    assert ssm.calls == [{"Name": "/rag-aws/arize-api-key", "WithDecryption": True}]


def test_unreadable_api_key_turns_tracing_off(monkeypatch, capsys):
    monkeypatch.setenv("RAG_TRACING_SPACE_ID", "space-123")
    monkeypatch.setenv("RAG_TRACING_API_KEY_PARAMETER", "/rag-aws/arize-api-key")
    assert tracing.export_headers(FakeSsm(error=PermissionError("AccessDenied"))) is None
    assert "Tracing off" in capsys.readouterr().out


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


def test_trace_id_is_the_current_trace():
    with tracing.capture_spans() as spans:
        with tracing.stage("outer"):
            current = tracing.trace_id()
    assert current == format(spans[0].context.trace_id, "032x")
    assert tracing.trace_id() == ""


def test_open_and_close_span_make_it_current():
    with tracing.capture_spans() as spans:
        span, token = tracing.open_span("tool", "tool", **{"tool.name": "get_anime"})
        assert trace.get_current_span() is span
        tracing.close_span(span, token)
    assert spans[0].attributes["openinference.span.kind"] == "TOOL"
    assert trace.get_current_span() is not span


def test_every_span_carries_the_release_in_its_metadata(monkeypatch):
    import json

    monkeypatch.setattr(tracing, "_release", "")
    assert tracing.setup_tracing("", release="v9.9.9") is False  # no export, but the release is kept
    with tracing.capture_spans() as spans:
        with tracing.stage("outer"), tracing.stage("inner", kind="llm"):
            pass
    assert {span.name: json.loads(span.attributes["metadata"]) for span in spans} == {
        "outer": {"release": "v9.9.9"}, "inner": {"release": "v9.9.9"}}


def test_span_id_is_the_current_span():
    with tracing.capture_spans() as spans:
        with tracing.stage("outer"):
            current = tracing.span_id()
    assert current == format(spans[0].context.span_id, "016x")
    assert tracing.span_id() == ""


class FakeResponse:
    def close(self):
        pass


def phoenix(monkeypatch, missing_config=False, down=False):
    """A fake Phoenix REST API; returns the (method, url, body) of every call."""
    import json
    import urllib.error

    calls = []

    def urlopen(request, timeout):
        calls.append((request.get_method(), request.full_url, json.loads(request.data) if request.data else None))
        if down:
            raise OSError("refused")
        if missing_config and len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)
        return FakeResponse()

    monkeypatch.setattr(tracing, "_phoenix_url", "http://phoenix:6006")
    monkeypatch.setattr(tracing, "_project", "recommender")
    monkeypatch.setattr(tracing, "_feedback_config", False)
    monkeypatch.setattr(tracing.urllib.request, "urlopen", urlopen)
    return calls


ASSIGN = ("PUT", "http://phoenix:6006/v1/projects/recommender/annotation_configs/user_feedback", None)


def test_feedback_is_a_human_annotation_on_the_span(monkeypatch):
    calls = phoenix(monkeypatch)
    assert tracing.send_feedback("ab" * 8, "down", "q" * 32) is True
    assert tracing.send_feedback("ab" * 8, "up", "q" * 32) is True
    annotation = ("POST", "http://phoenix:6006/v1/span_annotations?sync=true", {"data": [{
        "span_id": "ab" * 8, "name": "user_feedback", "annotator_kind": "HUMAN",
        "result": {"label": "negative", "score": 0}, "identifier": "q" * 32, "metadata": {"question_id": "q" * 32}}]})
    # The project gets Phoenix's user_feedback config once, so the UI's Annotations panel shows the ratings.
    assert calls[:2] == [ASSIGN, annotation]
    assert [call[0] for call in calls[2:]] == ["POST"]
    assert calls[2][2]["data"][0]["result"] == {"label": "positive", "score": 1}


def test_a_missing_feedback_config_is_created(monkeypatch):
    calls = phoenix(monkeypatch, missing_config=True)
    assert tracing.send_feedback("ab" * 8, "up", "q") is True
    assert calls[0] == ASSIGN and calls[2] == ASSIGN
    assert calls[1] == ("POST", "http://phoenix:6006/v1/annotation_configs", tracing.FEEDBACK_CONFIG)
    assert calls[3][1].endswith("/v1/span_annotations?sync=true")


def test_feedback_needs_phoenix_a_span_and_a_rating(monkeypatch):
    calls = phoenix(monkeypatch)
    assert tracing.send_feedback("", "up", "q") is False
    assert tracing.send_feedback("ab" * 8, "cleared", "q") is False
    monkeypatch.setattr(tracing, "_phoenix_url", "")
    assert tracing.send_feedback("ab" * 8, "up", "q") is False  # no tracing, or Arize
    assert calls == []


def test_feedback_errors_are_only_printed(monkeypatch, capsys):
    phoenix(monkeypatch, down=True)
    assert tracing.send_feedback("ab" * 8, "up", "q") is False
    out = capsys.readouterr().out
    assert "Feedback config not set up in Phoenix: OSError: refused" in out
    assert "Feedback not sent to Phoenix: OSError: refused" in out


def test_only_phoenix_gets_annotations(monkeypatch):
    monkeypatch.setattr(tracing, "_phoenix_url", "")
    monkeypatch.setattr(tracing, "_endpoints", set())
    monkeypatch.setattr(tracing, "_sdk_provider", lambda project: type("P", (), {"add_span_processor": lambda *_: None})())
    monkeypatch.setattr(tracing, "export_headers", lambda: {"arize-space-id": "s"})
    tracing.setup_tracing("https://otlp.arize.com/v1/traces")
    assert tracing._phoenix_url == ""
    monkeypatch.setattr(tracing, "export_headers", lambda: {})
    monkeypatch.setattr(tracing, "_project", "")
    tracing.setup_tracing("http://localhost:6006/v1/traces", "recommender")
    assert (tracing._phoenix_url, tracing._project) == ("http://localhost:6006", "recommender")
