"""The recommender's span helpers and the Phoenix exporter setup."""

from opentelemetry import trace

from src import tracing


def test_no_endpoint_means_no_export():
    assert tracing.setup_tracing("") is False


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
