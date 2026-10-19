"""OpenInference traces of the recommender, sent to Phoenix.

The spans go to the Phoenix OTLP endpoint from settings.toml [tracing] (or RAG_PHOENIX_ENDPOINT).
Without an endpoint the tracer records nothing, except inside capture_spans() (tests and evaluations).

Traces replace debug output: each pipeline stage is a span (stage()), notable decisions are span
events (event()) and structured results are span attributes (annotate()).
"""

from __future__ import annotations

import json
from contextlib import contextmanager

from openinference.instrumentation import OITracer, TraceConfig
from opentelemetry import trace
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import Status, StatusCode

# Attribute values longer than this are cut, so a large tool result can't blow the span size limits.
MAX_ATTRIBUTE_CHARS = 8000

tracer = OITracer(trace.get_tracer("rag-in-production"), config=TraceConfig())

_provider = None
_collector = None
_endpoints: set[str] = set()


def _sdk_provider(project: str):
    """The SDK provider every span goes through, created once per process."""
    global _provider, _collector
    if _provider is None:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        current = trace.get_tracer_provider()
        if isinstance(current, TracerProvider):
            # A test or another module installed one already; reuse it.
            _provider = current
        else:
            _provider = TracerProvider(resource=Resource({"openinference.project.name": project}))
            trace.set_tracer_provider(_provider)
    if _collector is None:
        _collector = _Collector()
        _provider.add_span_processor(_collector)
    return _provider


def setup_tracing(endpoint: str, project: str = "rag-in-production") -> bool:
    """Export spans to the Phoenix endpoint, if set; True if spans are exported."""
    if not endpoint:
        return False
    if endpoint in _endpoints:
        # Each pipeline calls this; one exporter per endpoint, or every span would be sent twice.
        return True
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _sdk_provider(project).add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint, timeout=3)))
    _endpoints.add(endpoint)
    return True


def flush() -> None:
    """Send the buffered spans, e.g. before a CLI process exits."""
    if _provider:
        # A slow collector costs at most a few seconds, never the answer.
        _provider.force_flush(timeout_millis=3000)


# --- Spans, events and attributes ---


def jsonable(value):
    """A span attribute value: primitives as they are, anything else as (truncated) JSON."""
    if isinstance(value, (bool, int, float)):
        return value
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= MAX_ATTRIBUTE_CHARS else text[:MAX_ATTRIBUTE_CHARS] + "…"


def _attributes(values: dict) -> dict:
    return {key: jsonable(value) for key, value in values.items() if value is not None}


@contextmanager
def stage(name: str, kind: str = "chain", **attributes):
    """A child span for one pipeline stage; an exception marks it as failed and propagates."""
    with tracer.start_as_current_span(name, openinference_span_kind=kind) as span:
        if attributes:
            span.set_attributes(_attributes(attributes))
        yield span


def open_span(name: str, kind: str, **attributes):
    """Start a span and make it current until close_span(), for loops where `with` doesn't fit."""
    from opentelemetry import context

    span = tracer.start_span(name, openinference_span_kind=kind)
    if attributes:
        span.set_attributes(_attributes(attributes))
    return span, context.attach(trace.set_span_in_context(span))


def close_span(span, token) -> None:
    from opentelemetry import context

    context.detach(token)
    span.end()


def event(name: str, **attributes) -> None:
    """Record a decision on the current span, e.g. which history policy was chosen."""
    trace.get_current_span().add_event(name, _attributes(attributes))


def annotate(**attributes) -> None:
    """Attach structured results (plan, scope, selected context) to the current span."""
    trace.get_current_span().set_attributes(_attributes(attributes))


def fail(span, error: BaseException | str) -> None:
    """Mark a span as failed without raising, for errors the pipeline handles itself."""
    span.set_status(Status(StatusCode.ERROR, str(error)))
    if isinstance(error, BaseException):
        span.record_exception(error)


def trace_id(span=None) -> str:
    """The current (or given) span's trace ID as Phoenix shows it, or "" without a trace."""
    context = (span or trace.get_current_span()).get_span_context()
    return format(context.trace_id, "032x") if context.is_valid else ""


# --- Capturing spans in-process (tests and evaluations) ---


class _Collector(SpanProcessor):
    """A span processor that hands finished spans to every active capture_spans() block."""

    def __init__(self):
        self.sinks: list[list] = []

    def on_end(self, span):
        for sink in self.sinks:
            sink.append(span)


@contextmanager
def capture_spans(project: str = "rag-in-production"):
    """Collect the spans finished inside the block, whether or not they are also exported."""
    _sdk_provider(project)
    sink: list = []
    _collector.sinks.append(sink)
    try:
        yield sink
    finally:
        _collector.sinks.remove(sink)
