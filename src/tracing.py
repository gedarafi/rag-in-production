"""OpenInference traces shared by the Lambdas and the recommender CLI.

The spans go to TRACING_ENDPOINT: Phoenix locally (LocalStack and the CLI), Arize AX on AWS, which also
needs TRACING_SPACE_ID and the API key in the SSM parameter TRACING_API_KEY_PARAMETER. Without an
endpoint the tracer records nothing, except inside capture_spans() (tests and evaluations).

Traces replace debug output: each pipeline stage is a span (stage()), notable decisions are span
events (event()) and structured results are span attributes (annotate()).
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager

from openinference.instrumentation import OITracer, TraceConfig
from opentelemetry import propagate, trace
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import Status, StatusCode

# Attribute values longer than this are cut, so a large tool result can't blow the span size limits.
MAX_ATTRIBUTE_CHARS = 8000

tracer = OITracer(trace.get_tracer("rag-in-production"), config=TraceConfig())

_provider = None
_collector = None


def tracing_headers(ssm=None) -> dict[str, str] | None:
    """Arize's auth headers (none for Phoenix), or None if the API key can't be read."""
    headers = {}
    if space_id := os.environ.get("TRACING_SPACE_ID"):
        headers["arize-space-id"] = space_id
    if parameter := os.environ.get("TRACING_API_KEY_PARAMETER"):
        try:
            if ssm is None:
                import boto3
                from botocore.config import Config

                ssm = boto3.client("ssm", config=Config(connect_timeout=2, read_timeout=2, retries={"max_attempts": 2}))
            headers["authorization"] = ssm.get_parameter(Name=parameter, WithDecryption=True)["Parameter"]["Value"]
        except Exception as err:
            # The app keeps working without traces.
            print(f"Tracing off: could not read {parameter}: {type(err).__name__}: {err}")
            return None
    return headers


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


def setup_tracing(default_project: str = "rag-in-production") -> bool:
    """Export spans to TRACING_ENDPOINT, if set; True if spans are exported."""
    endpoint = os.environ.get("TRACING_ENDPOINT")
    if not endpoint or (headers := tracing_headers()) is None:
        return False
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = _sdk_provider(os.environ.get("TRACING_PROJECT", default_project))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint, headers=headers or None, timeout=3)))
    return True


def flush() -> None:
    """Send the buffered spans; a Lambda is frozen after it returns, so call this before."""
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
    """The current (or given) span's trace ID as Phoenix and Arize show it, or "" without a trace."""
    context = (span or trace.get_current_span()).get_span_context()
    return format(context.trace_id, "032x") if context.is_valid else ""


# --- Propagation across SQS ---


def inject() -> dict[str, str]:
    """The current trace context (W3C traceparent), to send along with a message."""
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return carrier


def extract(carrier: dict[str, str] | None):
    """The sender's trace context, so the receiver's spans join the sender's trace."""
    return propagate.extract(carrier or {})


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
