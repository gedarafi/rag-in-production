"""OpenInference traces of the recommender, sent to Phoenix.

The spans go to the Phoenix OTLP endpoint from settings.toml [tracing] (or RAG_PHOENIX_ENDPOINT).
Without an endpoint the tracer records nothing, except inside capture_spans() (tests and evaluations).
On EC2 the endpoint is Arize AX, which also needs RAG_TRACING_SPACE_ID and the API key in the SSM
parameter RAG_TRACING_API_KEY_PARAMETER.

Traces replace debug output: each pipeline stage is a span (stage()), notable decisions are span
events (event()) and structured results are span attributes (annotate()).

Every span carries the code's release (RELEASES.md) in its OpenInference metadata, so Phoenix can
filter on it, e.g. `metadata["release"] == "v0.0.2"`; the trace resource has it as `service.version`.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
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
_release = ""
# Phoenix's REST API (the OTLP endpoint without /v1/traces), for annotations; "" for Arize or no tracing.
_phoenix_url = ""
_project = ""
# Whether the "user_feedback" annotation config is set up on the project; once per process.
_feedback_config = False


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
            resource = {"openinference.project.name": project, **({"service.version": _release} if _release else {})}
            _provider = TracerProvider(resource=Resource(resource))
            trace.set_tracer_provider(_provider)
    if _collector is None:
        _collector = _Collector()
        _provider.add_span_processor(_ReleaseTagger())
        _provider.add_span_processor(_collector)
    return _provider


def export_headers(ssm=None) -> dict[str, str] | None:
    """Arize's auth headers (none for Phoenix), or None if the API key can't be read."""
    headers = {}
    if space_id := os.environ.get("RAG_TRACING_SPACE_ID"):
        headers["arize-space-id"] = space_id
    if parameter := os.environ.get("RAG_TRACING_API_KEY_PARAMETER"):
        try:
            if ssm is None:
                import boto3
                from botocore.config import Config

                # EC2 sets only AWS_REGION, which boto3 doesn't read as its default region.
                ssm = boto3.client("ssm", region_name=os.environ.get("AWS_REGION") or None,
                                   config=Config(connect_timeout=2, read_timeout=2, retries={"max_attempts": 2}))
            headers["authorization"] = ssm.get_parameter(Name=parameter, WithDecryption=True)["Parameter"]["Value"]
        except Exception as err:
            # The recommender keeps answering without traces.
            print(f"Tracing off: could not read {parameter}: {type(err).__name__}: {err}", flush=True)
            return None
    return headers


def setup_tracing(endpoint: str, project: str = "rag-in-production", release: str = "") -> bool:
    """Export spans to the Phoenix (or Arize) endpoint, if set; True if spans are exported.

    release (a git tag such as "v0.0.2") is added to every span's metadata from now on.
    """
    global _release, _phoenix_url, _project
    _release = release or _release
    if not endpoint:
        return False
    if endpoint in _endpoints:
        # Each pipeline calls this; one exporter per endpoint, or every span would be sent twice.
        return True
    if (headers := export_headers()) is None:
        return False
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    exporter = OTLPSpanExporter(endpoint, headers=headers or None, timeout=3)
    _sdk_provider(project).add_span_processor(BatchSpanProcessor(exporter))
    _endpoints.add(endpoint)
    if not headers:
        # Arize (auth headers) annotates through its own API; there the rating stays in the log.
        _phoenix_url = endpoint.removesuffix("/").removesuffix("/v1/traces")
        _project = project
    return True


def flush() -> None:
    """Send the buffered spans, e.g. before a CLI process exits."""
    if _provider:
        # A slow collector costs at most a few seconds, never the answer.
        _provider.force_flush(timeout_millis=3000)


# Phoenix's own config for end-user feedback, which its UI shows on the span's Annotations panel.
FEEDBACK_CONFIG = {"name": "user_feedback", "type": "CATEGORICAL", "description": "User feedback labels for traces.",
                   "optimization_direction": "MAXIMIZE",
                   "values": [{"label": "positive", "score": 1}, {"label": "negative", "score": 0}]}


def _phoenix(method: str, path: str, body=None) -> None:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"{_phoenix_url}{path}", data, {"Content-Type": "application/json"},
                                     method=method)
    urllib.request.urlopen(request, timeout=3).close()


def _ensure_feedback_config() -> None:
    """Assign Phoenix's "user_feedback" config to the project (creating it if missing), so the UI shows the ratings."""
    global _feedback_config
    if _feedback_config:
        return
    assign = f"/v1/projects/{urllib.parse.quote(_project, safe='')}/annotation_configs/user_feedback"
    try:
        try:
            _phoenix("PUT", assign)
        except urllib.error.HTTPError as err:
            # 404: a Phoenix without the config (or before the project's first span); create it and retry.
            if err.code != 404:
                raise
            _phoenix("POST", "/v1/annotation_configs", FEEDBACK_CONFIG)
            _phoenix("PUT", assign)
        _feedback_config = True
    except Exception as err:
        # The annotation itself is still stored; only the UI panel misses it.
        print(f"Feedback config not set up in Phoenix: {type(err).__name__}: {err}", flush=True)


def send_feedback(span_id: str, rating: str, question_id: str) -> bool:
    """A user's 👍/👎 on an answer, as a HUMAN "user_feedback" annotation on its root span in Phoenix.

    The label is "positive" or "negative" (score 1 or 0), as in Phoenix's own user_feedback config.
    True if Phoenix took it. A new rating of the same question replaces the old one. Never raises.
    """
    if not (_phoenix_url and span_id and rating in ("up", "down")):
        return False
    # The span may still be in the exporter's buffer, and Phoenix drops annotations of unknown spans.
    flush()
    _ensure_feedback_config()
    label = "positive" if rating == "up" else "negative"
    body = {"data": [{"span_id": span_id, "name": "user_feedback", "annotator_kind": "HUMAN",
                      "result": {"label": label, "score": 1 if rating == "up" else 0},
                      "identifier": question_id, "metadata": {"question_id": question_id}}]}
    try:
        _phoenix("POST", "/v1/span_annotations?sync=true", body)
        return True
    except Exception as err:
        print(f"Feedback not sent to Phoenix: {type(err).__name__}: {err}", flush=True)
        return False


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


def span_id(span=None) -> str:
    """The current (or given) span's ID as Phoenix shows it, or "" without a trace."""
    context = (span or trace.get_current_span()).get_span_context()
    return format(context.span_id, "016x") if context.is_valid else ""


class _ReleaseTagger(SpanProcessor):
    """Sets each span's OpenInference metadata to {"release": ...} as the span starts.

    Metadata set later on a span (openinference's using_metadata(), or `metadata` in its attributes)
    replaces this; code that sets its own should include the release.
    """

    def on_start(self, span, parent_context=None):
        if _release:
            span.set_attribute("metadata", json.dumps({"release": _release}))


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
