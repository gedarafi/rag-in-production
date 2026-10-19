"""What the evaluations share: the pipeline's trace of each answer, as checkpoint events."""
from __future__ import annotations

from ..tracing import capture_spans


def span_summary(span) -> dict:
    """One finished span as a checkpoint event: what ran, how long, and whether it failed."""
    attributes = dict(span.attributes or {})
    kind = attributes.get("openinference.span.kind", "").lower()
    event = {"stage": "tool" if kind == "tool" else span.name, "kind": kind,
             "status": "error" if span.status.status_code.name == "ERROR" else "completed",
             "elapsed_seconds": round((span.end_time - span.start_time) / 1e9, 6)}
    if kind == "tool":
        event["tool"] = span.name
    if span.status.description:
        event["error"] = span.status.description
    for record in span.events:
        if record.name not in ("exception",):
            event.setdefault("events", []).append({"name": record.name, **dict(record.attributes or {})})
    return event


def traced_asker(pipeline, events: list):
    """pipeline.ask for one independent question, appending its span summaries to events, even on failure."""
    def ask(question: str) -> str:
        with capture_spans("recommender-eval") as spans:
            try:
                return pipeline.ask(question, history=[])
            finally:
                events.extend(span_summary(span) for span in spans)
    return ask
