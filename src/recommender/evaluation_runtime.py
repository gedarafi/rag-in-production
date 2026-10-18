"""What the evaluations share: the extractor model and the pipeline's trace of each answer."""
from __future__ import annotations

import os

from .bedrock import BedrockChat
from ..tracing import capture_spans


def extractor_chat(model: str | None = None) -> BedrockChat:
    """The model that parses answers: MODEL_ID unless the evaluation names another one."""
    chat = BedrockChat.from_env()
    if model:
        chat.model = model
    return chat


def extractor_model(model: str | None = None) -> str:
    return model or os.environ.get("MODEL_ID", "")


def complete_json(chat: BedrockChat, name: str, messages: list[dict], schema: dict, max_tokens: int = 1400) -> str:
    """The model's raw JSON reply; messages may start with a system message, as the extraction prompts do.

    The extractors validate and retry themselves, so this does no correction round of its own.
    """
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else "Return JSON."
    rest = messages[1:] if messages and messages[0]["role"] == "system" else messages
    return chat.json(name, system, rest, schema, max_tokens=max_tokens, validate=False)


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
