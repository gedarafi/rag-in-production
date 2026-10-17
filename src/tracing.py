"""OpenInference traces of the RAG pipeline, sent to Phoenix when [tracing] endpoint is set."""

from __future__ import annotations

from opentelemetry import trace

from .config import Settings

# Without a registered provider this is a no-op tracer, so tracing off costs nothing.
tracer = trace.get_tracer("src.rag")

_registered = False


def setup_tracing(settings: Settings) -> None:
    """Send the pipeline's spans to Phoenix, once per process (the GUI builds one pipeline per app)."""
    global _registered
    if _registered or not settings.tracing_endpoint:
        return
    # Imported here so the exporter stack only loads when tracing is on.
    from phoenix.otel import register

    register(
        endpoint=settings.tracing_endpoint,
        project_name=settings.tracing_project,
        batch=True,
        verbose=False,
    )
    _registered = True
