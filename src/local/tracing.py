"""Traces of the RAG pipeline, sent to Phoenix when [tracing] endpoint is set."""

from __future__ import annotations

from openinference.instrumentation import OITracer, TraceConfig
from opentelemetry import trace

from .config import Settings

# Decorators for the pipeline (@tracer.chain, @tracer.retriever, @tracer.llm).
# They record nothing until setup_tracing() sends the spans to Phoenix.
tracer = OITracer(trace.get_tracer("src.rag"), config=TraceConfig())


def setup_tracing(settings: Settings) -> None:
    if settings.tracing_endpoint:
        # Imported here so the exporter only loads when tracing is on.
        from phoenix.otel import register

        register(endpoint=settings.tracing_endpoint, project_name=settings.tracing_project, batch=True, verbose=False)
