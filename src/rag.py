from __future__ import annotations

import json

import numpy as np
from openinference.semconv.trace import (
    DocumentAttributes,
    EmbeddingAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

from .config import Settings
from .documents import Chunk, load_chunks
from .providers import LocalModels
from .tracing import setup_tracing, tracer

KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
CHAIN = OpenInferenceSpanKindValues.CHAIN.value
EMBEDDING = OpenInferenceSpanKindValues.EMBEDDING.value
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER.value


class RagPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        setup_tracing(settings)
        self.models = LocalModels(settings.embedding, settings.ollama_url, settings.llm)

    def ingest(self) -> int:
        with tracer.start_as_current_span("rag.ingest", attributes={KIND: CHAIN}) as span:
            span.set_attribute(SpanAttributes.INPUT_VALUE, str(self.settings.documents))
            chunks = load_chunks(
                self.settings.documents,
                self.settings.extensions,
                self.settings.chunk_size,
                self.settings.chunk_overlap,
            )
            if not chunks:
                raise RuntimeError(f"No supported documents found in {self.settings.documents}")
            with tracer.start_as_current_span("embed_chunks", attributes={KIND: EMBEDDING}) as embed_span:
                embed_span.set_attribute(SpanAttributes.EMBEDDING_MODEL_NAME, self.settings.embedding)
                embed_span.set_attribute("rag.chunk_count", len(chunks))
                vectors = self.models.embed([chunk.text for chunk in chunks])
            self.settings.index.mkdir(parents=True, exist_ok=True)
            np.save(self.settings.index / "vectors.npy", vectors)
            (self.settings.index / "chunks.json").write_text(
                json.dumps([chunk.__dict__ for chunk in chunks], ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, f"Indexed {len(chunks)} chunks")
            return len(chunks)

    def ask(self, question: str) -> str:
        with tracer.start_as_current_span("rag.ask", attributes={KIND: CHAIN}) as span:
            span.set_attribute(SpanAttributes.INPUT_VALUE, question)
            vectors_path = self.settings.index / "vectors.npy"
            chunks_path = self.settings.index / "chunks.json"
            if not vectors_path.exists() or not chunks_path.exists():
                self.ingest()
            with tracer.start_as_current_span("retrieve", attributes={KIND: RETRIEVER}) as retrieve_span:
                retrieve_span.set_attribute(SpanAttributes.INPUT_VALUE, question)
                vectors = np.load(vectors_path)
                chunks = [Chunk(**item) for item in json.loads(chunks_path.read_text(encoding="utf-8"))]
                with tracer.start_as_current_span("embed_question", attributes={KIND: EMBEDDING}) as embed_span:
                    embed_span.set_attribute(SpanAttributes.EMBEDDING_MODEL_NAME, self.settings.embedding)
                    embed_span.set_attribute(
                        f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_TEXT}", question
                    )
                    query = self.models.embed([question])
                scores = query @ vectors.T
                selected = np.argsort(scores[0])[-self.settings.top_k :][::-1]
                for rank, index in enumerate(selected):
                    prefix = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{rank}"
                    retrieve_span.set_attributes({
                        f"{prefix}.{DocumentAttributes.DOCUMENT_ID}": f"{chunks[index].source}#{index}",
                        f"{prefix}.{DocumentAttributes.DOCUMENT_CONTENT}": chunks[index].text,
                        f"{prefix}.{DocumentAttributes.DOCUMENT_SCORE}": float(scores[0][index]),
                        f"{prefix}.{DocumentAttributes.DOCUMENT_METADATA}": json.dumps(
                            {"source": chunks[index].source, "chunk": int(index)}
                        ),
                    })
            context = "\n\n".join(f"[{chunks[index].source}]\n{chunks[index].text}" for index in selected)
            template = self.settings.prompt.read_text(encoding="utf-8")
            prompt = template.format(context=context, question=question)
            answer = self.models.answer(
                prompt,
                self.settings.temperature,
                self.settings.max_tokens,
                template=(template, {"context": context, "question": question}),
            )
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, answer)
            return answer
