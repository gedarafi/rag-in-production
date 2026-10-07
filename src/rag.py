from __future__ import annotations

import json

import numpy as np

from .config import Settings
from .documents import Chunk, load_chunks
from .providers import LocalModels


class RagPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.models = LocalModels(settings.embedding, settings.ollama_url, settings.llm)

    def ingest(self) -> int:
        chunks = load_chunks(
            self.settings.documents,
            self.settings.extensions,
            self.settings.chunk_size,
            self.settings.chunk_overlap,
        )
        if not chunks:
            raise RuntimeError(f"No supported documents found in {self.settings.documents}")
        vectors = self.models.embed([chunk.text for chunk in chunks])
        self.settings.index.mkdir(parents=True, exist_ok=True)
        np.save(self.settings.index / "vectors.npy", vectors)
        (self.settings.index / "chunks.json").write_text(
            json.dumps([chunk.__dict__ for chunk in chunks], ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return len(chunks)

    def ask(self, question: str) -> str:
        vectors_path = self.settings.index / "vectors.npy"
        chunks_path = self.settings.index / "chunks.json"
        if not vectors_path.exists() or not chunks_path.exists():
            self.ingest()
        vectors = np.load(vectors_path)
        chunks = [Chunk(**item) for item in json.loads(chunks_path.read_text(encoding="utf-8"))]
        scores = self.models.embed([question]) @ vectors.T
        selected = np.argsort(scores[0])[-self.settings.top_k :][::-1]
        context = "\n\n".join(f"[{chunks[index].source}]\n{chunks[index].text}" for index in selected)
        prompt = self.settings.prompt.read_text(encoding="utf-8").format(
            context=context, question=question
        )
        return self.models.answer(prompt, self.settings.temperature, self.settings.max_tokens)