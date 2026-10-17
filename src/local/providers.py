from __future__ import annotations

import requests
from sentence_transformers import SentenceTransformer

from .tracing import tracer


class LocalModels:
    def __init__(self, embedding_model: str, ollama_url: str, llm: str) -> None:
        self.embedder = SentenceTransformer(embedding_model)
        self.ollama_url = ollama_url
        self.llm = llm

    def embed(self, texts: list[str]):
        return self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    @tracer.llm
    def answer(self, prompt: str, temperature: float, max_tokens: int) -> str:
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.llm,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
                timeout=300,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise RuntimeError(
                f"Could not reach Ollama at {self.ollama_url}. Start Ollama and pull '{self.llm}'."
            ) from error
        return response.json()["response"].strip()