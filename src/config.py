from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root: Path
    documents: Path
    index: Path
    prompt: Path
    provider: str
    llm: str
    embedding: str
    ollama_url: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    temperature: float
    max_tokens: int
    extensions: tuple[str, ...]


def load_settings(path: str | Path = "settings.toml") -> Settings:
    config_path = Path(path).resolve()
    with config_path.open("rb") as file:
        raw = tomllib.load(file)

    paths = raw["paths"]
    models = raw["models"]
    retrieval = raw["retrieval"]
    generation = raw["generation"]
    root = config_path.parent

    return Settings(
        root=root,
        documents=(root / paths["documents"]).resolve(),
        index=(root / paths["index"]).resolve(),
        prompt=(root / paths["prompt"]).resolve(),
        provider=models["provider"],
        llm=os.getenv("RAG_LLM_MODEL", models["llm"]),
        embedding=os.getenv("RAG_EMBEDDING_MODEL", models["embedding"]),
        ollama_url=os.getenv("RAG_OLLAMA_URL", models["ollama_url"]).rstrip("/"),
        chunk_size=int(retrieval["chunk_size"]),
        chunk_overlap=int(retrieval["chunk_overlap"]),
        top_k=int(retrieval["top_k"]),
        temperature=float(generation["temperature"]),
        max_tokens=int(generation["max_tokens"]),
        extensions=tuple(extension.lower() for extension in raw["files"]["extensions"]),
    )