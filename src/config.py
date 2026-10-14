from __future__ import annotations

import os
import re
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
    embedding_device: str = "auto"
    embedding_batch_size: int = 64
    embedding_cache: bool = True
    embedding_cache_version: str = "1"
    detail_top_k: int = 6
    context_chars: int = 14000
    router_top_k: int = 6
    context_window: int = 8192
    agent_max_steps: int = 8


KNOWLEDGE_BASES = ("custom", "multidoc2dial", "anilist", "vndb")


def select_knowledge_base(path: str | Path, name: str) -> None:
    """Persist a CLI selection while retaining other settings and comments."""
    load_settings(path, name)
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    section = re.search(r"(?ms)^\[knowledge_base\][^\n]*\n(?P<body>.*?)(?=^\[|\Z)", text)
    if section:
        body = section["body"]
        pattern = r"(?m)^(\s*default\s*=\s*)(?:\"[^\"]*\"|'[^']*')"
        if re.search(pattern, body):
            body = re.sub(pattern, lambda m: m[1] + f'"{name}"', body, count=1)
        else:
            body = f'default = "{name}"\n' + body
        text = text[:section.start("body")] + body + text[section.end("body"):]
    else:
        text = text.rstrip() + f'\n\n[knowledge_base]\ndefault = "{name}"\n'
    # Validate the resulting TOML before changing the file.
    tomllib.loads(text)
    config_path.write_text(text, encoding="utf-8")


def load_settings(path: str | Path = "settings.toml", knowledge_base: str | None = None) -> Settings:
    config_path = Path(path).resolve()
    with config_path.open("rb") as file:
        raw = tomllib.load(file)

    paths = raw["paths"]
    models = raw["models"]
    retrieval = raw["retrieval"]
    if not 1 <= int(retrieval.get('agent_max_steps', 8)) <= 20:
        raise ValueError('agent_max_steps must be between 1 and 20')
    if not 2 <= int(retrieval.get("router_top_k", 6)) <= 12:
        raise ValueError("router_top_k must be between 2 and 12")
    if not 1 <= int(retrieval.get("detail_top_k", 6)) <= 20:
        raise ValueError("detail_top_k must be between 1 and 20")
    if int(retrieval.get("context_chars", 14000)) < 1000:
        raise ValueError("context_chars must be at least 1000")
    generation = raw["generation"]
    if not 2048 <= int(generation.get('context_window', 8192)) <= 131072:
        raise ValueError('context_window must be between 2048 and 131072 tokens')
    device = models.get("embedding_device", "auto")
    batch_size = int(models.get("embedding_batch_size", 64))
    if device not in ("auto", "cuda", "mps", "cpu"):
        raise ValueError("embedding_device must be auto, cuda, mps or cpu")
    if batch_size < 1:
        raise ValueError("embedding_batch_size must be positive")
    root = config_path.parent
    selected = knowledge_base or raw.get("knowledge_base", {}).get("default", "custom")
    if selected not in KNOWLEDGE_BASES:
        raise ValueError(f"Unknown knowledge base: {selected}")
    documents = paths["documents"]
    index = paths["index"]
    if selected != "custom":
        documents = f"data/documents/{selected}"
        index = f"data/index/{selected}"

    return Settings(
        root=root,
        documents=(root / documents).resolve(),
        index=(root / index).resolve(),
        prompt=(root / paths["prompt"]).resolve(),
        provider=models["provider"],
        llm=os.getenv("RAG_LLM_MODEL", models["llm"]),
        embedding=os.getenv("RAG_EMBEDDING_MODEL", models["embedding"]),
        ollama_url=os.getenv("RAG_OLLAMA_URL", models["ollama_url"]).rstrip("/"),
        chunk_size=int(retrieval["chunk_size"]),
        chunk_overlap=int(retrieval["chunk_overlap"]),
        top_k=int(retrieval["top_k"]),
        detail_top_k=int(retrieval.get("detail_top_k", 6)),
        context_chars=int(retrieval.get("context_chars", 14000)),
        router_top_k=int(retrieval.get("router_top_k", 6)),
        temperature=float(generation["temperature"]),
        max_tokens=int(generation["max_tokens"]),
        context_window=int(generation.get('context_window', 8192)),
        agent_max_steps=int(retrieval.get('agent_max_steps', 8)),
        extensions=tuple(extension.lower() for extension in raw["files"]["extensions"]),
        embedding_device=device,
        embedding_batch_size=batch_size,
        embedding_cache=bool(models.get("embedding_cache", True)),
        embedding_cache_version=str(models.get("embedding_cache_version", "1")),
    )
