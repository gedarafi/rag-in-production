from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Validated on creation, also by dataclasses.replace; a bad value fails with the field's name."""
    root: Path
    documents: Path
    index: Path
    prompt: Path
    provider: Literal["ollama", "bedrock"]
    llm: str
    embedding: str
    ollama_url: str
    chunk_size: Annotated[int, Field(gt=0)]
    chunk_overlap: Annotated[int, Field(ge=0)]
    top_k: Annotated[int, Field(gt=0)]
    temperature: Annotated[float, Field(ge=0)]
    max_tokens: Annotated[int, Field(gt=0)]
    extensions: tuple[str, ...]
    tracing_endpoint: str
    tracing_project: str
    embedding_device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    embedding_batch_size: Annotated[int, Field(ge=1)] = 64
    embedding_cache: bool = True
    embedding_cache_version: str = "1"
    detail_top_k: Annotated[int, Field(ge=1, le=20)] = 6
    context_chars: Annotated[int, Field(ge=1000)] = 14000
    router_top_k: Annotated[int, Field(ge=2, le=12)] = 6
    context_window: Annotated[int, Field(ge=2048, le=131072)] = 8192
    agent_max_steps: Annotated[int, Field(ge=1, le=20)] = 8
    bedrock_region: str = "us-east-2"
    bedrock_model: str = "qwen.qwen3-32b"
    # A folder whose templates replace the packaged ones of the same name (src/recommender/prompt_library.py).
    prompt_overrides: Path | None = None
    # Every LLM call: attempts on throttling or a transient failure, the backoff between them, and the connect timeout.
    llm_max_attempts: Annotated[int, Field(ge=1, le=10)] = 4
    llm_backoff_seconds: Annotated[float, Field(ge=0)] = 1.0
    llm_max_backoff_seconds: Annotated[float, Field(ge=0)] = 20.0
    llm_connect_timeout: Annotated[float, Field(gt=0)] = 5.0

    @field_validator("ollama_url")
    @classmethod
    def _no_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("embedding_cache_version", mode="before")
    @classmethod
    def _text(cls, value) -> str:
        return str(value)

    @field_validator("extensions")
    @classmethod
    def _lowercase(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(extension.lower() for extension in value)


KNOWLEDGE_BASES = ("custom", "multidoc2dial", "anilist", "vndb")

# settings.toml (section, key) -> Settings field; `paths` and the knowledge base are resolved separately.
TOML_FIELDS = {
    ("models", "provider"): "provider", ("models", "llm"): "llm", ("models", "embedding"): "embedding",
    ("models", "ollama_url"): "ollama_url", ("models", "embedding_device"): "embedding_device",
    ("models", "embedding_batch_size"): "embedding_batch_size", ("models", "embedding_cache"): "embedding_cache",
    ("models", "embedding_cache_version"): "embedding_cache_version",
    ("bedrock", "region"): "bedrock_region", ("bedrock", "model_id"): "bedrock_model",
    ("retrieval", "chunk_size"): "chunk_size", ("retrieval", "chunk_overlap"): "chunk_overlap",
    ("retrieval", "top_k"): "top_k", ("retrieval", "detail_top_k"): "detail_top_k",
    ("retrieval", "context_chars"): "context_chars", ("retrieval", "router_top_k"): "router_top_k",
    ("retrieval", "agent_max_steps"): "agent_max_steps",
    ("generation", "context_window"): "context_window", ("generation", "temperature"): "temperature",
    ("generation", "max_tokens"): "max_tokens", ("files", "extensions"): "extensions",
    ("tracing", "endpoint"): "tracing_endpoint", ("tracing", "project"): "tracing_project",
    ("llm", "max_attempts"): "llm_max_attempts", ("llm", "backoff_seconds"): "llm_backoff_seconds",
    ("llm", "max_backoff_seconds"): "llm_max_backoff_seconds", ("llm", "connect_timeout"): "llm_connect_timeout",
}

# Settings the environment overrides, and the variables that do, the first one set winning.
ENVIRONMENT = {
    "provider": ("RAG_PROVIDER",), "llm": ("RAG_LLM_MODEL",), "embedding": ("RAG_EMBEDDING_MODEL",),
    "ollama_url": ("RAG_OLLAMA_URL",), "tracing_endpoint": ("RAG_PHOENIX_ENDPOINT",),
    "tracing_project": ("RAG_PHOENIX_PROJECT",), "bedrock_region": ("RAG_BEDROCK_REGION", "AWS_REGION"),
    "bedrock_model": ("RAG_BEDROCK_MODEL",),
}

DEFAULTS = {"tracing_endpoint": "", "tracing_project": "rag-in-production"}


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
    """settings.toml, then the environment on top; paths resolve against the file's folder."""
    config_path = Path(path).resolve()
    with config_path.open("rb") as file:
        raw = tomllib.load(file)

    values = dict(DEFAULTS)
    for (section, key), field in TOML_FIELDS.items():
        if key in raw.get(section, {}):
            values[field] = raw[section][key]
    for field, names in ENVIRONMENT.items():
        name = next((name for name in names if name in os.environ), None)
        if name is not None:
            values[field] = os.environ[name]

    root = config_path.parent
    paths = raw["paths"]
    selected = knowledge_base or raw.get("knowledge_base", {}).get("default", "custom")
    if selected not in KNOWLEDGE_BASES:
        raise ValueError(f"Unknown knowledge base: {selected}")
    documents, index = paths["documents"], paths["index"]
    if selected != "custom":
        documents, index = f"data/documents/{selected}", f"data/index/{selected}"
    overrides = paths.get("prompt_overrides")
    return Settings(root=root, documents=(root / documents).resolve(), index=(root / index).resolve(),
                    prompt=(root / paths["prompt"]).resolve(),
                    prompt_overrides=(root / overrides).resolve() if overrides else None, **values)
