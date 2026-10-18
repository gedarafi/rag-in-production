"""The recommender as a long-lived service: load a published bundle once, answer many questions."""
from __future__ import annotations

import json
from pathlib import Path

from .chunk_index import ChunkIndex
from .config import load_settings
from .rag import RagPipeline
from .vector_db import VectorDb
from ..tracing import stage


class Recommender:
    """The pipeline over one bundle folder (see artifacts.py), with the catalog and agent kept warm."""

    def __init__(self, root: Path, models=None):
        self.root = Path(root)
        self.settings = load_settings(self.root / "recommender.toml", "anilist")
        self.pipeline = RagPipeline(self.settings, readonly=True)
        if models is not None:
            self.pipeline.models = models
        # A published bundle's passages are in the vector database, in its version's table; a bundle
        # (or workspace) with its own chunk index answers from that.
        database, manifest = VectorDb.from_env(), self.root / "manifest.json"
        if database and manifest.is_file() and not ChunkIndex.exists(self.settings.index):
            version = json.loads(manifest.read_text(encoding="utf-8"))["version"]
            self.pipeline.passage_table = database.table(version)

    def warm(self) -> "Recommender":
        """Load the catalog, the agent and the embedding model before the first question."""
        with stage("warm_recommender"):
            if self.pipeline.anime_tools() is None:
                raise RuntimeError(f"No AniList catalog in {self.root}.")
            self.pipeline.models.embedder.load()
        return self

    def answer(self, question: str, history: list[dict] | None = None) -> str:
        return self.pipeline.ask(question, history or [])
