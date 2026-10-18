"""The recommender as a long-lived service: load a published bundle once, answer many questions."""
from __future__ import annotations

from pathlib import Path

from .config import load_settings
from .rag import RagPipeline
from ..tracing import stage


class Recommender:
    """The pipeline over one bundle folder (see artifacts.py), with the catalog and agent kept warm."""

    def __init__(self, root: Path, models=None):
        self.root = Path(root)
        self.settings = load_settings(self.root / "recommender.toml", "anilist")
        self.pipeline = RagPipeline(self.settings)
        if models is not None:
            self.pipeline.models = models

    def warm(self) -> "Recommender":
        """Load the catalog, the agent and the embedding model before the first question."""
        with stage("warm_recommender"):
            if self.pipeline.anime_tools() is None:
                raise RuntimeError(f"No AniList catalog in {self.root}.")
            self.pipeline.models.embedder.load()
        return self

    def answer(self, question: str, history: list[dict] | None = None) -> str:
        return self.pipeline.ask(question, history or [])
