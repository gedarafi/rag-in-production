"""The recommender's LLM instructions, kept as Jinja2 templates in src/recommender/prompts/.

Each pipeline step renders its own template; the code supplies only the data (JSON payloads, the
question, flags for the conditional parts). The templates ship inside the package, so a code release
always has the instructions its steps were written for. To change them without a code change, point
`prompt_overrides` in settings.toml at a folder: a file there replaces the packaged file of the same
name (a published bundle ships that folder with the settings).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from jinja2 import ChoiceLoader, Environment, FileSystemLoader, StrictUndefined

PACKAGED = Path(__file__).resolve().parent / "prompts"


class PromptLibrary:
    """Templates sent as written: no line folding, and the newline that ends the file is dropped
    (Jinja's default), so a prompt can end mid-sentence."""

    def __init__(self, overrides: Path | None = None) -> None:
        self.folders = [folder for folder in (overrides, PACKAGED) if folder is not None]
        self.environment = Environment(loader=ChoiceLoader([FileSystemLoader(f) for f in self.folders]),
                                       undefined=StrictUndefined, autoescape=False, keep_trailing_newline=False)

    def render(self, name: str, **values) -> str:
        return self.environment.get_template(name + ".j2").render(**values)

    @lru_cache(maxsize=None)
    def data(self, name: str):
        """A JSON file of the library, such as the tool descriptions."""
        path = next(folder / name for folder in self.folders if (folder / name).is_file() or folder == PACKAGED)
        return json.loads(path.read_text(encoding="utf-8"))


def library(overrides: Path | None = None) -> PromptLibrary:
    """One library per override folder (templates compile once)."""
    return _library(overrides)


@lru_cache(maxsize=None)
def _library(overrides: Path | None) -> PromptLibrary:
    return PromptLibrary(overrides)


def prompts_for(settings=None) -> PromptLibrary:
    """The packaged templates, with the settings' override folder (if any) in front."""
    return library(getattr(settings, "prompt_overrides", None))
