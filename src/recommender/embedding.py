"""Sentence embeddings with all-MiniLM-L6-v2 in ONNX Runtime: no PyTorch, same vectors.

The ONNX export in the model's own Hugging Face repository gives the same normalized mean-pooled
vectors as sentence-transformers (cosine ≥ 0.9999999 on the AniList chunks), so an index built
with either stays valid. The files are pinned by revision and downloaded once into a model folder:
the container image bakes them in at build time.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Only for annotations: the image build runs this file on its own to download the model.
    from .progress import Progress

REPOSITORY = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
FILES = ("onnx/model.onnx", "tokenizer.json")
# sentence-transformers truncates this model's input at 256 word pieces.
MAX_SEQ_LENGTH = 256


def default_model_dir() -> Path:
    if folder := os.environ.get("EMBEDDING_MODEL_DIR"):
        return Path(folder)
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "rag-in-production" / "all-MiniLM-L6-v2" / REVISION


def download_model(folder: Path) -> Path:
    """Fetch the pinned model files into folder, if they aren't there yet."""
    for name in FILES:
        target = folder / name
        if target.is_file() and target.stat().st_size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as file:
            while block := response.read(1 << 20):
                file.write(block)
        partial.replace(target)
    return folder


class OnnxEmbedder:
    """Normalized MiniLM embeddings, loaded on first use."""

    max_seq_length = MAX_SEQ_LENGTH

    def __init__(self, model_dir: Path | None = None, batch_size: int = 64):
        self.model_dir = Path(model_dir) if model_dir else default_model_dir()
        self.batch_size = batch_size
        self._session = self._tokenizer = None

    def load(self):
        if self._session is None:
            import onnxruntime
            from tokenizers import Tokenizer

            if not (self.model_dir / FILES[0]).is_file():
                download_model(self.model_dir)
            tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
            tokenizer.enable_truncation(MAX_SEQ_LENGTH)
            tokenizer.enable_padding()
            options = onnxruntime.SessionOptions()
            # One Lambda vCPU per 1769 MB; more threads than cores only adds contention.
            options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                         if p in onnxruntime.get_available_providers()]
            self._session = onnxruntime.InferenceSession(str(self.model_dir / FILES[0]), options, providers=providers)
            self._inputs = {i.name for i in self._session.get_inputs()}
            self._tokenizer = tokenizer
        return self

    @property
    def device(self) -> str:
        return "cuda" if "CUDAExecutionProvider" in self.load()._session.get_providers() else "cpu"

    def encode(self, texts: list[str]) -> np.ndarray:
        self.load()
        encoded = self._tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self._session.run(None, feeds)[0]
        weights = mask[..., None].astype(np.float32)
        vectors = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        return vectors.astype(np.float32)

    def embed(self, texts: list[str], progress: Progress | None = None) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        vectors = np.empty((len(texts), 384), dtype=np.float32)
        if progress:
            progress("Embedding chunks", 0, len(texts))
        for start in range(0, len(texts), self.batch_size):
            batch = self.encode(texts[start:start + self.batch_size])
            vectors[start:start + len(batch)] = batch
            if progress:
                progress("Embedding chunks", start + len(batch), len(texts))
        return vectors
