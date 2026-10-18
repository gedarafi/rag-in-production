"""Content-addressed embeddings with committed batch checkpoints in SQLite, and an in-memory memo."""
from __future__ import annotations

import hashlib
import sqlite3
from collections import OrderedDict
from contextlib import closing
from pathlib import Path

import numpy as np

from .progress import Progress


def cached_embed(texts: list[str], models, path: Path, namespace: str, progress: Progress | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE IF NOT EXISTS embeddings (namespace TEXT, hash TEXT, vector BLOB, PRIMARY KEY(namespace, hash))")
        vectors = None
        completed = 0
        reused = 0
        # Bound cache lookup and model work to small batches; commit each batch.
        for start in range(0, len(texts), 256):
            batch = texts[start:start + 256]
            keys = [hashlib.sha256(text.encode('utf-8')).hexdigest() for text in batch]
            unique = dict(zip(keys, batch))
            placeholders = ','.join('?' for _ in unique)
            found = {key: np.frombuffer(blob, dtype=np.float32) for key, blob in db.execute(
                f"SELECT hash, vector FROM embeddings WHERE namespace=? AND hash IN ({placeholders})",
                [namespace, *unique])}
            reused += sum(key in found for key in keys)
            missing = [key for key in unique if key not in found]
            if missing:
                if progress:
                    progress("Embedding chunks (cache + checkpoints)", completed, len(texts))
                encoded = np.asarray(models.embed([unique[key] for key in missing]), dtype=np.float32)
                if encoded.ndim != 2 or len(encoded) != len(missing) or not np.isfinite(encoded).all():
                    raise ValueError("Embedding model returned invalid vectors")
                for key, vector in zip(missing, encoded):
                    found[key] = vector
                db.executemany("INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)",
                               [(namespace, key, found[key].tobytes()) for key in missing])
                db.commit()
            if vectors is None:
                vectors = np.empty((len(texts), len(found[keys[0]])), dtype=np.float32)
            for offset, key in enumerate(keys):
                vectors[start + offset] = found[key]
            completed += len(batch)
            if progress:
                progress("Embedding chunks (cache + checkpoints)", completed, len(texts))
        if progress:
            progress("Cached chunks reused", reused, len(texts))
    return vectors


class EmbeddingMemo:
    """Embeds each text once per process: the constant attribute, operator and field descriptions,
    and a question that several stages of one request compare. Least recently used texts go first."""

    def __init__(self, models, size: int = 4096):
        # models.embed is looked up on each call, so a replaced embed method takes effect.
        self.models, self.size = models, size
        self.vectors: OrderedDict[str, np.ndarray] = OrderedDict()

    def __call__(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return self.models.embed([])
        missing = [text for text in dict.fromkeys(texts) if text not in self.vectors]
        if missing:
            encoded = np.asarray(self.models.embed(missing), dtype=np.float32)
            if encoded.ndim != 2 or len(encoded) != len(missing):
                raise ValueError("Embedding model returned invalid vectors")
            self.vectors.update(zip(missing, encoded))
        for text in texts:
            self.vectors.move_to_end(text)
        vectors = np.stack([self.vectors[text] for text in texts])
        while len(self.vectors) > self.size:
            self.vectors.popitem(last=False)
        return vectors
