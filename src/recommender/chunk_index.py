"""The passage index: chunk texts in SQLite, their vectors in one float16 matrix, grouped by anime.

`recommender build` writes it (RagPipeline.ingest). The rows are sorted by anime_id, so each anime's
chunks are one contiguous slice of the matrix and a chunk's id is its row: a search within some anime
scores only their slices, and only the k best chunks' texts are read back. float16 halves the
matrix (215k × 384: 165 MB instead of 330 MB); scores are computed in float32.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np

from .documents import Chunk

DATABASE = "chunks.sqlite3"
VECTORS = "chunk-vectors.npy"
# The earlier format (one JSON list and a float32 matrix in reading order); write_index removes it.
LEGACY = ("chunks.json", "vectors.npy")
BLOCK = 8192

SCHEMA = """
CREATE TABLE chunks(id INTEGER PRIMARY KEY, anime_id TEXT, source TEXT NOT NULL, entity_type TEXT,
 character_id TEXT, text TEXT NOT NULL);
CREATE TABLE anime_chunks(anime_id TEXT PRIMARY KEY, start INTEGER NOT NULL, stop INTEGER NOT NULL);
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def write_index(folder: Path, chunks: list[Chunk], vectors: np.ndarray) -> None:
    """Replace the index in folder with chunks and their vectors (row i embeds chunks[i])."""
    vectors = np.asarray(vectors)
    if vectors.ndim != 2 or len(vectors) != len(chunks):
        raise ValueError("Each chunk needs exactly one vector.")
    # A stable sort keeps each anime's chunks in reading order.
    order = np.array(sorted(range(len(chunks)), key=lambda i: chunks[i].anime_id or ""), dtype=np.int64)
    folder.mkdir(parents=True, exist_ok=True)
    database, matrix = folder / (DATABASE + ".tmp"), folder / (VECTORS + ".tmp")
    database.unlink(missing_ok=True)
    with closing(sqlite3.connect(database)) as db:
        db.executescript(SCHEMA)
        db.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?)",
                       ((row, c.anime_id, c.source, c.entity_type, c.character_id, c.text)
                        for row, c in enumerate(chunks[i] for i in order)))
        db.execute("INSERT INTO anime_chunks SELECT anime_id, MIN(id), MAX(id) + 1 FROM chunks "
                   "WHERE anime_id IS NOT NULL GROUP BY anime_id")
        db.executemany("INSERT INTO meta VALUES (?,?)", [("count", str(len(chunks))),
                                                         ("dimension", str(vectors.shape[1])), ("dtype", "float16")])
        db.commit()
    # Reorder in blocks: no second full-size float32 copy of the matrix.
    sorted_vectors = np.empty(vectors.shape, dtype=np.float16)
    for start in range(0, len(order), BLOCK):
        sorted_vectors[start:start + BLOCK] = vectors[order[start:start + BLOCK]]
    with matrix.open("wb") as file:
        np.save(file, sorted_vectors)
    matrix.replace(folder / VECTORS)
    database.replace(folder / DATABASE)
    for name in LEGACY:
        (folder / name).unlink(missing_ok=True)


class ChunkIndex:
    """Searches the index in a folder. The matrix stays memory-mapped: pages are read as scored."""

    def __init__(self, folder: Path):
        self.database = folder / DATABASE
        self.vectors = np.load(folder / VECTORS, mmap_mode="r")
        count = self._query("SELECT value FROM meta WHERE key='count'")
        if self.vectors.ndim != 2 or not count or int(count[0][0]) != len(self.vectors):
            raise ValueError(f"The chunk index in {folder} is incomplete; run `recommender build` again.")

    @staticmethod
    def exists(folder: Path) -> bool:
        return (folder / DATABASE).is_file() and (folder / VECTORS).is_file()

    @staticmethod
    def stamp(folder: Path) -> tuple:
        return tuple((folder / name).stat().st_mtime_ns for name in (DATABASE, VECTORS))

    def _query(self, sql: str, parameters=()) -> list[tuple]:
        with closing(sqlite3.connect(self.database)) as db:
            return db.execute(sql, parameters).fetchall()

    def spans(self, anime_ids: list[str] | None) -> list[tuple[int, int]]:
        """The [start, stop) row ranges to score: everything, or the given anime's chunks."""
        if anime_ids is None:
            return [(0, len(self.vectors))]
        ids = list(dict.fromkeys(str(i) for i in anime_ids))
        if not ids:
            return []
        return self._query(f"SELECT start, stop FROM anime_chunks WHERE anime_id IN ({','.join('?' * len(ids))}) "
                           "ORDER BY start", ids)

    def search(self, vector: np.ndarray, anime_ids: list[str] | None = None,
               k: int = 5) -> tuple[list[Chunk], list[float]]:
        """The k chunks most similar to a normalized query vector, best first, and their cosine scores."""
        query = np.asarray(vector, dtype=np.float32)
        rows, scores = [], []
        for start, stop in self.spans(anime_ids):
            for begin in range(start, stop, BLOCK):
                end = min(stop, begin + BLOCK)
                scores.append(np.asarray(self.vectors[begin:end], dtype=np.float32) @ query)
                rows.append(np.arange(begin, end))
        if not scores or k < 1:
            return [], []
        rows, scores = np.concatenate(rows), np.concatenate(scores)
        best = np.argpartition(-scores, min(k, len(scores)) - 1)[:k]
        best = best[np.argsort(-scores[best], kind="stable")]
        ids = [int(rows[i]) for i in best]
        found = {row[0]: Chunk(*row[1:]) for row in self._query(
            "SELECT id, text, source, anime_id, entity_type, character_id FROM chunks "
            f"WHERE id IN ({','.join('?' * len(ids))})", ids)}
        return [found[i] for i in ids], [float(scores[i]) for i in best]
