"""The passage index in Aurora PostgreSQL with pgvector, loaded and searched through the RDS Data API.

Publishing a bundle loads its chunk index (chunk_index.py: chunks.sqlite3, chunk-vectors.npy) into the
table chunks_<version>, and the worker searches the table of the version it serves: activating a
version, or rolling back, switches tables with it. The Data API is HTTPS signed with IAM and the
cluster's master secret, so neither a VPC nor a database driver is needed. Vectors are halfvec (float16,
as in chunk-vectors.npy) and normalized, so the inner product is the cosine similarity. A paused cluster
answers DatabaseResumingException until it has resumed (about 15 s).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import numpy as np

from .chunk_index import BLOCK, DATABASE, VECTORS
from .documents import Chunk
from .progress import Progress
from ..tracing import event

VERSION = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
# A BatchExecuteStatement request may be at most 4 MiB.
BATCH_BYTES = 3_500_000
COLUMNS = "id, text, source, anime_id, entity_type, character_id"


def _value(value) -> dict:
    if value is None:
        return {"isNull": True}
    if isinstance(value, int):
        return {"longValue": value}
    return {"stringValue": value}


def _parameters(**values) -> list[dict]:
    return [{"name": name, "value": _value(value)} for name, value in values.items()]


def _code(error: Exception) -> str:
    return getattr(error, "response", {}).get("Error", {}).get("Code", "")


class VectorDb:
    """One Aurora database of passage tables, one table per bundle version."""

    def __init__(self, cluster_arn: str, secret_arn: str, database: str = "vectors", *, client=None,
                 resume_seconds: float = 90, wait_seconds: float = 5):
        self.cluster_arn, self.secret_arn, self.database = cluster_arn, secret_arn, database
        self._client = client
        self.resume_seconds, self.wait_seconds = resume_seconds, wait_seconds

    @classmethod
    def from_env(cls) -> VectorDb | None:
        """The worker's vector database (VECTOR_DB_* from modules/recommender), or None without one."""
        if not os.environ.get("VECTOR_DB_CLUSTER_ARN"):
            return None
        return cls(os.environ["VECTOR_DB_CLUSTER_ARN"], os.environ["VECTOR_DB_SECRET_ARN"],
                   os.environ.get("VECTOR_DB_NAME") or "vectors")

    @property
    def client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config
            # On AWS a call ends within 45 s (continueAfterTimeout keeps long statements running); LocalStack
            # answers when the statement is done. A shorter client timeout would make boto3 send it again.
            self._client = boto3.client("rds-data", config=Config(read_timeout=900))
        return self._client

    def call(self, operation: str, **request) -> dict:
        """One Data API call, waiting while a paused cluster resumes."""
        deadline = time.monotonic() + self.resume_seconds
        while True:
            try:
                return getattr(self.client, operation)(resourceArn=self.cluster_arn, secretArn=self.secret_arn,
                                                       database=self.database, **request)
            except Exception as error:
                if _code(error) != "DatabaseResumingException" or time.monotonic() >= deadline:
                    raise
                time.sleep(self.wait_seconds)

    def execute(self, sql: str, **values) -> list[list[dict]]:
        request = {"sql": sql, **({"parameters": _parameters(**values)} if values else {})}
        return self.call("execute_statement", **request).get("records", [])

    def table(self, version: str) -> PassageTable:
        return PassageTable(self, version)

    def versions(self) -> list[str]:
        """Versions with a passage table, newest first."""
        rows = self.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename LIKE 'chunks\\_%'")
        return sorted((row[0]["stringValue"].removeprefix("chunks_") for row in rows), reverse=True)

    def prune(self, keep: int, active: str | None = None) -> list[str]:
        """Drop the tables of all but the newest `keep` versions and the active one; returns the dropped versions."""
        dropped = [version for version in self.versions()[keep:] if version != (active or "").lower()]
        for version in dropped:
            self.execute(f"DROP TABLE IF EXISTS {self.table(version).name}")
        return dropped


class PassageTable:
    """The passages of one bundle version: the same search as chunk_index.ChunkIndex."""

    def __init__(self, db: VectorDb, version: str):
        if not VERSION.match(version):
            raise ValueError(f"Not a bundle version: {version!r}")
        self.db, self.version = db, version
        # A table name can't be a bind parameter; the version is checked above.
        self.name = f'"chunks_{version.lower()}"'
        self.index = f'"chunks_{version.lower()}_hnsw"'

    def count(self) -> int | None:
        """Rows in the table, or None if there is no table."""
        exists = self.db.execute("SELECT to_regclass(:name) IS NOT NULL", name=self.name)[0][0]["booleanValue"]
        return self.db.execute(f"SELECT COUNT(*) FROM {self.name}")[0][0]["longValue"] if exists else None

    def ready(self) -> bool:
        """Loaded completely, with the index that unfiltered searches need."""
        return self.db.execute("SELECT to_regclass(:name) IS NOT NULL", name=self.index)[0][0]["booleanValue"]

    def search(self, vector: np.ndarray, anime_ids: list[str] | None = None,
               k: int = 5) -> tuple[list[Chunk], list[float]]:
        """The k passages closest to a normalized query vector, best first; nothing if the database fails.

        A failure only loses the semantic fallback for this question, so it is recorded, not raised.
        """
        query = np.asarray(vector, dtype=np.float64)
        distance = f"embedding <#> CAST(:query AS halfvec({len(query)}))"
        values = {"query": json.dumps(query.round(6).tolist()), "k": int(k)}
        if anime_ids is None:
            sql = f"SELECT {COLUMNS}, -({distance}) FROM {self.name} ORDER BY {distance} LIMIT :k"
        else:
            values["ids"] = ",".join(dict.fromkeys(str(i) for i in anime_ids))
            if not values["ids"]:
                return [], []
            # Exact search within a few anime: the materialized rows keep the planner from walking the
            # HNSW index and filtering afterwards, which could return fewer than k passages.
            sql = (f"WITH scoped AS MATERIALIZED (SELECT {COLUMNS}, embedding FROM {self.name} "
                   "WHERE anime_id = ANY(string_to_array(:ids, ','))) "
                   f"SELECT {COLUMNS}, -({distance}) FROM scoped ORDER BY {distance} LIMIT :k")
        try:
            records = self.db.execute(sql, **values)
        except Exception as error:
            event("vector_db_unavailable", error=f"{type(error).__name__}: {error}", version=self.version)
            return [], []
        chunks = [Chunk(*(field.get("stringValue") for field in record[1:6])) for record in records]
        return chunks, [float(record[6]["doubleValue"]) for record in records]

    def load(self, folder: Path, progress: Progress | None = None) -> int:
        """Load a chunk index (chunk_index.py's files in folder) as this version's table.

        A complete table is kept (publishing again only adds what is missing); a partial one is loaded again.
        """
        with closing(sqlite3.connect(folder / DATABASE)) as db:
            meta = dict(db.execute("SELECT key, value FROM meta"))
        count, dimension = int(meta["count"]), int(meta["dimension"])
        if self.count() != count:
            self._insert(folder, count, dimension, progress)
        self.db.execute(f"CREATE INDEX IF NOT EXISTS \"chunks_{self.version.lower()}_anime\" ON {self.name} (anime_id)")
        if not self.ready():
            self._build_hnsw(progress)
        self.db.execute(f"ANALYZE {self.name}")
        return count

    def _insert(self, folder: Path, count: int, dimension: int, progress: Progress | None) -> None:
        self.db.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.db.execute(f"DROP TABLE IF EXISTS {self.name}")
        self.db.execute(f"CREATE TABLE {self.name}(id integer PRIMARY KEY, anime_id text, source text NOT NULL, "
                        f"entity_type text, character_id text, text text NOT NULL, embedding halfvec({dimension}) NOT NULL)")
        sql = (f"INSERT INTO {self.name} VALUES (:id, :anime_id, :source, :entity_type, :character_id, :text, "
               f"CAST(:embedding AS halfvec({dimension})))")
        vectors = np.load(folder / VECTORS, mmap_mode="r")
        batch, size, done = [], 0, 0

        def send():
            nonlocal batch, size
            if batch:
                self.db.call("batch_execute_statement", sql=sql, parameterSets=batch)
                batch, size = [], 0

        with closing(sqlite3.connect(folder / DATABASE)) as db:
            rows = db.execute("SELECT id, anime_id, source, entity_type, character_id, text FROM chunks ORDER BY id")
            while block := rows.fetchmany(BLOCK):
                # float64, so that six decimals print as six decimals.
                embeddings = np.asarray(vectors[block[0][0]:block[-1][0] + 1], dtype=np.float64).round(6)
                for (identifier, anime_id, source, entity_type, character_id, text), embedding in zip(block, embeddings):
                    parameters = _parameters(id=identifier, anime_id=anime_id, source=source, entity_type=entity_type,
                                             character_id=character_id, text=text, embedding=json.dumps(embedding.tolist()))
                    row_size = len(json.dumps(parameters))
                    if size + row_size > BATCH_BYTES:
                        send()
                    batch.append(parameters)
                    size += row_size
                done += len(block)
                if progress:
                    progress("Loading passages into the vector database", done, count)
        send()

    def _build_hnsw(self, progress: Progress | None, timeout: float = 3600) -> None:
        if progress:
            progress("Indexing passage vectors", 0, 1)
        deadline = time.monotonic() + timeout
        try:
            # Building the index outlasts a Data API call (45 s) on a large catalog: let it run on.
            self.db.call("execute_statement", continueAfterTimeout=True,
                         sql=f"CREATE INDEX IF NOT EXISTS {self.index} ON {self.name} USING hnsw (embedding halfvec_ip_ops)")
        except Exception as error:
            # Still building: the call timed out, or a resent statement found the first build's name
            # taken (SQLSTATE 23505, unique_violation) while that build goes on.
            if _code(error) != "StatementTimeoutException" and "23505" not in str(error):
                raise
        while not self.ready():
            if time.monotonic() >= deadline:
                raise RuntimeError(f"The passage index of {self.version} wasn't built within {timeout:.0f} s.")
            time.sleep(self.db.wait_seconds)
        if progress:
            progress("Indexing passage vectors", 1, 1)
