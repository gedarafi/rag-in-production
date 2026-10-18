import io
import json
import shutil
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.recommender import artifacts
from src.recommender.anime_tools import AnimeTools, structured
from src.recommender.chunk_index import write_index
from src.recommender.documents import Chunk
from src.recommender.service import Recommender
from src.recommender.vector_db import VectorDb
from tests.recommender.test_vector_db import FakeDataApi

REPOSITORY = Path(__file__).resolve().parents[2]


class FakeS3:
    def __init__(self):
        self.objects = {}

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        self.objects[(bucket, key)] = Path(path).read_bytes()

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)
        return {}


class FakeSsm:
    def __init__(self):
        self.values = {}

    def put_parameter(self, Name, Value, Type, Overwrite):
        self.values[Name] = Value

    def get_parameter(self, Name):
        return {"Parameter": {"Value": self.values[Name]}}


def workspace(root: Path, marker="v1") -> Path:
    (root / "prompts").mkdir(parents=True)
    (root / "prompts/answer.txt").write_text("{context} {question}", encoding="utf-8")
    (root / "recommender.toml").write_text("[knowledge_base]\ndefault = \"custom\"\n", encoding="utf-8")
    documents, index = root / "data/documents/anilist", root / "data/index/anilist"
    documents.mkdir(parents=True)
    index.mkdir(parents=True)
    with closing(sqlite3.connect(documents / "catalog.sqlite3")) as db:
        db.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
        db.execute("INSERT INTO meta VALUES ('marker', ?)", (marker,))
        db.commit()
    (documents / "anime-1.md").write_text("title: First", encoding="utf-8")
    # Build inputs and caches the worker doesn't need.
    (documents / "catalog.jsonl").write_text("{}", encoding="utf-8")
    (documents / "catalog.sqlite3-wal").write_bytes(b"")
    (index / "setup.json").write_text("{}", encoding="utf-8")
    (index / "embeddings.sqlite3").write_bytes(b"cache")
    (index / "vectors.npy").write_bytes(b"\x00" * 64)
    return root


def catalog_marker(folder: Path) -> str:
    with closing(sqlite3.connect(folder / "data/documents/anilist/catalog.sqlite3")) as db:
        return db.execute("SELECT value FROM meta WHERE key='marker'").fetchone()[0]


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.s3, self.ssm = FakeS3(), FakeSsm()

    def build(self, version, marker):
        source = workspace(self.root / f"src-{version}", marker)
        return artifacts.bundle(source, self.root / "out", version, settings_text='[knowledge_base]\ndefault = "anilist"\n')

    def test_bundle_has_the_runtime_files_and_their_checksums(self):
        target = self.build("v1", "catalog")
        manifest = json.loads((target / "manifest.json").read_text())
        self.assertEqual(manifest["version"], "v1")
        self.assertIn("data/documents/anilist/catalog.sqlite3", manifest["files"])
        self.assertIn("data/documents/anilist/anime-1.md", manifest["files"])
        for excluded in ("data/documents/anilist/catalog.jsonl", "data/documents/anilist/catalog.sqlite3-wal",
                         "data/index/anilist/setup.json", "data/index/anilist/embeddings.sqlite3",
                         "data/index/anilist/vectors.npy"):
            self.assertNotIn(excluded, manifest["files"])
        self.assertEqual(manifest["bundle"]["sha256"], artifacts.sha256(target / "bundle.tar.zst"))

    def test_publish_then_fetch_and_switch_versions(self):
        for version, marker in (("v1", "first"), ("v2", "second")):
            artifacts.publish(self.build(version, marker), "bucket", "/p", s3=self.s3, ssm=self.ssm, activate=False)
        store = artifacts.ArtifactStore("bucket", "/p", self.root / "tmp", s3=self.s3, ssm=self.ssm, check_seconds=0)
        artifacts.activate_version("v1", "bucket", "/p", s3=self.s3, ssm=self.ssm)
        version, folder, fresh = store.current()
        self.assertEqual((version, fresh), ("v1", True))
        self.assertEqual(catalog_marker(folder), "first")
        self.assertIn('default = "anilist"', (folder / "recommender.toml").read_text())
        artifacts.verify(folder, json.loads((folder / "manifest.json").read_text()))
        self.assertEqual(store.current()[2], False)
        artifacts.activate_version("v2", "bucket", "/p", s3=self.s3, ssm=self.ssm)
        version, folder, fresh = store.current()
        self.assertEqual((version, fresh), ("v2", True))
        self.assertEqual(catalog_marker(folder), "second")
        self.assertFalse((self.root / "tmp/v1").exists())

    def test_unpublished_version_cannot_be_activated(self):
        with self.assertRaises(KeyError):
            artifacts.activate_version("v9", "bucket", "/p", s3=self.s3, ssm=self.ssm)

    def test_corrupted_bundle_is_rejected(self):
        artifacts.publish(self.build("v1", "first"), "bucket", "/p", s3=self.s3, ssm=self.ssm)
        key = ("bucket", "recommender/v1/manifest.json")
        manifest = json.loads(self.s3.objects[key])
        manifest["bundle"]["sha256"] = "0" * 64
        self.s3.objects[key] = json.dumps(manifest).encode()
        store = artifacts.ArtifactStore("bucket", "/p", self.root / "tmp", s3=self.s3, ssm=self.ssm)
        with self.assertRaisesRegex(ValueError, "checksum"):
            store.current()
        self.assertFalse((self.root / "tmp/v1").exists())

    def test_verify_detects_changed_files(self):
        target = self.build("v1", "first")
        manifest = json.loads((target / "manifest.json").read_text())
        source = self.root / "src-v1"
        manifest["files"]["recommender.toml"] = {"size": (source / "recommender.toml").stat().st_size,
                                                 "sha256": artifacts.sha256(source / "recommender.toml")}
        artifacts.verify(source, manifest)
        (source / "data/documents/anilist/anime-1.md").write_text("changed")
        with self.assertRaisesRegex(ValueError, "anime-1.md"):
            artifacts.verify(source, manifest)

    def test_worker_serves_the_published_catalog_without_rebuilding_it(self):
        # Extraction changes the files' modification times, which the build uses to detect a stale
        # catalog; the worker must open the bundle as built instead of rebuilding it on every cold start.
        source = self.root / "build"
        documents = source / "data/documents/anilist"
        documents.mkdir(parents=True)
        shutil.copytree(REPOSITORY / "prompts", source / "prompts")
        shutil.copy(REPOSITORY / "recommender.toml", source / "recommender.toml")
        row = {"id": "1", "title_english": "Re:Zero", "popularity": 500, "genres": ["Fantasy"],
               "description": "Subaru returns after death.", "characters": []}
        (documents / "catalog.jsonl").write_text(json.dumps(structured(row, "anime-1.md")), encoding="utf-8")
        AnimeTools.load(documents)
        artifacts.publish(artifacts.bundle(source, self.root / "out", "v1"), "bucket", "/p", s3=self.s3, ssm=self.ssm)
        _, folder, _ = artifacts.ArtifactStore("bucket", "/p", self.root / "tmp", s3=self.s3, ssm=self.ssm).current()
        self.assertFalse((folder / "data/documents/anilist/catalog.jsonl").exists())
        rebuilt = AssertionError("the worker rebuilt part of the published catalog")
        with (patch("src.recommender.sqlite_catalog.ensure_database", side_effect=rebuilt),
              patch("src.recommender.catalog_metadata.ensure_production_metadata", side_effect=rebuilt),
              patch("src.recommender.entity_store.EntityStore.ensure_dictionary", side_effect=rebuilt)):
            models = SimpleNamespace(embedder=SimpleNamespace(load=lambda: None))
            recommender = Recommender(folder, models=models).warm()
            catalog = recommender.pipeline.anime_tools()
            self.assertEqual(catalog.query("SELECT title FROM anime")[0][0], "Re:Zero")
            self.assertEqual(catalog.query("SELECT COUNT(*) FROM entity_dictionary WHERE kind='anime'")[0][0], 1)

    def test_passages_go_to_the_vector_database_before_the_version_is_complete(self):
        source = workspace(self.root / "src-v1", "first")
        write_index(source / "data/index/anilist", [Chunk("A passage.", "anime-1.md", "1", "anime")], np.ones((1, 4)))
        target = artifacts.bundle(source, self.root / "out", "v1")
        manifest = json.loads((target / "manifest.json").read_text())
        self.assertEqual(manifest["passages"], "vector-db")
        self.assertFalse([name for name in manifest["files"] if "chunk" in name])
        self.assertTrue((target / "passages/chunks.sqlite3").is_file() and (target / "passages/chunk-vectors.npy").is_file())
        with self.assertRaisesRegex(ValueError, "vector-db-cluster"):
            artifacts.publish(target, "bucket", "/p", s3=self.s3, ssm=self.ssm)
        api = FakeDataApi()
        vectors = VectorDb("arn:cluster", "arn:secret", client=api)
        uploaded = []
        upload = self.s3.upload_file
        def record(path, bucket, key, ExtraArgs=None):
            uploaded.append((key, len(api.tables.get('"chunks_v1"', []))))
            upload(path, bucket, key, ExtraArgs)
        self.s3.upload_file = record
        artifacts.publish(target, "bucket", "/p", s3=self.s3, ssm=self.ssm, vector_db=vectors)
        # The manifest, which completes a version, is uploaded once the passages are loaded.
        self.assertEqual(uploaded, [("recommender/v1/bundle.tar.zst", 0), ("recommender/v1/manifest.json", 1)])
        self.assertEqual(self.ssm.values["/p"], "v1")
        api.tables.clear()
        api.indexes.clear()
        with self.assertRaisesRegex(ValueError, "aren't in the vector database"):
            artifacts.activate_version("v1", "bucket", "/p", s3=self.s3, ssm=self.ssm, vector_db=vectors)
        with self.assertRaisesRegex(ValueError, "vector-db-cluster"):
            artifacts.activate_version("v1", "bucket", "/p", s3=self.s3, ssm=self.ssm)

