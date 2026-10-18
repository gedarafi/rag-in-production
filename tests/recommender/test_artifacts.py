import io
import json
import tempfile
import unittest
from pathlib import Path

from src.recommender import artifacts


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
    (documents / "catalog.jsonl").write_text(marker, encoding="utf-8")
    (index / "vectors.npy").write_bytes(b"\x00" * 64)
    (index / "embeddings.sqlite3").write_bytes(b"cache")  # not needed at runtime
    return root


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
        self.assertIn("data/documents/anilist/catalog.jsonl", manifest["files"])
        self.assertNotIn("data/index/anilist/embeddings.sqlite3", manifest["files"])
        self.assertEqual(manifest["bundle"]["sha256"], artifacts.sha256(target / "bundle.tar.zst"))

    def test_publish_then_fetch_and_switch_versions(self):
        for version, marker in (("v1", "first"), ("v2", "second")):
            artifacts.publish(self.build(version, marker), "bucket", "/p", s3=self.s3, ssm=self.ssm, activate=False)
        store = artifacts.ArtifactStore("bucket", "/p", self.root / "tmp", s3=self.s3, ssm=self.ssm, check_seconds=0)
        artifacts.activate_version("v1", "bucket", "/p", s3=self.s3, ssm=self.ssm)
        version, folder, fresh = store.current()
        self.assertEqual((version, fresh), ("v1", True))
        self.assertEqual((folder / "data/documents/anilist/catalog.jsonl").read_text(), "first")
        self.assertIn('default = "anilist"', (folder / "recommender.toml").read_text())
        artifacts.verify(folder, json.loads((folder / "manifest.json").read_text()))
        self.assertEqual(store.current()[2], False)
        artifacts.activate_version("v2", "bucket", "/p", s3=self.s3, ssm=self.ssm)
        version, folder, fresh = store.current()
        self.assertEqual((version, fresh), ("v2", True))
        self.assertEqual((folder / "data/documents/anilist/catalog.jsonl").read_text(), "second")
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
        (source / "data/documents/anilist/catalog.jsonl").write_text("changed")
        with self.assertRaisesRegex(ValueError, "catalog.jsonl"):
            artifacts.verify(source, manifest)
