"""Release the prepared catalog and index as one versioned bundle, and fetch it in the worker.

A bundle is one zstd-compressed tar of everything the pipeline reads at runtime: recommender.toml,
prompts/, data/documents/anilist/ (the SQLite catalog and the documents) and data/index/anilist/
(vectors, chunks, tool and feature embeddings). The worker opens it as built (RagPipeline readonly),
so build inputs such as catalog.jsonl stay out. Next to it, manifest.json records the version, the
bundle's SHA-256 and each file's size and SHA-256.

The passage index (chunk_index.py) is not in the tar: bundle() puts it next to it, in passages/, and
publishing loads it into the vector database (vector_db.py) as that version's table.

In S3 a version is the prefix recommender/<version>/. The SSM parameter named by
ARTIFACT_VERSION_PARAMETER says which version the worker serves: publishing a version writes the
bundle and its passages first and then points the parameter at it; pointing it back is the rollback.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sqlite3
import tarfile
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from ..tracing import stage
from .chunk_index import DATABASE as CHUNKS_DATABASE, LEGACY as LEGACY_INDEX, VECTORS as CHUNK_VECTORS

FORMAT = 1
BUNDLE = "bundle.tar.zst"
MANIFEST = "manifest.json"
# Not needed to answer: the build's inputs and records (the catalog.sqlite3 built from catalog.jsonl
# is bundled instead), the full chunk embedding cache, and caches that the worker fills itself.
EXCLUDED = {"catalog.jsonl", "setup.json", "embedding-state.json", "embeddings.sqlite3",
            "routing.sqlite3", "character-comparison.sqlite3", *LEGACY_INDEX}
# SQLite's side files; bundle() checkpoints the databases first, so their content is in the main file.
SQLITE_SIDE_FILES = ("-wal", "-shm", "-journal")
# The passage index goes to the vector database instead: next to the bundle, in this folder.
PASSAGES = "passages"
PASSAGE_FILES = (CHUNKS_DATABASE, CHUNK_VECTORS)
# Passage tables kept in the vector database besides the active one: the reach of a rollback.
KEEP_VERSIONS = 3


def runtime_files(root: Path) -> list[Path]:
    """The files of a prepared workspace that answering needs, relative to root."""
    files = [Path("recommender.toml")]
    files += sorted(p.relative_to(root) for p in (root / "prompts").rglob("*") if p.is_file())
    for folder in ("data/documents/anilist", "data/index/anilist"):
        files += sorted(p.relative_to(root) for p in (root / folder).rglob("*")
                        if p.is_file() and p.name not in EXCLUDED and p.name not in PASSAGE_FILES
                        and not p.name.endswith((".tmp", ".part", *SQLITE_SIDE_FILES)))
    return files


def checkpoint(path: Path) -> None:
    """Move a WAL database's committed pages into the main file, so the file alone is complete."""
    with closing(sqlite3.connect(path)) as db:
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1 << 22):
            digest.update(block)
    return digest.hexdigest()


def new_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def bundle(root: Path, out: Path, version: str | None = None, *, settings_text: str | None = None) -> Path:
    """Write <out>/<version>/{bundle.tar.zst, manifest.json} from the workspace at root."""
    version = version or new_version()
    target = out / version
    target.mkdir(parents=True, exist_ok=False)
    files = runtime_files(root)
    for relative in files:
        if relative.suffix == ".sqlite3":
            checkpoint(root / relative)
    entries = {}
    with tarfile.open(target / BUNDLE, "w:zst") as archive:
        for relative in files:
            path = root / relative
            if relative == Path("recommender.toml") and settings_text is not None:
                # The bundle's settings select the AniList library regardless of the builder's default.
                info = tarfile.TarInfo(str(relative))
                data = settings_text.encode()
                info.size, info.mtime = len(data), int(time.time())
                archive.addfile(info, io.BytesIO(data))
                entries[str(relative)] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                continue
            archive.add(path, arcname=str(relative), recursive=False)
            entries[str(relative)] = {"size": path.stat().st_size, "sha256": sha256(path)}
    manifest = {"format": FORMAT, "version": version, "created": datetime.now(timezone.utc).isoformat(),
                "bundle": {"name": BUNDLE, "size": (target / BUNDLE).stat().st_size, "sha256": sha256(target / BUNDLE)},
                "files": entries}
    index = root / "data/index/anilist"
    if all((index / name).is_file() for name in PASSAGE_FILES):
        (target / PASSAGES).mkdir()
        for name in PASSAGE_FILES:
            try:
                os.link(index / name, target / PASSAGES / name)  # a snapshot: rebuilds replace the files
            except OSError:
                shutil.copy2(index / name, target / PASSAGES / name)
        manifest["passages"] = "vector-db"
    (target / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def verify(folder: Path, manifest: dict) -> None:
    """Every file of the manifest is in folder with its size and SHA-256."""
    for name, expected in manifest["files"].items():
        path = folder / name
        if not path.is_file() or path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Artifact file {name} is missing or differs from manifest {manifest['version']}.")


def publish(bundle_dir: Path, bucket: str, parameter: str, *, s3=None, ssm=None, activate: bool = True,
            vector_db=None, progress=None) -> str:
    """Upload a bundle to s3://bucket/recommender/<version>/ and load its passages into the vector
    database, then point the SSM parameter at it."""
    import boto3

    s3 = s3 or boto3.client("s3")
    ssm = ssm or boto3.client("ssm")
    manifest = json.loads((bundle_dir / MANIFEST).read_text(encoding="utf-8"))
    version = manifest["version"]
    if manifest.get("passages") == "vector-db" and vector_db is None:
        raise ValueError(f"Bundle {version} keeps its passages for the vector database: pass --vector-db-cluster "
                         "and --vector-db-secret (or set VECTOR_DB_CLUSTER_ARN and VECTOR_DB_SECRET_ARN).")
    prefix = f"recommender/{version}/"
    s3.upload_file(str(bundle_dir / BUNDLE), bucket, prefix + BUNDLE,
                   ExtraArgs={"Metadata": {"sha256": manifest["bundle"]["sha256"]}})
    if manifest.get("passages") == "vector-db":
        vector_db.table(version).load(bundle_dir / PASSAGES, progress)
    # The manifest goes last: a version is complete once its manifest exists.
    s3.upload_file(str(bundle_dir / MANIFEST), bucket, prefix + MANIFEST,
                   ExtraArgs={"ContentType": "application/json"})
    if activate:
        activate_version(version, bucket, parameter, s3=s3, ssm=ssm, vector_db=vector_db)
    if vector_db is not None:
        try:
            active = ssm.get_parameter(Name=parameter)["Parameter"]["Value"]
        except Exception:  # noqa: BLE001 - no active version yet: keep the newest tables only
            active = None
        vector_db.prune(KEEP_VERSIONS, active=active)
    return version


def activate_version(version: str, bucket: str, parameter: str, *, s3=None, ssm=None, vector_db=None) -> None:
    """Serve version from now on (also the rollback); refuses a version that isn't fully uploaded,
    including its passages in the vector database."""
    import boto3

    s3 = s3 or boto3.client("s3")
    ssm = ssm or boto3.client("ssm")
    key = f"recommender/{version}/{MANIFEST}"
    s3.head_object(Bucket=bucket, Key=key)
    if json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read()).get("passages") == "vector-db":
        if vector_db is None:
            raise ValueError(f"Version {version} keeps its passages in the vector database: pass --vector-db-cluster "
                             "and --vector-db-secret (or set VECTOR_DB_CLUSTER_ARN and VECTOR_DB_SECRET_ARN).")
        if not vector_db.table(version).ready():
            raise ValueError(f"The passages of {version} aren't in the vector database (pruned or never loaded); "
                             "publish the bundle again.")
    ssm.put_parameter(Name=parameter, Value=version, Type="String", Overwrite=True)


class _Hashing:
    """A file-like stream that hashes what is read through it."""

    def __init__(self, stream):
        self.stream, self.digest, self.size = stream, hashlib.sha256(), 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        self.size += len(data)
        return data


class ArtifactStore:
    """The worker's copy of the active bundle under a local folder (Lambda: /tmp), one version at a time."""

    def __init__(self, bucket: str, parameter: str, folder: Path, *, s3=None, ssm=None, check_seconds: float = 300):
        self.bucket, self.parameter, self.folder = bucket, parameter, Path(folder)
        self._s3, self._ssm = s3, ssm
        self.check_seconds = check_seconds
        self.version: str | None = None
        self._checked = 0.0

    @classmethod
    def from_env(cls) -> "ArtifactStore":
        return cls(os.environ["ARTIFACT_BUCKET"], os.environ["ARTIFACT_VERSION_PARAMETER"],
                   Path(os.environ.get("ARTIFACT_DIR", "/tmp/recommender")),
                   check_seconds=float(os.environ.get("ARTIFACT_CHECK_SECONDS", "300")))

    @property
    def s3(self):
        if self._s3 is None:
            import boto3
            self._s3 = boto3.client("s3")
        return self._s3

    @property
    def ssm(self):
        if self._ssm is None:
            import boto3
            self._ssm = boto3.client("ssm")
        return self._ssm

    def active_version(self) -> str:
        """The version to serve, re-read from SSM at most every check_seconds."""
        now = time.monotonic()
        if self.version is None or now - self._checked >= self.check_seconds:
            self._checked = now
            return self.ssm.get_parameter(Name=self.parameter)["Parameter"]["Value"]
        return self.version

    def current(self) -> tuple[str, Path, bool]:
        """(version, its folder, whether it was just downloaded) for the active version."""
        version = self.active_version()
        target = self.folder / version
        if version == self.version and (target / MANIFEST).is_file():
            return version, target, False
        if not (target / MANIFEST).is_file():
            self.download(version, target)
        old, self.version = self.version, version
        if old and old != version:
            # Only one version fits comfortably in /tmp.
            shutil.rmtree(self.folder / old, ignore_errors=True)
        return version, target, True

    def download(self, version: str, target: Path) -> None:
        prefix = f"recommender/{version}/"
        with stage("download_artifacts", **{"recommender.artifact_version": version}) as span:
            manifest = json.loads(self.s3.get_object(Bucket=self.bucket, Key=prefix + MANIFEST)["Body"].read())
            partial = target.with_name(target.name + ".partial")
            shutil.rmtree(partial, ignore_errors=True)
            # Remove other versions first: a bundle and its extracted copy need the space.
            if self.folder.is_dir():
                for other in self.folder.iterdir():
                    if other.name not in (version, self.version):
                        shutil.rmtree(other, ignore_errors=True)
            partial.mkdir(parents=True)
            started = time.monotonic()
            body = _Hashing(self.s3.get_object(Bucket=self.bucket, Key=prefix + BUNDLE)["Body"])
            # Stream straight from S3 into the folder: no second copy of the compressed bundle.
            with tarfile.open(fileobj=body, mode="r|zst") as archive:
                archive.extractall(partial, filter="data")
            # The tar reader stops at the end-of-archive marker; hash the rest of the object too.
            while body.read(1 << 20):
                pass
            if body.digest.hexdigest() != manifest["bundle"]["sha256"]:
                shutil.rmtree(partial, ignore_errors=True)
                raise ValueError(f"Bundle {version} doesn't match its manifest checksum.")
            (partial / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
            partial.replace(target)
            span.set_attributes({"recommender.bundle_bytes": body.size,
                                 "recommender.download_seconds": round(time.monotonic() - started, 3)})
