"""Publish the built AniList library to S3 once; pull it onto any server that answers questions.

A bundle is one zstd-compressed tar of what the pipeline reads at runtime:

  settings.toml                  the settings it was built with (RAG_* variables still override them)
  prompts/
  data/documents/anilist/        catalog.sqlite3 and the anime-*.md documents
  data/index/anilist/            chunks.json, vectors.npy and the tool/feature embedding caches

Build inputs and build caches stay out: catalog.jsonl (the pipeline serves catalog.sqlite3 as built
when its source is absent), the full chunk embedding cache and the setup markers.

In S3 a version lives under <prefix>/<version>/ (bundle.tar.zst and manifest.json), and the object
<prefix>/current names the version servers pull. Publishing uploads the version first and then moves
`current`; pointing `current` back at an older version is the rollback.

On the server, pull() extracts a version into <target>/releases/<version>/ and points the symlink
<target>/current at it, so a half-downloaded version is never served.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tarfile
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings

BUNDLE = "bundle.tar.zst"
MANIFEST = "manifest.json"
CURRENT = "current"
# Build inputs, build caches and setup records: a server never reads them.
EXCLUDED = {"catalog.jsonl", "embeddings.sqlite3", "setup.json", "embedding-state.json",
            "chunks.sqlite3", "chunk-vectors.npy"}
# SQLite's side files; publish() checkpoints the databases first, so their content is in the main file.
SQLITE_SIDE_FILES = ("-wal", "-shm", "-journal")


def runtime_files(settings: Settings, config: Path) -> dict[str, Path]:
    """Archive name → local path for everything a server needs."""
    files = {"settings.toml": config.resolve()}
    for path in sorted((settings.root / "prompts").glob("*")):
        if path.is_file():
            files[f"prompts/{path.name}"] = path
    for folder in (settings.documents, settings.index):
        for path in sorted(folder.iterdir()):
            if (path.is_file() and path.name not in EXCLUDED and not path.name.endswith(SQLITE_SIDE_FILES)
                    and not path.name.endswith((".tmp", ".part"))):
                files[str(path.relative_to(settings.root))] = path
    if "data/documents/anilist/catalog.sqlite3" not in files or "data/index/anilist/vectors.npy" not in files:
        raise RuntimeError("The AniList library is not built. Run `just recommender-setup` first.")
    return files


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def checkpoint(paths) -> None:
    """Fold each SQLite write-ahead log into its database, so the main file alone is complete."""
    for path in paths:
        if path.suffix == ".sqlite3":
            with closing(sqlite3.connect(path, timeout=120)) as db:
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def client():
    # boto3 reads AWS_ENDPOINT_URL, so the same code talks to LocalStack.
    import boto3
    return boto3.client("s3")


def publish(settings: Settings, config: Path, bucket: str, prefix: str = "recommender", s3=None) -> dict:
    """Upload a new version and make it current; returns its manifest."""
    s3 = s3 or client()
    files = runtime_files(settings, config)
    checkpoint(files.values())
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with tempfile.TemporaryDirectory() as folder:
        archive = Path(folder) / BUNDLE
        with tarfile.open(archive, "w:zst") as tar:
            for name, path in files.items():
                tar.add(path, arcname=name, recursive=False)
        manifest = {"version": version, "bundle": BUNDLE, "bundle_sha256": sha256(archive),
                    "bundle_size": archive.stat().st_size,
                    "files": {name: {"size": path.stat().st_size} for name, path in files.items()}}
        s3.upload_file(str(archive), bucket, f"{prefix}/{version}/{BUNDLE}")
    s3.put_object(Bucket=bucket, Key=f"{prefix}/{version}/{MANIFEST}",
                  Body=json.dumps(manifest, indent=2).encode(), ContentType="application/json")
    # Last: servers only see the version once all of it is in S3.
    s3.put_object(Bucket=bucket, Key=f"{prefix}/{CURRENT}", Body=version.encode(), ContentType="text/plain")
    return manifest


def current_version(bucket: str, prefix: str = "recommender", s3=None) -> str:
    s3 = s3 or client()
    return s3.get_object(Bucket=bucket, Key=f"{prefix}/{CURRENT}")["Body"].read().decode().strip()


def pull(bucket: str, target: Path, prefix: str = "recommender", version: str | None = None, s3=None) -> Path:
    """Make <target>/current the published version (or `version`); returns its folder.

    Does nothing when that version is already current, so a server can pull on every start.
    """
    s3 = s3 or client()
    target = Path(target)
    version = version or current_version(bucket, prefix, s3)
    if not version or "/" in version or version.startswith("."):
        raise RuntimeError(f"Invalid bundle version in s3://{bucket}/{prefix}/{CURRENT}: {version!r}")
    link, release = target / CURRENT, target / "releases" / version
    if link.is_symlink() and link.resolve() == release.resolve() and (release / MANIFEST).is_file():
        return release
    manifest = json.loads(s3.get_object(Bucket=bucket, Key=f"{prefix}/{version}/{MANIFEST}")["Body"].read())
    (target / "releases").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target) as folder:
        archive = Path(folder) / BUNDLE
        s3.download_file(bucket, f"{prefix}/{version}/{BUNDLE}", str(archive))
        if sha256(archive) != manifest["bundle_sha256"]:
            raise RuntimeError(f"Bundle {version} does not match its manifest checksum; not serving it.")
        staging = Path(folder) / "release"
        with tarfile.open(archive, "r:zst") as tar:
            tar.extractall(staging, filter="data")
        for name, entry in manifest["files"].items():
            path = staging / name
            if not path.is_file() or path.stat().st_size != entry["size"]:
                raise RuntimeError(f"Bundle {version} is missing or truncates {name}.")
        (staging / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        shutil.rmtree(release, ignore_errors=True)
        staging.replace(release)
    # Swap the symlink atomically, then drop releases older than the previous one.
    previous = link.resolve() if link.is_symlink() else None
    temporary = target / (CURRENT + ".tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(release.relative_to(target))
    os.replace(temporary, link)
    for old in (target / "releases").iterdir():
        if old.resolve() not in (release.resolve(), previous):
            shutil.rmtree(old, ignore_errors=True)
    return release
