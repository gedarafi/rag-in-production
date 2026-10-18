"""Release the prepared catalog and index as one versioned bundle, and fetch it in the worker.

A bundle is one zstd-compressed tar of everything the pipeline reads at runtime: recommender.toml,
prompts/, data/documents/anilist/ (catalog, SQLite database, documents) and data/index/anilist/
(vectors, chunks, tool and feature embeddings). Next to it, manifest.json records the version, the
bundle's SHA-256 and each file's size and SHA-256.

In S3 a version is the prefix recommender/<version>/. The SSM parameter named by
ARTIFACT_VERSION_PARAMETER says which version the worker serves: publishing a version writes the
bundle first and then points the parameter at it; pointing it back is the rollback.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

from ..tracing import stage

FORMAT = 1
BUNDLE = "bundle.tar.zst"
MANIFEST = "manifest.json"
# Built and used at runtime only: the full chunk embedding cache isn't needed to answer.
EXCLUDED = {"embeddings.sqlite3", "embeddings.sqlite3-journal", "routing.sqlite3", "character-comparison.sqlite3"}


def runtime_files(root: Path) -> list[Path]:
    """The files of a prepared workspace that answering needs, relative to root."""
    files = [Path("recommender.toml")]
    files += sorted(p.relative_to(root) for p in (root / "prompts").rglob("*") if p.is_file())
    for folder in ("data/documents/anilist", "data/index/anilist"):
        files += sorted(p.relative_to(root) for p in (root / folder).rglob("*")
                        if p.is_file() and p.name not in EXCLUDED and not p.name.endswith((".tmp", ".part")))
    return files


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
    (target / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def verify(folder: Path, manifest: dict) -> None:
    """Every file of the manifest is in folder with its size and SHA-256."""
    for name, expected in manifest["files"].items():
        path = folder / name
        if not path.is_file() or path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Artifact file {name} is missing or differs from manifest {manifest['version']}.")


def publish(bundle_dir: Path, bucket: str, parameter: str, *, s3=None, ssm=None, activate: bool = True) -> str:
    """Upload a bundle to s3://bucket/recommender/<version>/, then point the SSM parameter at it."""
    import boto3

    s3 = s3 or boto3.client("s3")
    manifest = json.loads((bundle_dir / MANIFEST).read_text(encoding="utf-8"))
    version = manifest["version"]
    prefix = f"recommender/{version}/"
    s3.upload_file(str(bundle_dir / BUNDLE), bucket, prefix + BUNDLE,
                   ExtraArgs={"Metadata": {"sha256": manifest["bundle"]["sha256"]}})
    # The manifest goes last: a version is complete once its manifest exists.
    s3.upload_file(str(bundle_dir / MANIFEST), bucket, prefix + MANIFEST,
                   ExtraArgs={"ContentType": "application/json"})
    if activate:
        activate_version(version, bucket, parameter, s3=s3, ssm=ssm)
    return version


def activate_version(version: str, bucket: str, parameter: str, *, s3=None, ssm=None) -> None:
    """Serve version from now on (also the rollback); refuses a version that isn't fully uploaded."""
    import boto3

    s3 = s3 or boto3.client("s3")
    ssm = ssm or boto3.client("ssm")
    s3.head_object(Bucket=bucket, Key=f"recommender/{version}/{MANIFEST}")
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
