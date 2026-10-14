"""Convert dataset records into the Markdown documents used by the RAG pipeline."""
from __future__ import annotations

import csv
import html
import io
import json
import re
import tempfile
import zipfile
from pathlib import Path

import requests

from .anime_tools import structured

ANILIST_FILENAME = "anilist_anime_data_complete.csv"
ANILIST_URL = f"https://www.kaggle.com/api/v1/datasets/download/calebmwelsh/anilist-anime-dataset/{ANILIST_FILENAME}"
MULTIDOC_URL = "https://raw.githubusercontent.com/doc2dial/multidoc2dial/main/file/multidoc2dial.zip"


def download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=(30, 300), stream=True) as response:
        response.raise_for_status()
        with target.open("wb") as file:
            for block in response.iter_content(1024 * 1024):
                file.write(block)
    return target


def clean(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", value)).strip()


def read_anilist_csv(source: Path) -> str:
    """Read only CSV content, including Kaggle responses wrapped in ZIP."""
    # Only read the CSV; never execute the scripts or deserialize the pickle in the archive.
    # Kaggle may wrap a single-file download in ZIP regardless of its filename.
    if source.suffix.lower() == ".zip" or zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            candidates = sorted(name for name in archive.namelist() if name.lower().endswith(".csv"))
            preferred = [name for name in candidates if Path(name).name == "anilist_anime_data_complete.csv"]
            if not preferred and len(candidates) != 1:
                raise ValueError("Archive must contain anilist_anime_data_complete.csv or exactly one CSV.")
            return archive.read((preferred or candidates)[0]).decode("utf-8-sig")
    elif source.suffix.lower() == ".csv":
        return source.read_text(encoding="utf-8-sig")
    else:
        raise ValueError("AniList source must be a CSV or ZIP file.")


def prepare_anilist(root: Path, source: Path | None = None) -> int:
    if source is None:
        candidates = [root / "data/raw" / ANILIST_FILENAME, root / "data-raw" / ANILIST_FILENAME]
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            source = download(ANILIST_URL, candidates[0])
    content = read_anilist_csv(source)
    csv.field_size_limit(16 * 1024 * 1024)
    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames or not {"title_romaji", "title_english", "title"}.intersection(reader.fieldnames):
        raise ValueError("CSV is missing AniList title columns.")
    output = root / "data/documents/anilist"
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    seen = set()
    catalog = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output,
                                         prefix="catalog-", suffix=".tmp", delete=False)
    catalog_path = Path(catalog.name)
    try:
        count = _write_anilist(reader, output, seen, catalog)
    except BaseException:
        catalog.close()
        catalog_path.unlink(missing_ok=True)
        raise
    finally:
        catalog.close()
    if not count:
        catalog_path.unlink(missing_ok=True)
        raise ValueError("No anime records found in the CSV.")
    catalog_path.replace(output / "catalog.jsonl")
    for path in output.glob("anime-*.md"):
        if path.stem.removeprefix("anime-") not in seen:
            path.unlink()
    return count


def _write_anilist(reader, output, seen, catalog):
    count = 0
    for number, row in enumerate(reader, 1):
        title = row.get("title_english") or row.get("title_romaji") or row.get("title")
        if not title:
            continue
        identifier = row.get("id") or str(number)
        identifier = re.sub(r"[^a-zA-Z0-9_-]", "_", identifier)
        if identifier in seen:
            raise ValueError(f"Duplicate anime identifier: {identifier}")
        seen.add(identifier)
        lines = [f"# {clean(title)}", "", "Dataset: calebmwelsh/anilist-anime-dataset", ""]
        for key, value in row.items():
            if key and value and not key.startswith("Unnamed:") and value.lower() not in {"nan", "none", "null"}:
                # HTML decoding an entire JSON string can turn &quot; into unescaped
                # quotes and invalidate nested character/tag descriptions.
                if value.lstrip().startswith(("[", "{")):
                    try:
                        value = json.dumps(json.loads(value), ensure_ascii=False)
                    except json.JSONDecodeError:
                        value = clean(value)
                else:
                    value = clean(value)
                lines.append(f"{key}: {value}")
        (output / f"anime-{identifier}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        catalog.write(json.dumps(structured(row, f"anime-{identifier}.md"), ensure_ascii=False) + "\n")
        count += 1
    return count


def prepare_multidoc2dial(root: Path, source: Path | None = None) -> int:
    output = root / "data/documents/multidoc2dial"
    if source is None and any(output.rglob("*.md")):
        return sum(1 for _ in output.rglob("*.md"))
    source = source or download(MULTIDOC_URL, root / "data/raw/multidoc2dial.zip")
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            candidates = [name for name in archive.namelist() if Path(name).name == "multidoc2dial_doc.json"]
            if len(candidates) != 1:
                raise ValueError("Archive must contain exactly one multidoc2dial_doc.json.")
            data = json.loads(archive.read(candidates[0]).decode("utf-8"))
    else:
        data = json.loads(source.read_text(encoding="utf-8"))
    count = 0
    for domain, documents in data["doc_data"].items():
        for identifier, document in documents.items():
            safe_domain = re.sub(r"[^a-zA-Z0-9_-]", "_", domain)
            safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", identifier)
            target = output / safe_domain / f"{safe_id}.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"# {document.get('title', identifier)}\n\n{document['doc_text']}\n", encoding="utf-8")
            count += 1
    if not count:
        raise ValueError("No MultiDoc2Dial documents found.")
    return count
