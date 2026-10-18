from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from docx import Document
from pypdf import PdfReader
from .progress import Progress
from .anime_tools import AnimeTools


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    anime_id: str | None = None
    entity_type: str | None = None
    character_id: str | None = None


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if suffix == ".docx":
        return "\n".join(paragraph.text for paragraph in Document(path).paragraphs)
    raise ValueError(f"Unsupported document type: {path.suffix}")


def load_chunks(folder: Path, extensions: tuple[str, ...], size: int, overlap: int,
                progress: Progress | None = None) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks: list[Chunk] = []
    if progress:
        progress("Scanning documents", 0, 1)
    paths = sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in extensions)
    catalog = AnimeTools.load(folder) if folder.name == "anilist" else None
    if progress:
        progress("Scanning documents", 1, 1)
        progress("Loading documents", 0, len(paths))
    for number, path in enumerate(paths, 1):
        record = catalog.record_by_source(path.name) if catalog else None
        if record:
            header = f"Anime: {record['title']}. Aliases: {', '.join(record['aliases'])}."
            sections = [("Synopsis", record['description'], "anime", None),
                        ("Metadata", f"Genres: {', '.join(record['genres'])}. Tags: {', '.join(record['tags'])}. "
                         f"Popularity: {record['popularity']}; score: {record['score']}; year: {record['year']}.", "metadata", None)]
            for char in record["characters"]:
                sections.append((f"Character: {char['name']}. Names: {', '.join(char['aliases'])}. Role: {char['role']}",
                                 char['description'] or "No description provided in the dataset.", "character",
                                 str(char.get('id') or 'name:' + char['name'])))
            for label, section, entity_type, character_id in sections:
                section = " ".join(section.split())
                for start in range(0, len(section), size - overlap):
                    chunks.append(Chunk(header + "\n" + label + ": " + section[start:start + size], path.name,
                                        record['id'], entity_type, character_id))
            if progress:
                progress("Loading documents", number, len(paths))
            continue
        text = " ".join(read_document(path).split())
        step = size - overlap
        for start in range(0, len(text), step):
            piece = text[start : start + size].strip()
            if piece:
                chunks.append(Chunk(piece, str(path.relative_to(folder))))
        if progress:
            progress("Loading documents", number, len(paths))
    return chunks
