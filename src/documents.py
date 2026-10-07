from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from docx import Document
from pypdf import PdfReader


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if suffix == ".docx":
        return "\n".join(paragraph.text for paragraph in Document(path).paragraphs)
    raise ValueError(f"Unsupported document type: {path.suffix}")


def load_chunks(folder: Path, extensions: tuple[str, ...], size: int, overlap: int) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks: list[Chunk] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        text = " ".join(read_document(path).split())
        step = size - overlap
        for start in range(0, len(text), step):
            piece = text[start : start + size].strip()
            if piece:
                chunks.append(Chunk(piece, str(path.relative_to(folder))))
    return chunks