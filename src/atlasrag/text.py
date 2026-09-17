from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .models import ChildChunk, ParentChunk


ASCII_WORD = re.compile(r"[A-Za-z0-9_]+")
CJK_CHAR = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """Tokenize English words and Chinese unigrams/bigrams."""
    lowered = text.lower()
    tokens = ASCII_WORD.findall(lowered)
    chars = CJK_CHAR.findall(lowered)
    tokens.extend(chars)
    tokens.extend(a + b for a, b in zip(chars, chars[1:]))
    return tokens


def stable_id(*parts: str) -> str:
    raw = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _markdown_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    title = "Document"
    buffer: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            if buffer and "\n".join(buffer).strip():
                sections.append((title, "\n".join(buffer).strip()))
            title = line.lstrip("# ").strip() or "Untitled"
            buffer = []
        else:
            buffer.append(line)
    if buffer and "\n".join(buffer).strip():
        sections.append((title, "\n".join(buffer).strip()))
    return sections or [(title, text.strip())]


def _window_text(text: str, size: int, overlap: int) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= size:
        return [normalized] if normalized else []
    chunks = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + size)
        if end < len(normalized):
            boundary = normalized.rfind(". ", start, end)
            if boundary > start + size // 2:
                end = boundary + 1
        chunks.append(normalized[start:end].strip())
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


def chunk_document(
    source: str,
    text: str,
    metadata: dict[str, str] | None = None,
    child_size: int = 420,
    child_overlap: int = 70,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []
    metadata = dict(metadata or {})
    for section_index, (section, body) in enumerate(_markdown_sections(text)):
        parent_id = stable_id(source, str(section_index), section)
        parent = ParentChunk(parent_id, source, section, body, metadata)
        parents.append(parent)
        for child_index, child_text in enumerate(
            _window_text(body, child_size, child_overlap)
        ):
            child_id = stable_id(parent_id, str(child_index), child_text)
            children.append(
                ChildChunk(
                    child_id,
                    parent_id,
                    source,
                    section,
                    child_text,
                    metadata,
                )
            )
    return parents, children


def load_documents(path: str | Path) -> list[tuple[str, str]]:
    candidate = Path(path)
    files = [candidate] if candidate.is_file() else sorted(candidate.rglob("*"))
    documents = []
    for file_path in files:
        if file_path.is_file() and file_path.suffix.lower() in {".md", ".txt"}:
            documents.append((file_path.name, file_path.read_text(encoding="utf-8")))
    return documents

