from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .models import ChildChunk, ParentChunk, SearchHit
from .text import chunk_document, load_documents, tokenize


class HashingEmbedder:
    """Deterministic signed feature hashing for an offline dense baseline."""

    def __init__(self, dimensions: int = 192):
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            token_hash = hash(token)
            index = token_hash % self.dimensions
            vector[index] += 1.0 if token_hash & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class HybridIndex:
    def __init__(self, embedder: HashingEmbedder | None = None):
        self.embedder = embedder or HashingEmbedder()
        self.parents: dict[str, ParentChunk] = {}
        self.children: dict[str, ChildChunk] = {}
        self._tokens: dict[str, list[str]] = {}
        self._vectors: dict[str, list[float]] = {}
        self._document_frequency: Counter[str] = Counter()
        self._average_length = 1.0

    def add_text(
        self,
        source: str,
        text: str,
        metadata: dict[str, str] | None = None,
    ) -> int:
        parents, children = chunk_document(source, text, metadata)
        for parent in parents:
            self.parents[parent.id] = parent
        for child in children:
            self.children[child.id] = child
        self._rebuild_statistics()
        return len(children)

    def ingest_path(self, path: str | Path) -> int:
        count = 0
        for source, text in load_documents(path):
            count += self.add_text(source, text)
        return count

    def _rebuild_statistics(self) -> None:
        self._tokens = {
            chunk_id: tokenize(chunk.text)
            for chunk_id, chunk in self.children.items()
        }
        self._vectors = {
            chunk_id: self.embedder.embed(chunk.text)
            for chunk_id, chunk in self.children.items()
        }
        self._document_frequency = Counter()
        for tokens in self._tokens.values():
            self._document_frequency.update(set(tokens))
        lengths = [len(tokens) for tokens in self._tokens.values()]
        self._average_length = sum(lengths) / len(lengths) if lengths else 1.0

    def _bm25(self, query_tokens: list[str], chunk_id: str) -> float:
        tokens = self._tokens[chunk_id]
        frequencies = Counter(tokens)
        total_documents = max(1, len(self.children))
        k1, b = 1.5, 0.75
        score = 0.0
        for token in set(query_tokens):
            frequency = frequencies[token]
            if not frequency:
                continue
            document_frequency = self._document_frequency[token]
            idf = math.log(1 + (total_documents - document_frequency + 0.5) /
                           (document_frequency + 0.5))
            denominator = frequency + k1 * (
                1 - b + b * len(tokens) / self._average_length
            )
            score += idf * frequency * (k1 + 1) / denominator
        return score

    def search(self, query: str, limit: int = 5) -> list[SearchHit]:
        if not self.children:
            return []
        query_tokens = tokenize(query)
        query_vector = self.embedder.embed(query)
        raw = []
        for chunk_id, child in self.children.items():
            sparse = self._bm25(query_tokens, chunk_id)
            dense = max(0.0, cosine(query_vector, self._vectors[chunk_id]))
            overlap = len(set(query_tokens) & set(self._tokens[chunk_id]))
            lexical_bonus = overlap / max(1, len(set(query_tokens)))
            raw.append((chunk_id, sparse, dense, lexical_bonus, child))

        max_sparse = max((item[1] for item in raw), default=1.0) or 1.0
        ranked = []
        for chunk_id, sparse, dense, bonus, child in raw:
            normalized_sparse = sparse / max_sparse
            score = 0.55 * normalized_sparse + 0.35 * dense + 0.10 * bonus
            ranked.append(
                SearchHit(
                    chunk_id=chunk_id,
                    parent_id=child.parent_id,
                    source=child.source,
                    section=child.section,
                    text=child.text,
                    score=round(score, 6),
                    sparse_score=round(normalized_sparse, 6),
                    dense_score=round(dense, 6),
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:limit]

    def parent_text(self, parent_id: str) -> str:
        parent = self.parents.get(parent_id)
        return parent.text if parent else ""

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "parents": [asdict(item) for item in self.parents.values()],
            "children": [asdict(item) for item in self.children.values()],
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                          encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HybridIndex":
        source = Path(path)
        index = cls()
        if not source.exists():
            return index
        payload = json.loads(source.read_text(encoding="utf-8"))
        index.parents = {
            item["id"]: ParentChunk(**item) for item in payload.get("parents", [])
        }
        index.children = {
            item["id"]: ChildChunk(**item) for item in payload.get("children", [])
        }
        index._rebuild_statistics()
        return index

