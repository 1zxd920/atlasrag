from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ParentChunk:
    id: str
    source: str
    section: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ChildChunk:
    id: str
    parent_id: str
    source: str
    section: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class SearchHit:
    chunk_id: str
    parent_id: str
    source: str
    section: str
    text: str
    score: float
    sparse_score: float
    dense_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Citation:
    number: int
    source: str
    section: str
    excerpt: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QueryResult:
    question: str
    rewritten_question: str
    answer: str
    citations: list[Citation]
    hits: list[SearchHit]
    clarification_needed: bool = False
    trace_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "rewritten_question": self.rewritten_question,
            "answer": self.answer,
            "citations": [item.to_dict() for item in self.citations],
            "hits": [item.to_dict() for item in self.hits],
            "clarification_needed": self.clarification_needed,
            "trace_id": self.trace_id,
        }


@dataclass(slots=True)
class EvaluationSample:
    question: str
    expected_source: str
    expected_keywords: list[str]

