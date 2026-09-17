from __future__ import annotations

import re
from collections import defaultdict

from .index import HybridIndex
from .models import Citation, QueryResult, SearchHit
from .text import tokenize
from .trace import TraceRecorder


AMBIGUOUS_ENGLISH = {"it", "that", "this", "thing"}
AMBIGUOUS_CHINESE = {"那个", "这个", "它", "该内容"}
SPLIT_PATTERN = re.compile(r"\s+(?:and|plus)\s+|[;；]|以及|并且", re.IGNORECASE)


def contains_ambiguous_reference(text: str) -> bool:
    english_words = set(re.findall(r"[a-z]+", text.lower()))
    return bool(english_words & AMBIGUOUS_ENGLISH) or any(
        term in text for term in AMBIGUOUS_CHINESE
    )


class AtlasAgent:
    def __init__(self, index: HybridIndex, trace_path: str | None = None):
        self.index = index
        self.trace_path = trace_path

    def _rewrite(self, question: str, history: list[str]) -> str:
        normalized = re.sub(r"\s+", " ", question).strip()
        if history and contains_ambiguous_reference(normalized):
            return f"{history[-1]} Follow-up question: {normalized}"
        return normalized

    def _needs_clarification(self, question: str, history: list[str]) -> bool:
        tokens = tokenize(question)
        ambiguous = contains_ambiguous_reference(question)
        return len(tokens) < 2 or (ambiguous and not history)

    def _decompose(self, question: str) -> list[str]:
        parts = [part.strip(" ,.?\u3002") for part in SPLIT_PATTERN.split(question)]
        return [part for part in parts if part] or [question]

    def _retrieve(self, questions: list[str], limit: int) -> list[SearchHit]:
        merged: dict[str, SearchHit] = {}
        for question in questions:
            for hit in self.index.search(question, limit=limit):
                current = merged.get(hit.chunk_id)
                if not current or hit.score > current.score:
                    merged[hit.chunk_id] = hit
        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return ranked[:limit]

    def _retry_query(self, question: str) -> str:
        tokens = [token for token in tokenize(question) if len(token) > 1]
        return " ".join(dict.fromkeys(tokens)) or question

    def _answer(self, question: str, hits: list[SearchHit]) -> tuple[str, list[Citation]]:
        if not hits or hits[0].score < 0.08:
            return (
                "I could not find enough evidence in the indexed documents. "
                "Please add a source or make the question more specific.",
                [],
            )

        question_tokens = set(tokenize(question))
        chosen: list[tuple[SearchHit, str]] = []
        seen_parents = set()
        for hit in hits:
            if hit.parent_id in seen_parents:
                continue
            parent = self.index.parent_text(hit.parent_id) or hit.text
            sentences = re.split(r"(?<=[.!?。！？])\s*", parent)
            sentence = max(
                (item.strip() for item in sentences if item.strip()),
                key=lambda item: len(question_tokens & set(tokenize(item))),
                default=hit.text,
            )
            chosen.append((hit, sentence))
            seen_parents.add(hit.parent_id)
            if len(chosen) == 3:
                break

        citations = [
            Citation(index, hit.source, hit.section, sentence[:220])
            for index, (hit, sentence) in enumerate(chosen, 1)
        ]
        answer_parts = [f"{sentence} [{index}]" for index, (_, sentence) in enumerate(chosen, 1)]
        sources = " ".join(
            f"[{item.number}] {item.source}#{item.section}" for item in citations
        )
        return " ".join(answer_parts) + "\n\nSources: " + sources, citations

    def query(
        self,
        question: str,
        history: list[str] | None = None,
        limit: int = 6,
    ) -> QueryResult:
        history = list(history or [])
        trace = TraceRecorder(self.trace_path)
        rewritten = self._rewrite(question, history)
        trace.record("rewrite", original=question, rewritten=rewritten)

        if self._needs_clarification(rewritten, history):
            trace.record("clarification", needed=True)
            return QueryResult(
                question=question,
                rewritten_question=rewritten,
                answer="Please clarify the document, policy, or operation you mean.",
                citations=[],
                hits=[],
                clarification_needed=True,
                trace_id=trace.trace_id,
            )

        sub_questions = self._decompose(rewritten)
        trace.record("decompose", questions=sub_questions)
        hits = self._retrieve(sub_questions, limit)
        trace.record("retrieve", hit_count=len(hits), top_score=hits[0].score if hits else 0)

        if not hits or hits[0].score < 0.12:
            retry = self._retry_query(rewritten)
            hits = self._retrieve([retry], limit)
            trace.record("retry", query=retry, hit_count=len(hits))

        answer, citations = self._answer(rewritten, hits)
        trace.record("answer", citation_count=len(citations), answer_length=len(answer))
        return QueryResult(
            question=question,
            rewritten_question=rewritten,
            answer=answer,
            citations=citations,
            hits=hits,
            trace_id=trace.trace_id,
        )
