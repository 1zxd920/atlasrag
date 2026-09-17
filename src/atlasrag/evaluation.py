from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from .agent import AtlasAgent
from .models import EvaluationSample


def load_samples(path: str | Path) -> list[EvaluationSample]:
    samples = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                samples.append(EvaluationSample(**json.loads(line)))
    return samples


def evaluate(agent: AtlasAgent, samples: list[EvaluationSample], k: int = 5) -> dict[str, float]:
    recalls = []
    reciprocal_ranks = []
    citation_rates = []
    coverages = []
    for sample in samples:
        result = agent.query(sample.question, limit=k)
        sources = [hit.source for hit in result.hits[:k]]
        ranks = [index + 1 for index, source in enumerate(sources)
                 if source == sample.expected_source]
        recalls.append(1.0 if ranks else 0.0)
        reciprocal_ranks.append(1.0 / ranks[0] if ranks else 0.0)
        citation_rates.append(1.0 if result.citations else 0.0)
        expected = [item.lower() for item in sample.expected_keywords]
        answer = result.answer.lower()
        coverages.append(
            sum(keyword in answer for keyword in expected) / max(1, len(expected))
        )
    metric = lambda values: round(mean(values), 4) if values else 0.0
    return {
        "samples": float(len(samples)),
        "recall_at_k": metric(recalls),
        "mrr": metric(reciprocal_ranks),
        "citation_rate": metric(citation_rates),
        "keyword_coverage": metric(coverages),
    }

