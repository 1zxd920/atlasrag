# AtlasRAG

AtlasRAG is a runnable, dependency-light reference project for an enterprise
knowledge agent. It focuses on the engineering parts that are often hidden by
framework demos: document structure, hybrid retrieval, citations, bounded
agent loops, evaluation, and traceability.

The default implementation uses only the Python standard library, so it runs
without an API key or external database. The interfaces are intentionally
small enough to replace with LangGraph, Qdrant, Redis, or an OpenAI-compatible
model in a production deployment.

## Features

- Markdown and text ingestion with parent/child chunking
- BM25 plus deterministic dense hashing retrieval
- Reciprocal score fusion and lexical reranking
- Query rewriting, ambiguity checks, multi-query decomposition, and one retry
- Extractive answers with source citations
- JSON trace events for every workflow stage
- Recall@K, MRR, citation rate, and keyword coverage evaluation
- CLI, JSON HTTP API, Docker image, example corpus, and unit tests

## Architecture

```text
documents -> parent sections -> child chunks -> hybrid index
                                              |
question -> rewrite -> clarify -> decompose -> retrieve -> rerank
                                              |             |
                                              +-- retry -----+
                                                            |
                                             cited extractive answer
                                                            |
                                                   trace + evaluation
```

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
python -m pip install -e .

atlasrag ingest examples/docs --index data/index.json
atlasrag query "How long are audit logs retained?" --index data/index.json
atlasrag evaluate examples/eval.jsonl --index data/index.json
atlasrag serve --index data/index.json --port 8080
```

On macOS/Linux, activate the environment with `source .venv/bin/activate`.

## HTTP API

```bash
curl http://localhost:8080/health

curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What should an operator do after a failed refund?"}'

curl -X POST http://localhost:8080/ingest \
  -H "Content-Type: application/json" \
  -d '{"path":"/docs"}'
```

## Evaluation

The included evaluation set is a smoke test, not a claim about production
model quality. Each sample specifies an expected source and answer keywords.
The evaluator reports:

- `recall_at_k`: expected source found in the top K results
- `mrr`: reciprocal rank of the first expected source
- `citation_rate`: responses containing at least one source citation
- `keyword_coverage`: expected answer keywords present in the response

For a real deployment, replace the sample set with reviewed business questions
and add human grading for correctness, groundedness, and refusal quality.

## Testing

```bash
python -m unittest discover -s tests -v
```

## Production extension points

- Replace `HashingEmbedder` with a hosted or local embedding model.
- Replace `HybridIndex` persistence with Qdrant hybrid search.
- Replace `AtlasAgent` orchestration with LangGraph nodes and checkpoints.
- Send `TraceRecorder` events to Langfuse or OpenTelemetry.
- Add authentication and document-level ACL checks in the API layer.

## License

MIT

