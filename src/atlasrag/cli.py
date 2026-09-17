from __future__ import annotations

import argparse
import json

from .agent import AtlasAgent
from .api import serve
from .evaluation import evaluate, load_samples
from .index import HybridIndex


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="atlasrag")
    commands = root.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="Ingest a file or directory")
    ingest.add_argument("path")
    ingest.add_argument("--index", default="data/index.json")

    query = commands.add_parser("query", help="Query an existing index")
    query.add_argument("question")
    query.add_argument("--index", default="data/index.json")
    query.add_argument("--trace", default="data/traces.jsonl")

    evaluation = commands.add_parser("evaluate", help="Run a JSONL evaluation set")
    evaluation.add_argument("dataset")
    evaluation.add_argument("--index", default="data/index.json")
    evaluation.add_argument("--k", type=int, default=5)

    server = commands.add_parser("serve", help="Run the JSON HTTP API")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8080)
    server.add_argument("--index", default="data/index.json")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "ingest":
        index = HybridIndex.load(args.index)
        count = index.ingest_path(args.path)
        index.save(args.index)
        print(json.dumps({"ingested_chunks": count, "index": args.index}))
    elif args.command == "query":
        index = HybridIndex.load(args.index)
        result = AtlasAgent(index, args.trace).query(args.question)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    elif args.command == "evaluate":
        index = HybridIndex.load(args.index)
        report = evaluate(AtlasAgent(index), load_samples(args.dataset), args.k)
        print(json.dumps(report, indent=2))
    elif args.command == "serve":
        serve(args.host, args.port, args.index)

