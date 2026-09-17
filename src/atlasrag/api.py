from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .agent import AtlasAgent
from .index import HybridIndex


class AtlasRequestHandler(BaseHTTPRequestHandler):
    index_path = Path("data/index.json")
    index = HybridIndex()

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path == "/health":
            self._json(HTTPStatus.OK, {
                "status": "ok",
                "parents": len(self.index.parents),
                "children": len(self.index.children),
            })
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._body()
            path = urlparse(self.path).path
            if path == "/query":
                question = str(payload.get("question", "")).strip()
                if not question:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": "question_required"})
                    return
                result = AtlasAgent(self.index, "data/traces.jsonl").query(
                    question,
                    history=list(payload.get("history", [])),
                )
                self._json(HTTPStatus.OK, result.to_dict())
                return
            if path == "/ingest":
                source_path = str(payload.get("path", "")).strip()
                if not source_path:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": "path_required"})
                    return
                count = self.index.ingest_path(source_path)
                self.index.save(self.index_path)
                self._json(HTTPStatus.OK, {"ingested_chunks": count})
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def log_message(self, format: str, *args) -> None:
        return


def serve(host: str, port: int, index_path: str) -> None:
    AtlasRequestHandler.index_path = Path(index_path)
    AtlasRequestHandler.index = HybridIndex.load(index_path)
    server = ThreadingHTTPServer((host, port), AtlasRequestHandler)
    print(f"AtlasRAG listening on http://{host}:{port}")
    server.serve_forever()

