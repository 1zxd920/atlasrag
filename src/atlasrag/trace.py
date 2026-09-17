from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TraceEvent:
    trace_id: str
    stage: str
    timestamp: float
    elapsed_ms: float
    payload: dict[str, Any]


class TraceRecorder:
    def __init__(self, output_path: str | Path | None = None):
        self.output_path = Path(output_path) if output_path else None
        self.trace_id = uuid.uuid4().hex[:16]
        self.started = time.perf_counter()
        self.events: list[TraceEvent] = []

    def record(self, stage: str, **payload: Any) -> None:
        event = TraceEvent(
            trace_id=self.trace_id,
            stage=stage,
            timestamp=time.time(),
            elapsed_ms=round((time.perf_counter() - self.started) * 1000, 3),
            payload=payload,
        )
        self.events.append(event)
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

