from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FitDecision:
    decision: str
    score: float
    reason: str


def discover_job_files(source_dir: Path, limit: int | None = None) -> list[Path]:
    job_files = sorted(path for path in source_dir.iterdir() if path.suffix == ".json")
    if limit is None:
        return job_files
    return job_files[:limit]


def parse_llm_response(text: str) -> FitDecision:
    payload = json.loads(text)
    decision = payload["decision"]
    score = payload["score"]
    reason = payload["reason"]
    return FitDecision(decision=decision, score=score, reason=reason)
