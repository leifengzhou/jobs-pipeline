from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class FitDecision:
    decision: str
    score: float
    reason: str


class ClassifierClient(Protocol):
    def classify(self, system_prompt: str, user_message: str, model: str) -> FitDecision: ...


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


def build_user_message(job: dict[str, object]) -> str:
    fields = [
        f"Title: {job.get('title', '')}",
        f"Company: {job.get('company', '')}",
        f"Location: {job.get('location', '')}",
        f"Salary: {job.get('salary', '')}",
        "Description:",
        str(job.get("description", "")),
    ]
    return "\n".join(fields)


def process_job_file(
    job_path: Path,
    system_prompt: str,
    client: ClassifierClient,
    model: str,
    good_fit_dir: Path,
    no_good_fit_dir: Path,
) -> Path:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    result = client.classify(system_prompt, build_user_message(job), model)

    job["fitDecision"] = result.decision
    job["fitScore"] = result.score
    job["fitReason"] = result.reason
    job["fitModel"] = model
    job["fitEvaluatedAt"] = datetime.now(timezone.utc).isoformat()

    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")

    destination_dir = good_fit_dir if result.decision == "good_fit" else no_good_fit_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / job_path.name
    job_path.replace(destination_path)
    return destination_path
