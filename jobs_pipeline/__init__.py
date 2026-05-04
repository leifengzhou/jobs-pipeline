from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Sequence
from urllib import error, request

DEFAULT_ENV_FILE = Path("~/.secrets/deepseek.env").expanduser()
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_API_URL = "https://api.deepseek.com/chat/completions"


@dataclass(frozen=True)
class FitDecision:
    decision: str
    score: float
    reason: str


class ClassifierClient(Protocol):
    def classify(self, system_prompt: str, user_message: str, model: str) -> FitDecision: ...


class DeepSeekClient:
    def __init__(self, api_key: str, api_url: str = DEFAULT_API_URL, timeout: int = 60) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.timeout = timeout

    def classify(self, system_prompt: str, user_message: str, model: str) -> FitDecision:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 400,
        }
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            self.api_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek API error: {exc.code} {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"DeepSeek request failed: {exc.reason}") from exc

        content = response_payload["choices"][0]["message"]["content"]
        if not content:
            raise ValueError("DeepSeek returned an empty response body")
        return parse_llm_response(content)


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
    if decision not in {"good_fit", "no_good_fit"}:
        raise ValueError(f"Unexpected decision: {decision}")
    if not isinstance(score, (int, float)):
        raise ValueError("score must be numeric")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be non-empty text")
    return FitDecision(decision=decision, score=float(score), reason=reason.strip())


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise ValueError(f"Invalid env line: {raw_line}")
        values[key.strip()] = value.strip()
    return values


def load_prompt_file(path: Path) -> str:
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {path}")
    return prompt


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


def run_batch(
    source_dir: Path,
    good_fit_dir: Path,
    no_good_fit_dir: Path,
    system_prompt: str,
    client: ClassifierClient,
    model: str,
    limit: int | None = None,
) -> tuple[int, int, int, int]:
    processed = 0
    good_fit = 0
    no_good_fit = 0
    errors = 0

    for job_path in discover_job_files(source_dir, limit=limit):
        try:
            destination = process_job_file(
                job_path=job_path,
                system_prompt=system_prompt,
                client=client,
                model=model,
                good_fit_dir=good_fit_dir,
                no_good_fit_dir=no_good_fit_dir,
            )
        except Exception as exc:
            errors += 1
            print(f"ERROR {job_path.name}: {exc}", file=sys.stderr)
            continue

        processed += 1
        if destination.parent == good_fit_dir:
            good_fit += 1
        else:
            no_good_fit += 1
        print(f"{job_path.name} -> {destination.parent.name}")

    return processed, good_fit, no_good_fit, errors


def main(argv: Sequence[str] | None = None, client: ClassifierClient | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Classify scraped jobs with DeepSeek.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source-dir", default="01-source-jobs")
    parser.add_argument("--good-fit-dir", default="02-good-fit")
    parser.add_argument("--no-good-fit-dir", default="03-no-good-fit")
    parser.add_argument("--prompt-file", default="prompts/job_fit_system_prompt.txt")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-url", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        env_values = load_env_file(Path(args.env_file).expanduser())
        system_prompt = load_prompt_file(Path(args.prompt_file))
        api_key = env_values["DEEPSEEK_API_KEY"]
    except Exception as exc:
        print(f"Startup error: {exc}", file=sys.stderr)
        return 1

    model = args.model or env_values.get("DEEPSEEK_MODEL", DEFAULT_MODEL)
    api_url = args.api_url or env_values.get("DEEPSEEK_API_URL", DEFAULT_API_URL)
    batch_client = client or DeepSeekClient(api_key=api_key, api_url=api_url)
    processed, good_fit, no_good_fit, errors = run_batch(
        source_dir=Path(args.source_dir),
        good_fit_dir=Path(args.good_fit_dir),
        no_good_fit_dir=Path(args.no_good_fit_dir),
        system_prompt=system_prompt,
        client=batch_client,
        model=model,
        limit=args.limit,
    )
    print(
        f"Processed={processed} good_fit={good_fit} no_good_fit={no_good_fit} errors={errors}"
    )
    return 0
