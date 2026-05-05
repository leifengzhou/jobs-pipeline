from __future__ import annotations

import json
import logging
import sys
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol, Sequence
from urllib import error, request

DEFAULT_ENV_FILE = Path("~/.secrets/deepseek.env").expanduser()
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_DIR = Path("logs/job-classifier")
DEFAULT_ERROR_ARTIFACT_DIR = Path(".job-classifier-errors")
LOGGER_NAME = "jobs_pipeline"


@dataclass(frozen=True)
class FitDecision:
    decision: str
    score: float
    reason: str


class ModelResponseError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        raw_response: str | None = None,
        response_payload: object | None = None,
        request_payload: object | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.response_payload = response_payload
        self.request_payload = request_payload


class JobProcessingError(RuntimeError):
    def __init__(
        self,
        *,
        job_path: Path,
        model: str,
        system_prompt: str | None,
        user_message: str | None,
        cause: Exception,
    ) -> None:
        super().__init__(str(cause))
        self.job_path = job_path
        self.model = model
        self.system_prompt = system_prompt
        self.user_message = user_message
        self.cause = cause


class ClassifierClient(Protocol):
    def classify(self, system_prompt: str, user_message: str, model: str) -> FitDecision: ...


class DeepSeekClient:
    def __init__(self, api_key: str, api_url: str = DEFAULT_API_URL, timeout: int = 60) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.timeout = timeout

    def classify(self, system_prompt: str, user_message: str, model: str) -> FitDecision:
        logger = logging.getLogger(LOGGER_NAME)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 800,
        }
        logger.info(
            "LLM request payload: %s",
            json.dumps(payload, ensure_ascii=False),
        )
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
                response_body = response.read().decode("utf-8", errors="replace")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            logger.error("LLM HTTP error response: %s", detail)
            raise RuntimeError(f"DeepSeek API error: {exc.code} {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"DeepSeek request failed: {exc.reason}") from exc

        logger.info("LLM raw response: %s", response_body)
        try:
            response_payload = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise ModelResponseError(
                f"DeepSeek returned invalid API JSON: {exc}",
                raw_response=response_body,
                request_payload=payload,
            ) from exc

        try:
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelResponseError(
                "DeepSeek response was missing choices/message/content",
                response_payload=response_payload,
                request_payload=payload,
            ) from exc
        if not content:
            raise ModelResponseError(
                "DeepSeek returned an empty response body",
                response_payload=response_payload,
                request_payload=payload,
            )
        logger.info("LLM message content: %s", content)
        return parse_llm_response(content)


def discover_job_files(source_dir: Path, limit: int | None = None) -> list[Path]:
    job_files = sorted(path for path in source_dir.iterdir() if path.suffix == ".json")
    if limit is None:
        return job_files
    return job_files[:limit]


def derive_company_role_key(job_path: Path) -> str:
    stem = job_path.stem
    prefix, separator, _ = stem.rpartition("_")
    return prefix if separator else stem


def collect_company_role_keys(source_dir: Path) -> set[str]:
    if not source_dir.exists():
        return set()
    return {derive_company_role_key(path) for path in discover_job_files(source_dir)}


def collect_duplicate_company_role_keys(*source_dirs: Path) -> set[str]:
    duplicate_keys: set[str] = set()
    for source_dir in source_dirs:
        duplicate_keys.update(collect_company_role_keys(source_dir))
    return duplicate_keys


def move_duplicate_job_file(job_path: Path, duplicates_dir: Path) -> Path:
    duplicates_dir.mkdir(parents=True, exist_ok=True)
    destination_path = duplicates_dir / job_path.name
    job_path.replace(destination_path)
    return destination_path


def collect_job_urls(job: dict[str, object]) -> list[str]:
    urls: list[str] = []

    apply_url = job.get("applyUrl")
    if isinstance(apply_url, str) and apply_url.strip():
        urls.append(apply_url.strip())

    hiring_team = job.get("hiringTeam")
    if isinstance(hiring_team, list):
        for member in hiring_team:
            if not isinstance(member, dict):
                continue
            linkedin_url = member.get("linkedinUrl")
            if isinstance(linkedin_url, str) and linkedin_url.strip():
                urls.append(linkedin_url.strip())

    return urls


def open_job_file(
    job_path: Path,
    destination_dir: Path,
    opener: Callable[[str], bool] | None = None,
) -> Path:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    urls = collect_job_urls(job)
    if not urls:
        raise ValueError(f"No URLs found for {job_path.name}")

    open_url = opener or webbrowser.open
    for url in urls:
        open_url(url)

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / job_path.name
    job_path.replace(destination_path)
    return destination_path


def parse_llm_response(text: str) -> FitDecision:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ModelResponseError(
            f"Malformed model response JSON: {exc}",
            raw_response=text,
        ) from exc
    decision = payload["decision"]
    score = payload["score"]
    reason = payload["reason"]
    if decision not in {"good_fit", "no_good_fit"}:
        raise ModelResponseError(
            f"Unexpected decision: {decision}",
            raw_response=text,
            response_payload=payload,
        )
    if not isinstance(score, (int, float)):
        raise ModelResponseError(
            "score must be numeric",
            raw_response=text,
            response_payload=payload,
        )
    if not isinstance(reason, str) or not reason.strip():
        raise ModelResponseError(
            "reason must be non-empty text",
            raw_response=text,
            response_payload=payload,
        )
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
    user_message = build_user_message(job)
    try:
        result = client.classify(system_prompt, user_message, model)
    except Exception as exc:
        raise JobProcessingError(
            job_path=job_path,
            model=model,
            system_prompt=system_prompt,
            user_message=user_message,
            cause=exc,
        ) from exc

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


def configure_logging(level_name: str = DEFAULT_LOG_LEVEL, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = False

    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {level_name}")
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def default_log_file_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return DEFAULT_LOG_DIR / f"{timestamp}.log"


def write_failure_artifact(
    artifact_dir: Path,
    *,
    job_path: Path,
    model: str,
    system_prompt: str | None,
    user_message: str | None,
    error: Exception,
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifact_path = artifact_dir / f"{timestamp}-{job_path.stem}.json"

    raw_response: str | None = None
    response_payload: object | None = None
    request_payload: object | None = None
    if isinstance(error, ModelResponseError):
        raw_response = error.raw_response
        response_payload = error.response_payload
        request_payload = error.request_payload

    artifact = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "jobFile": job_path.name,
        "jobPath": str(job_path),
        "model": model,
        "errorType": type(error).__name__,
        "errorMessage": str(error),
        "systemPrompt": system_prompt,
        "userMessage": user_message,
        "requestPayload": request_payload,
        "rawResponse": raw_response,
        "responsePayload": response_payload,
    }
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact_path


def run_batch(
    source_dir: Path,
    good_fit_dir: Path,
    no_good_fit_dir: Path,
    system_prompt: str,
    client: ClassifierClient,
    model: str,
    opened_or_applied_dir: Path = Path("04-opened-or-applied"),
    duplicates_dir: Path = Path("05-duplicates"),
    limit: int | None = None,
    error_artifact_dir: Path | None = DEFAULT_ERROR_ARTIFACT_DIR,
    logger: logging.Logger | None = None,
) -> tuple[int, int, int, int]:
    batch_logger = logger or logging.getLogger(LOGGER_NAME)
    processed = 0
    good_fit = 0
    no_good_fit = 0
    errors = 0
    duplicate_company_role_keys = collect_duplicate_company_role_keys(
        opened_or_applied_dir,
        no_good_fit_dir,
    )

    for job_path in discover_job_files(source_dir, limit=limit):
        if derive_company_role_key(job_path) in duplicate_company_role_keys:
            destination = move_duplicate_job_file(job_path, duplicates_dir)
            processed += 1
            batch_logger.info(
                "%s -> %s (duplicate company+role match in %s or %s)",
                job_path.name,
                destination.parent.name,
                opened_or_applied_dir.name,
                no_good_fit_dir.name,
            )
            continue

        batch_logger.info("Processing %s with model=%s", job_path.name, model)
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
            artifact_path: Path | None = None
            artifact_model = model
            artifact_system_prompt: str | None = system_prompt
            artifact_user_message: str | None = None
            root_error = exc
            if isinstance(exc, JobProcessingError):
                artifact_model = exc.model
                artifact_system_prompt = exc.system_prompt
                artifact_user_message = exc.user_message
                root_error = exc.cause
            if error_artifact_dir is not None:
                artifact_path = write_failure_artifact(
                    error_artifact_dir,
                    job_path=job_path,
                    model=artifact_model,
                    system_prompt=artifact_system_prompt,
                    user_message=artifact_user_message,
                    error=root_error,
                )
            batch_logger.exception("Failed to process %s: %s", job_path.name, root_error)
            if artifact_path is not None:
                batch_logger.error("Saved failure artifact for %s to %s", job_path.name, artifact_path)
            continue

        processed += 1
        if destination.parent == good_fit_dir:
            good_fit += 1
        else:
            no_good_fit += 1
        batch_logger.info("%s -> %s", job_path.name, destination.parent.name)

    return processed, good_fit, no_good_fit, errors


def run_open_batch(
    source_dir: Path,
    destination_dir: Path,
    limit: int | None = None,
    opener: Callable[[str], bool] | None = None,
    logger: logging.Logger | None = None,
) -> tuple[int, int]:
    batch_logger = logger or logging.getLogger(LOGGER_NAME)
    processed = 0
    errors = 0

    for job_path in discover_job_files(source_dir, limit=limit):
        batch_logger.info("Opening %s", job_path.name)
        try:
            destination = open_job_file(
                job_path=job_path,
                destination_dir=destination_dir,
                opener=opener,
            )
        except Exception as exc:
            errors += 1
            batch_logger.exception("Failed to open %s: %s", job_path.name, exc)
            continue

        processed += 1
        batch_logger.info("%s -> %s", job_path.name, destination.parent.name)

    return processed, errors


def main(argv: Sequence[str] | None = None, client: ClassifierClient | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Classify scraped jobs with DeepSeek.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source-dir", default="01-source-jobs")
    parser.add_argument("--good-fit-dir", default="02-good-fit")
    parser.add_argument("--no-good-fit-dir", default="03-no-good-fit")
    parser.add_argument("--opened-or-applied-dir", default="04-opened-or-applied")
    parser.add_argument("--duplicates-dir", default="05-duplicates")
    parser.add_argument("--prompt-file", default="prompts/job_fit_system_prompt.txt")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--log-level", default=DEFAULT_LOG_LEVEL)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--error-artifact-dir", default=str(DEFAULT_ERROR_ARTIFACT_DIR))
    args = parser.parse_args(list(argv) if argv is not None else None)

    log_file = Path(args.log_file).expanduser() if args.log_file else default_log_file_path()
    try:
        logger = configure_logging(
            level_name=args.log_level,
            log_file=log_file,
        )
    except ValueError as exc:
        print(f"Startup error: {exc}", file=sys.stderr)
        return 1

    try:
        env_values = load_env_file(Path(args.env_file).expanduser())
        system_prompt = load_prompt_file(Path(args.prompt_file))
        api_key = env_values["DEEPSEEK_API_KEY"]
    except Exception as exc:
        logger.error("Startup error: %s", exc)
        return 1

    model = args.model or env_values.get("DEEPSEEK_MODEL", DEFAULT_MODEL)
    api_url = args.api_url or env_values.get("DEEPSEEK_API_URL", DEFAULT_API_URL)
    batch_client = client or DeepSeekClient(api_key=api_key, api_url=api_url)
    processed, good_fit, no_good_fit, errors = run_batch(
        source_dir=Path(args.source_dir),
        good_fit_dir=Path(args.good_fit_dir),
        no_good_fit_dir=Path(args.no_good_fit_dir),
        opened_or_applied_dir=Path(args.opened_or_applied_dir),
        duplicates_dir=Path(args.duplicates_dir),
        system_prompt=system_prompt,
        client=batch_client,
        model=model,
        limit=args.limit,
        error_artifact_dir=Path(args.error_artifact_dir).expanduser(),
        logger=logger,
    )
    logger.info(
        "Processed=%s good_fit=%s no_good_fit=%s errors=%s",
        processed,
        good_fit,
        no_good_fit,
        errors,
    )
    return 0


def open_jobs_main(
    argv: Sequence[str] | None = None,
    opener: Callable[[str], bool] | None = None,
) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Open good-fit job application URLs.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source-dir", default="02-good-fit")
    parser.add_argument("--destination-dir", default="04-opened-or-applied")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    processed, errors = run_open_batch(
        source_dir=Path(args.source_dir),
        destination_dir=Path(args.destination_dir),
        limit=args.limit,
        opener=opener,
        logger=logger,
    )
    logger.info("Processed=%s errors=%s", processed, errors)
    return 0
