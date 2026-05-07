import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock
from urllib import request

from jobs_pipeline import (
    DeepSeekClient,
    FitDecision,
    ModelResponseError,
    collect_job_urls,
    configure_logging,
    discover_job_files,
    load_env_file,
    main,
    open_job_file,
    open_jobs_main,
    parse_llm_response,
    process_job_file,
    run_batch,
)


class FakeClassifier:
    def __init__(self, decision: FitDecision) -> None:
        self.decision = decision
        self.calls: list[tuple[str, str, str]] = []

    def classify(self, system_prompt: str, user_message: str, model: str) -> FitDecision:
        self.calls.append((system_prompt, user_message, model))
        return self.decision


class RaisingClassifier:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def classify(self, system_prompt: str, user_message: str, model: str) -> FitDecision:
        raise self.exc


@contextmanager
def temporary_cwd(path: Path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


class JobFitClassifierTests(unittest.TestCase):
    def test_discover_job_files_sorts_and_applies_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp)
            (source_dir / "b.json").write_text("{}", encoding="utf-8")
            (source_dir / "a.json").write_text("{}", encoding="utf-8")
            (source_dir / "ignore.txt").write_text("", encoding="utf-8")

            files = discover_job_files(source_dir, limit=2)

            self.assertEqual([path.name for path in files], ["a.json", "b.json"])

    def test_parse_llm_response_requires_expected_schema(self) -> None:
        result = parse_llm_response(
            '{"decision":"good_fit","score":88,"reason":"Strong match"}'
        )

        self.assertEqual(result.decision, "good_fit")
        self.assertEqual(result.score, 88)
        self.assertEqual(result.reason, "Strong match")

    def test_parse_llm_response_preserves_raw_text_on_json_error(self) -> None:
        raw_response = '{"decision":"good_fit","score":88,"reason":"Missing quote}'

        with self.assertRaises(ModelResponseError) as context:
            parse_llm_response(raw_response)

        self.assertEqual(context.exception.raw_response, raw_response)
        self.assertIn("Malformed model response JSON", str(context.exception))

    def test_process_job_file_updates_metadata_and_moves_good_fit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "01-source-jobs"
            good_fit_dir = root / "02-good-fit"
            no_good_fit_dir = root / "03-no-good-fit"
            source_dir.mkdir()
            good_fit_dir.mkdir()
            no_good_fit_dir.mkdir()

            job_path = source_dir / "job.json"
            job_path.write_text(
                (
                    '{"title":"Product Manager","company":"Acme","location":"Remote",'
                    '"description":"Own roadmap"}'
                ),
                encoding="utf-8",
            )

            classifier = FakeClassifier(FitDecision("good_fit", 91, "Strong fit"))

            destination = process_job_file(
                job_path=job_path,
                system_prompt="system prompt",
                client=classifier,
                model="deepseek-v4-flash",
                good_fit_dir=good_fit_dir,
                no_good_fit_dir=no_good_fit_dir,
            )

            self.assertEqual(destination.parent.name, "02-good-fit")
            self.assertFalse(job_path.exists())
            self.assertTrue(destination.exists())

    def test_load_env_file_reads_key_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / "deepseek.env"
            env_path.write_text(
                "# comment\nDEEPSEEK_API_KEY=test-key\nDEEPSEEK_MODEL=deepseek-v4-flash\n",
                encoding="utf-8",
            )

            values = load_env_file(env_path)

            self.assertEqual(values["DEEPSEEK_API_KEY"], "test-key")
            self.assertEqual(values["DEEPSEEK_MODEL"], "deepseek-v4-flash")

    def test_cli_processes_a_single_file_with_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "01-source-jobs"
            good_fit_dir = root / "02-good-fit"
            no_good_fit_dir = root / "03-no-good-fit"
            prompt_file = root / "prompt.txt"
            env_file = root / "deepseek.env"

            source_dir.mkdir()
            good_fit_dir.mkdir()
            no_good_fit_dir.mkdir()
            prompt_file.write_text("Return json.", encoding="utf-8")
            env_file.write_text("DEEPSEEK_API_KEY=test-key\n", encoding="utf-8")
            (source_dir / "b.json").write_text(
                '{"title":"B","company":"Acme","location":"Remote","description":"Desc B"}',
                encoding="utf-8",
            )
            (source_dir / "a.json").write_text(
                '{"title":"A","company":"Acme","location":"Remote","description":"Desc A"}',
                encoding="utf-8",
            )

            with temporary_cwd(root):
                exit_code = main(
                    [
                        "--limit",
                        "1",
                        "--source-dir",
                        str(source_dir),
                        "--good-fit-dir",
                        str(good_fit_dir),
                        "--no-good-fit-dir",
                        str(no_good_fit_dir),
                        "--prompt-file",
                        str(prompt_file),
                        "--env-file",
                        str(env_file),
                    ],
                    client=FakeClassifier(FitDecision("good_fit", 93, "Worth reviewing")),
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue((good_fit_dir / "a.json").exists())
            self.assertTrue((source_dir / "b.json").exists())

    def test_cli_uses_model_from_env_when_flag_is_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "01-source-jobs"
            good_fit_dir = root / "02-good-fit"
            no_good_fit_dir = root / "03-no-good-fit"
            prompt_file = root / "prompt.txt"
            env_file = root / "deepseek.env"

            source_dir.mkdir()
            good_fit_dir.mkdir()
            no_good_fit_dir.mkdir()
            prompt_file.write_text("Return json.", encoding="utf-8")
            env_file.write_text(
                "DEEPSEEK_API_KEY=test-key\nDEEPSEEK_MODEL=deepseek-v4-pro\n",
                encoding="utf-8",
            )
            (source_dir / "job.json").write_text(
                '{"title":"A","company":"Acme","location":"Remote","description":"Desc A"}',
                encoding="utf-8",
            )

            classifier = FakeClassifier(FitDecision("good_fit", 93, "Worth reviewing"))
            with temporary_cwd(root):
                exit_code = main(
                    [
                        "--source-dir",
                        str(source_dir),
                        "--good-fit-dir",
                        str(good_fit_dir),
                        "--no-good-fit-dir",
                        str(no_good_fit_dir),
                        "--prompt-file",
                        str(prompt_file),
                        "--env-file",
                        str(env_file),
                    ],
                    client=classifier,
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(classifier.calls[0][2], "deepseek-v4-pro")

    def test_main_moves_opened_or_applied_duplicates_without_calling_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "01-source-jobs"
            good_fit_dir = root / "02-good-fit"
            no_good_fit_dir = root / "03-no-good-fit"
            opened_or_applied_dir = root / "04-opened-or-applied"
            duplicates_dir = root / "05-duplicates"
            prompt_file = root / "prompt.txt"
            env_file = root / "deepseek.env"

            source_dir.mkdir()
            good_fit_dir.mkdir()
            no_good_fit_dir.mkdir()
            opened_or_applied_dir.mkdir()
            prompt_file.write_text("Return json.", encoding="utf-8")
            env_file.write_text("DEEPSEEK_API_KEY=test-key\n", encoding="utf-8")
            (opened_or_applied_dir / "Acme_Product-Manager_111.json").write_text(
                "{}",
                encoding="utf-8",
            )
            duplicate_job = source_dir / "Acme_Product-Manager_222.json"
            duplicate_job.write_text(
                '{"title":"Product Manager","company":"Acme","location":"Remote","description":"Desc"}',
                encoding="utf-8",
            )

            with temporary_cwd(root):
                exit_code = main(
                    [
                        "--source-dir",
                        str(source_dir),
                        "--good-fit-dir",
                        str(good_fit_dir),
                        "--no-good-fit-dir",
                        str(no_good_fit_dir),
                        "--prompt-file",
                        str(prompt_file),
                        "--env-file",
                        str(env_file),
                    ],
                    client=RaisingClassifier(AssertionError("LLM should not be called")),
                )

            self.assertEqual(exit_code, 0)
            self.assertFalse(duplicate_job.exists())
            self.assertTrue((duplicates_dir / "Acme_Product-Manager_222.json").exists())
            self.assertTrue((opened_or_applied_dir / "Acme_Product-Manager_111.json").exists())

    def test_main_moves_no_good_fit_duplicates_without_calling_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "01-source-jobs"
            good_fit_dir = root / "02-good-fit"
            no_good_fit_dir = root / "03-no-good-fit"
            duplicates_dir = root / "05-duplicates"
            prompt_file = root / "prompt.txt"
            env_file = root / "deepseek.env"

            source_dir.mkdir()
            good_fit_dir.mkdir()
            no_good_fit_dir.mkdir()
            prompt_file.write_text("Return json.", encoding="utf-8")
            env_file.write_text("DEEPSEEK_API_KEY=test-key\n", encoding="utf-8")
            (no_good_fit_dir / "Acme_Product-Manager_111.json").write_text(
                "{}",
                encoding="utf-8",
            )
            duplicate_job = source_dir / "Acme_Product-Manager_222.json"
            duplicate_job.write_text(
                '{"title":"Product Manager","company":"Acme","location":"Remote","description":"Desc"}',
                encoding="utf-8",
            )

            with temporary_cwd(root):
                exit_code = main(
                    [
                        "--source-dir",
                        str(source_dir),
                        "--good-fit-dir",
                        str(good_fit_dir),
                        "--no-good-fit-dir",
                        str(no_good_fit_dir),
                        "--prompt-file",
                        str(prompt_file),
                        "--env-file",
                        str(env_file),
                    ],
                    client=RaisingClassifier(AssertionError("LLM should not be called")),
                )

            self.assertEqual(exit_code, 0)
            self.assertFalse(duplicate_job.exists())
            self.assertTrue((duplicates_dir / "Acme_Product-Manager_222.json").exists())
            self.assertTrue((no_good_fit_dir / "Acme_Product-Manager_111.json").exists())

    def test_main_logs_duplicate_count_in_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "01-source-jobs"
            good_fit_dir = root / "02-good-fit"
            no_good_fit_dir = root / "03-no-good-fit"
            opened_or_applied_dir = root / "04-opened-or-applied"
            prompt_file = root / "prompt.txt"
            env_file = root / "deepseek.env"
            log_file = root / "classifier.log"

            source_dir.mkdir()
            good_fit_dir.mkdir()
            no_good_fit_dir.mkdir()
            opened_or_applied_dir.mkdir()
            prompt_file.write_text("Return json.", encoding="utf-8")
            env_file.write_text("DEEPSEEK_API_KEY=test-key\n", encoding="utf-8")
            (opened_or_applied_dir / "Acme_Product-Manager_111.json").write_text(
                "{}",
                encoding="utf-8",
            )
            (source_dir / "Acme_Product-Manager_222.json").write_text(
                '{"title":"Product Manager","company":"Acme","location":"Remote","description":"Desc"}',
                encoding="utf-8",
            )

            with temporary_cwd(root):
                exit_code = main(
                    [
                        "--source-dir",
                        str(source_dir),
                        "--good-fit-dir",
                        str(good_fit_dir),
                        "--no-good-fit-dir",
                        str(no_good_fit_dir),
                        "--opened-or-applied-dir",
                        str(opened_or_applied_dir),
                        "--prompt-file",
                        str(prompt_file),
                        "--env-file",
                        str(env_file),
                        "--log-file",
                        str(log_file),
                    ],
                    client=RaisingClassifier(AssertionError("LLM should not be called")),
                )

            self.assertEqual(exit_code, 0)
            log_text = log_file.read_text(encoding="utf-8")
            self.assertIn("Processed=1 good_fit=0 no_good_fit=0 duplicates=1 errors=0", log_text)

    def test_main_creates_default_persistent_log_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "01-source-jobs"
            good_fit_dir = root / "02-good-fit"
            no_good_fit_dir = root / "03-no-good-fit"
            prompt_file = root / "prompt.txt"
            env_file = root / "deepseek.env"

            source_dir.mkdir()
            good_fit_dir.mkdir()
            no_good_fit_dir.mkdir()
            prompt_file.write_text("Return json.", encoding="utf-8")
            env_file.write_text("DEEPSEEK_API_KEY=test-key\n", encoding="utf-8")
            (source_dir / "job.json").write_text(
                '{"title":"A","company":"Acme","location":"Remote","description":"Desc A"}',
                encoding="utf-8",
            )

            with temporary_cwd(root):
                exit_code = main(
                    [
                        "--source-dir",
                        str(source_dir),
                        "--good-fit-dir",
                        str(good_fit_dir),
                        "--no-good-fit-dir",
                        str(no_good_fit_dir),
                        "--prompt-file",
                        str(prompt_file),
                        "--env-file",
                        str(env_file),
                    ],
                    client=FakeClassifier(FitDecision("good_fit", 93, "Worth reviewing")),
                )

            self.assertEqual(exit_code, 0)
            log_dir = root / "logs" / "job-classifier"
            log_files = list(log_dir.glob("*.log"))
            self.assertEqual(len(log_files), 1)
            log_text = log_files[0].read_text(encoding="utf-8")
            self.assertIn("Processed=1 good_fit=1 no_good_fit=0 duplicates=0 errors=0", log_text)

    def test_collect_job_urls_includes_apply_and_all_hiring_team_links(self) -> None:
        urls = collect_job_urls(
            {
                "applyUrl": "https://example.com/apply",
                "hiringTeam": [
                    {"linkedinUrl": "https://linkedin.com/in/first"},
                    {"linkedinUrl": "https://linkedin.com/in/second"},
                ],
            }
        )

        self.assertEqual(
            urls,
            [
                "https://example.com/apply",
                "https://linkedin.com/in/first",
                "https://linkedin.com/in/second",
            ],
        )

    def test_open_job_file_opens_urls_and_moves_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "02-good-fit"
            destination_dir = root / "04-opened-or-applied"
            source_dir.mkdir()

            job_path = source_dir / "job.json"
            job_path.write_text(
                json.dumps(
                    {
                        "applyUrl": "https://example.com/apply",
                        "hiringTeam": [
                            {"linkedinUrl": "https://linkedin.com/in/first"},
                            {"linkedinUrl": "https://linkedin.com/in/second"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            opened: list[str] = []
            destination = open_job_file(
                job_path,
                destination_dir,
                opener=lambda url: opened.append(url) or True,
            )

            self.assertEqual(
                opened,
                [
                    "https://example.com/apply",
                    "https://linkedin.com/in/first",
                    "https://linkedin.com/in/second",
                ],
            )
            self.assertEqual(destination, destination_dir / "job.json")
            self.assertFalse(job_path.exists())
            self.assertTrue(destination.exists())

    def test_open_jobs_main_respects_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "02-good-fit"
            destination_dir = root / "04-opened-or-applied"
            source_dir.mkdir()

            (source_dir / "b.json").write_text(
                json.dumps({"applyUrl": "https://example.com/b"}),
                encoding="utf-8",
            )
            (source_dir / "a.json").write_text(
                json.dumps({"applyUrl": "https://example.com/a"}),
                encoding="utf-8",
            )

            opened: list[str] = []
            with temporary_cwd(root):
                exit_code = open_jobs_main(
                    [
                        "--limit",
                        "1",
                        "--source-dir",
                        str(source_dir),
                        "--destination-dir",
                        str(destination_dir),
                    ],
                    opener=lambda url: opened.append(url) or True,
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(opened, ["https://example.com/a"])
            self.assertTrue((destination_dir / "a.json").exists())
            self.assertTrue((source_dir / "b.json").exists())

    def test_main_uses_explicit_log_file_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "01-source-jobs"
            good_fit_dir = root / "02-good-fit"
            no_good_fit_dir = root / "03-no-good-fit"
            prompt_file = root / "prompt.txt"
            env_file = root / "deepseek.env"
            explicit_log_file = root / "custom-logs" / "manual.log"

            source_dir.mkdir()
            good_fit_dir.mkdir()
            no_good_fit_dir.mkdir()
            prompt_file.write_text("Return json.", encoding="utf-8")
            env_file.write_text("DEEPSEEK_API_KEY=test-key\n", encoding="utf-8")
            (source_dir / "job.json").write_text(
                '{"title":"A","company":"Acme","location":"Remote","description":"Desc A"}',
                encoding="utf-8",
            )

            with temporary_cwd(root):
                exit_code = main(
                    [
                        "--source-dir",
                        str(source_dir),
                        "--good-fit-dir",
                        str(good_fit_dir),
                        "--no-good-fit-dir",
                        str(no_good_fit_dir),
                        "--prompt-file",
                        str(prompt_file),
                        "--env-file",
                        str(env_file),
                        "--log-file",
                        str(explicit_log_file),
                    ],
                    client=FakeClassifier(FitDecision("good_fit", 93, "Worth reviewing")),
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(explicit_log_file.exists())
            log_text = explicit_log_file.read_text(encoding="utf-8")
            self.assertIn("Processed=1 good_fit=1 no_good_fit=0 duplicates=0 errors=0", log_text)

    def test_deepseek_client_logs_request_and_raw_response_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_file = root / "logs" / "classifier.log"
            configure_logging("INFO", log_file=log_file)
            response_body = (
                '{"choices":[{"message":{"content":"{\\"decision\\":\\"good_fit\\",'
                '\\"score\\":88,\\"reason\\":\\"Strong match\\"}"}}]}'
            )

            class FakeHTTPResponse:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return None

                def read(self) -> bytes:
                    return response_body.encode("utf-8")

            with mock.patch.object(request, "urlopen", return_value=FakeHTTPResponse()):
                client = DeepSeekClient(api_key="test-key")
                result = client.classify(
                    system_prompt="Return strict JSON.",
                    user_message="Title: Product Manager",
                    model="deepseek-v4-flash",
                )

            self.assertEqual(result.decision, "good_fit")
            log_text = log_file.read_text(encoding="utf-8")
            self.assertIn("LLM request payload:", log_text)
            self.assertIn('"content": "Return strict JSON."', log_text)
            self.assertIn('"content": "Title: Product Manager"', log_text)
            self.assertIn("LLM raw response:", log_text)
            self.assertIn('\\"decision\\":\\"good_fit\\"', log_text)

    def test_deepseek_client_uses_800_max_tokens(self) -> None:
        captured_request: request.Request | None = None
        response_body = (
            '{"choices":[{"message":{"content":"{\\"decision\\":\\"good_fit\\",'
            '\\"score\\":88,\\"reason\\":\\"Strong match\\"}"}}]}'
        )

        class FakeHTTPResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def read(self) -> bytes:
                return response_body.encode("utf-8")

        def fake_urlopen(http_request: request.Request, timeout: int):
            nonlocal captured_request
            captured_request = http_request
            return FakeHTTPResponse()

        with mock.patch.object(request, "urlopen", side_effect=fake_urlopen):
            client = DeepSeekClient(api_key="test-key")
            result = client.classify(
                system_prompt="Return strict JSON.",
                user_message="Title: Product Manager",
                model="deepseek-v4-flash",
            )

        self.assertEqual(result.decision, "good_fit")
        self.assertIsNotNone(captured_request)
        request_payload = json.loads(captured_request.data.decode("utf-8"))
        self.assertEqual(request_payload["max_tokens"], 800)

    def test_run_batch_writes_failure_artifact_for_model_response_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "01-source-jobs"
            good_fit_dir = root / "02-good-fit"
            no_good_fit_dir = root / "03-no-good-fit"
            error_artifact_dir = root / ".job-classifier-errors"
            source_dir.mkdir()
            raw_response = '{"decision":"good_fit","score":88,"reason":"Missing quote}'
            (source_dir / "job.json").write_text(
                '{"title":"Product Manager","company":"Acme","location":"Remote","description":"Own roadmap"}',
                encoding="utf-8",
            )

            logger = logging.getLogger("jobs_pipeline.tests")

            with self.assertLogs(logger, level="ERROR") as captured:
                result = run_batch(
                    source_dir=source_dir,
                    good_fit_dir=good_fit_dir,
                    no_good_fit_dir=no_good_fit_dir,
                    system_prompt="Return json.",
                    client=RaisingClassifier(
                        ModelResponseError(
                            "Malformed model response JSON",
                            raw_response=raw_response,
                        )
                    ),
                    model="deepseek-v4-flash",
                    error_artifact_dir=error_artifact_dir,
                    logger=logger,
                )

            self.assertEqual(result, (0, 0, 0, 0, 1))
            artifacts = list(error_artifact_dir.glob("*.json"))
            self.assertEqual(len(artifacts), 1)
            artifact = json.loads(artifacts[0].read_text(encoding="utf-8"))
            self.assertEqual(artifact["jobFile"], "job.json")
            self.assertEqual(artifact["model"], "deepseek-v4-flash")
            self.assertEqual(artifact["errorType"], "ModelResponseError")
            self.assertEqual(artifact["rawResponse"], raw_response)
            self.assertEqual(artifact["systemPrompt"], "Return json.")
            self.assertIn("Title: Product Manager", artifact["userMessage"])
            self.assertIn("Saved failure artifact", "\n".join(captured.output))

    def test_script_wrapper_can_render_help(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        result = subprocess.run(
            [sys.executable, "scripts/classify_jobs.py", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Classify scraped jobs with DeepSeek.", result.stdout)


if __name__ == "__main__":
    unittest.main()
