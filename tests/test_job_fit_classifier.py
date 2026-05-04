import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jobs_pipeline import (
    FitDecision,
    discover_job_files,
    load_env_file,
    main,
    parse_llm_response,
    process_job_file,
)


class FakeClassifier:
    def __init__(self, decision: FitDecision) -> None:
        self.decision = decision
        self.calls: list[tuple[str, str, str]] = []

    def classify(self, system_prompt: str, user_message: str, model: str) -> FitDecision:
        self.calls.append((system_prompt, user_message, model))
        return self.decision


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
