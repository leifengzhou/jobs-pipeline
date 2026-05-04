import tempfile
import unittest
from pathlib import Path

from jobs_pipeline import FitDecision, discover_job_files, parse_llm_response, process_job_file


class FakeClassifier:
    def __init__(self, decision: FitDecision) -> None:
        self.decision = decision

    def classify(self, system_prompt: str, user_message: str, model: str) -> FitDecision:
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


if __name__ == "__main__":
    unittest.main()
