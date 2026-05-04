import tempfile
import unittest
from pathlib import Path

from jobs_pipeline import discover_job_files, parse_llm_response


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


if __name__ == "__main__":
    unittest.main()
