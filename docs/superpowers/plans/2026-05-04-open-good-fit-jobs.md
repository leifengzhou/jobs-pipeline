# Open Good-Fit Jobs Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI that opens each selected good-fit job's `applyUrl`, opens every hiring-team LinkedIn URL when present, and then moves the processed JSON into `04-opened-or-applied`.

**Architecture:** Reuse the existing `jobs_pipeline` module for deterministic file discovery so `--limit` stays consistent with the classifier. Add a small opener workflow in the package, keep browser-opening behind an injectable function for testability, and expose it through a thin `scripts/open_jobs.py` wrapper.

**Tech Stack:** Python 3.13, `argparse`, `json`, `pathlib`, `webbrowser`, `unittest`

---

## Chunk 1: Core Opener Workflow

### Task 1: Add failing tests for URL selection and file moves

**Files:**
- Modify: `tests/test_job_fit_classifier.py`
- Test: `tests/test_job_fit_classifier.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_collect_job_urls_includes_apply_and_all_hiring_team_links(self):
    urls = collect_job_urls({...})
    self.assertEqual(urls, ["https://apply.example", "https://linkedin.example/a"])

def test_open_job_file_opens_urls_and_moves_file(self):
    opened = []
    destination = open_job_file(job_path, destination_dir, opener=opened.append)
    self.assertEqual(opened, [...])
    self.assertTrue((destination_dir / "job.json").exists())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_job_fit_classifier.JobFitClassifierTests.test_collect_job_urls_includes_apply_and_all_hiring_team_links tests.test_job_fit_classifier.JobFitClassifierTests.test_open_job_file_opens_urls_and_moves_file -v`
Expected: FAIL because the opener helpers do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement:
- `collect_job_urls(job: dict[str, object]) -> list[str]`
- `open_job_file(job_path: Path, destination_dir: Path, opener: Callable[[str], bool] | None = None) -> Path`

Behavior:
- open `applyUrl` first when present
- open every non-empty `hiringTeam[*].linkedinUrl`
- move the JSON file into `04-opened-or-applied` after the open loop completes

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_job_fit_classifier.JobFitClassifierTests.test_collect_job_urls_includes_apply_and_all_hiring_team_links tests.test_job_fit_classifier.JobFitClassifierTests.test_open_job_file_opens_urls_and_moves_file -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add jobs_pipeline/__init__.py tests/test_job_fit_classifier.py
git commit -m "feat: add good-fit job opener workflow"
```

### Task 2: Add batch and CLI limit coverage

**Files:**
- Create: `scripts/open_jobs.py`
- Modify: `jobs_pipeline/__init__.py`
- Modify: `tests/test_job_fit_classifier.py`
- Test: `tests/test_job_fit_classifier.py`

- [ ] **Step 1: Write the failing test**

```python
def test_open_jobs_main_respects_limit(self):
    exit_code = open_jobs_main(["--limit", "1", ...], opener=opened.append)
    self.assertEqual(exit_code, 0)
    self.assertEqual(opened, ["https://first.example"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_job_fit_classifier.JobFitClassifierTests.test_open_jobs_main_respects_limit -v`
Expected: FAIL because the opener batch CLI does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement:
- `run_open_batch(source_dir: Path, destination_dir: Path, limit: int | None = None, opener: Callable[[str], bool] | None = None) -> tuple[int, int]`
- `open_jobs_main(argv: Sequence[str] | None = None, opener: Callable[[str], bool] | None = None) -> int`
- `scripts/open_jobs.py` wrapper

CLI flags:
- `--limit`
- `--source-dir` defaulting to `02-good-fit`
- `--destination-dir` defaulting to `04-opened-or-applied`

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_job_fit_classifier.JobFitClassifierTests.test_open_jobs_main_respects_limit -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add jobs_pipeline/__init__.py scripts/open_jobs.py tests/test_job_fit_classifier.py
git commit -m "feat: add CLI for opening good-fit jobs"
```

## Chunk 2: Documentation and Verification

### Task 3: Document usage and rerun focused verification

**Files:**
- Modify: `README.md`
- Test: `tests/test_job_fit_classifier.py`

- [ ] **Step 1: Define verification target**

Run: `python3 -m unittest tests.test_job_fit_classifier -v`
Expected: PASS after code changes land.

- [ ] **Step 2: Update docs**

Document:
- `python3 scripts/open_jobs.py --limit 10`
- source/destination defaults
- behavior for hiring-team LinkedIn URLs

- [ ] **Step 3: Run verification**

Run: `python3 -m unittest tests.test_job_fit_classifier -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add README.md tests/test_job_fit_classifier.py jobs_pipeline/__init__.py scripts/open_jobs.py
git commit -m "docs: add good-fit opener usage"
```
