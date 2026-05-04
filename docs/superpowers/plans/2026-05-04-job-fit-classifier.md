# Job Fit Classifier Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI that classifies each job JSON in `01-source-jobs` with DeepSeek, writes fit metadata into the file, and moves it to `02-good-fit` or `03-no-good-fit`.

**Architecture:** Use a small importable Python module for deterministic logic and a thin CLI wrapper under `scripts/`. Keep dependencies to the Python standard library, isolate the HTTP call behind a client class, and test all non-network behavior with `unittest` and temporary directories.

**Tech Stack:** Python 3.13, `argparse`, `json`, `pathlib`, `urllib`, `unittest`

---

## Chunk 1: Core Classification Module

### Task 1: Create the test scaffold and package boundary

**Files:**
- Create: `jobs_pipeline/__init__.py`
- Create: `tests/test_job_fit_classifier.py`
- Test: `tests/test_job_fit_classifier.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_discover_job_files_sorts_and_applies_limit(self):
    files = discover_job_files(self.source_dir, limit=2)
    self.assertEqual([path.name for path in files], ["a.json", "b.json"])

def test_parse_llm_response_requires_expected_schema(self):
    result = parse_llm_response('{"decision":"good_fit","score":88,"reason":"Strong match"}')
    self.assertEqual(result.decision, "good_fit")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_job_fit_classifier -v`
Expected: FAIL with import or missing symbol errors for the not-yet-created module/functions.

- [ ] **Step 3: Write minimal implementation**

Create the importable module with:

```python
@dataclass
class FitDecision:
    decision: str
    score: float
    reason: str

def discover_job_files(source_dir: Path, limit: int | None = None) -> list[Path]:
    ...

def parse_llm_response(text: str) -> FitDecision:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_job_fit_classifier -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add jobs_pipeline/__init__.py tests/test_job_fit_classifier.py
git commit -m "test: add core classifier parsing tests"
```

### Task 2: Add file update and routing behavior

**Files:**
- Modify: `tests/test_job_fit_classifier.py`
- Modify: `jobs_pipeline/__init__.py`
- Test: `tests/test_job_fit_classifier.py`

- [ ] **Step 1: Write the failing test**

```python
def test_process_job_file_updates_metadata_and_moves_good_fit(self):
    classifier = FakeClassifier(FitDecision("good_fit", 91, "Strong fit"))
    destination = process_job_file(...)
    self.assertEqual(destination.parent.name, "02-good-fit")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_job_fit_classifier -v`
Expected: FAIL because `process_job_file` does not exist or does not write/move correctly.

- [ ] **Step 3: Write minimal implementation**

Implement:

```python
def build_user_message(job: dict[str, object]) -> str:
    ...

def process_job_file(
    job_path: Path,
    system_prompt: str,
    client: ClassifierClient,
    model: str,
    good_fit_dir: Path,
    no_good_fit_dir: Path,
    now: Callable[[], datetime] | None = None,
) -> Path:
    ...
```

Behavior:
- read job JSON
- call classifier
- update `fitDecision`, `fitScore`, `fitReason`, `fitModel`, `fitEvaluatedAt`
- write JSON back
- move to the correct directory

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_job_fit_classifier -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add jobs_pipeline/__init__.py tests/test_job_fit_classifier.py
git commit -m "feat: add job routing and metadata persistence"
```

## Chunk 2: CLI, Prompt, and Documentation

### Task 3: Add env loading and CLI entrypoint

**Files:**
- Create: `scripts/classify_jobs.py`
- Modify: `jobs_pipeline/__init__.py`
- Modify: `tests/test_job_fit_classifier.py`
- Test: `tests/test_job_fit_classifier.py`

- [ ] **Step 1: Write the failing test**

```python
def test_load_env_file_reads_key_values(self):
    values = load_env_file(self.env_path)
    self.assertEqual(values["DEEPSEEK_API_KEY"], "test-key")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_job_fit_classifier -v`
Expected: FAIL because env loading/CLI wiring does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement:
- `load_env_file(path: Path) -> dict[str, str]`
- `DeepSeekClient` using `urllib.request`
- `main(argv: Sequence[str] | None = None) -> int`
- CLI flags:
  - `--limit`
  - `--source-dir`
  - `--good-fit-dir`
  - `--no-good-fit-dir`
  - `--prompt-file`
  - `--env-file`
  - `--model`

The script should default to:
- env file: `~/.secrets/deepseek.env`
- model: `deepseek-v4-flash`
- API base URL: `https://api.deepseek.com/chat/completions`

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_job_fit_classifier -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add jobs_pipeline/__init__.py scripts/classify_jobs.py tests/test_job_fit_classifier.py
git commit -m "feat: add DeepSeek-backed classification CLI"
```

### Task 4: Add prompt template and usage docs

**Files:**
- Create: `prompts/job_fit_system_prompt.txt`
- Create: `README.md`
- Test: `python3 -m unittest tests.test_job_fit_classifier -v`

- [ ] **Step 1: Write the failing test**

No new automated test required. Instead, define the verification target before editing:

Run: `python3 -m unittest tests.test_job_fit_classifier -v`
Expected before docs edits: PASS, establishing a clean baseline before non-code additions.

- [ ] **Step 2: Run verification baseline**

Run: `python3 -m unittest tests.test_job_fit_classifier -v`
Expected: PASS

- [ ] **Step 3: Write minimal implementation**

Add:
- editable prompt template that instructs the model to return strict JSON
- README setup/usage covering:
  - required secret file key `DEEPSEEK_API_KEY`
  - prompt editing
  - `--limit N`
  - full-batch behavior when `--limit` is omitted

- [ ] **Step 4: Run test to verify it still passes**

Run: `python3 -m unittest tests.test_job_fit_classifier -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add README.md prompts/job_fit_system_prompt.txt
git commit -m "docs: add job classifier prompt and usage guide"
```

## Chunk 3: Final Verification

### Task 5: End-to-end dry smoke check with a test fixture

**Files:**
- Modify: `tests/test_job_fit_classifier.py`
- Test: `tests/test_job_fit_classifier.py`

- [ ] **Step 1: Write the failing test**

```python
def test_cli_processes_a_single_file_with_limit(self):
    exit_code = main([...])
    self.assertEqual(exit_code, 0)
    self.assertTrue((self.good_fit_dir / "job.json").exists())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_job_fit_classifier -v`
Expected: FAIL due to missing CLI path or incomplete orchestration.

- [ ] **Step 3: Write minimal implementation**

Complete any missing wiring so the CLI can:
- load prompt text
- load env values
- discover jobs
- process a limited batch
- continue on per-file errors
- return non-zero only for startup/configuration failure

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_job_fit_classifier -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add jobs_pipeline/__init__.py scripts/classify_jobs.py tests/test_job_fit_classifier.py
git commit -m "test: cover CLI batch orchestration"
```

## Final Verification Commands

- [ ] Run: `python3 -m unittest tests.test_job_fit_classifier -v`
- [ ] Run: `python3 scripts/classify_jobs.py --help`
- [ ] Run: `git status --short`

Plan complete and saved to `docs/superpowers/plans/2026-05-04-job-fit-classifier.md`. Ready to execute.
