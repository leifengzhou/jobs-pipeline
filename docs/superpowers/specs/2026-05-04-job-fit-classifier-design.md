# Job Fit Classifier Design

## Goal

Classify each scraped job in `01-source-jobs` as either a medium-confidence fit worth opening/applying or not a fit, using a DeepSeek V4 chat completion driven by an editable system prompt file. After classification, persist the decision into the job JSON and move the file into the next stage of the filesystem pipeline:

- `02-good-fit` for jobs the model considers worth reviewing
- `03-no-good-fit` for jobs the model rejects

## Current Context

The repository is a stage-oriented file pipeline with one JSON file per job. There is no existing application runtime or script scaffold in the repo yet. The input job JSON already contains the fields needed for an LLM-based decision, notably:

- `title`
- `company`
- `location`
- `salary`
- `description`
- `applyUrl`
- `linkedinUrl`
- `jobId`

The user wants:

- Python CLI implementation
- direct DeepSeek API usage
- credentials loaded from `~/.secrets/deepseek.env`
- editable prompt file rather than hardcoded profile text
- a `--limit N` option
- decision metadata stored back into each job JSON
- immediate routing of rejected jobs into `03-no-good-fit`

## Recommended Architecture

Implement a single Python CLI script that:

1. Loads configuration from environment variables and CLI flags.
2. Reads the system prompt from a tracked text file in the repo.
3. Iterates over job JSON files from `01-source-jobs`, optionally capped by `--limit`.
4. Calls DeepSeek with:
   - the system prompt
   - a structured user message containing selected job metadata and the full job description
5. Requires the model to return strict JSON with a binary decision and explanation.
6. Validates and normalizes the LLM response.
7. Writes decision metadata into the job JSON.
8. Moves the updated file into `02-good-fit` or `03-no-good-fit`.
9. Prints per-file progress and an end-of-run summary.

This keeps the pipeline simple, auditable, and easy to rerun without introducing a database or additional queueing layer.

## File Layout

Planned repo additions:

- `scripts/classify_jobs.py`
  - main CLI entrypoint
  - loads prompt/env
  - iterates jobs
  - performs classification
  - writes metadata
  - moves files
- `prompts/job_fit_system_prompt.txt`
  - editable fit rubric describing the user’s background, preferences, and decision rules
- `README.md`
  - minimal setup and usage instructions
- optionally `requirements.txt`
  - only if external dependencies are used

## Data Flow

### Input Selection

The CLI scans `01-source-jobs` for `*.json` files. Files are processed in deterministic order so `--limit N` produces predictable batches.

### Prompt Construction

The request to DeepSeek has two parts:

- system prompt
  - loaded from `prompts/job_fit_system_prompt.txt`
  - contains the user’s fit criteria and output contract
- user message
  - generated from the job JSON
  - includes concise job metadata followed by the full description

The prompt contract should explicitly require a JSON object with:

- `decision`: `good_fit` or `no_good_fit`
- `score`: numeric fit score
- `reason`: short explanation focused on why the role is or is not worth reviewing

### Response Handling

The CLI parses the model output as JSON and validates:

- `decision` is one of the two allowed values
- `score` is numeric
- `reason` is non-empty text

If validation fails, the script should:

- leave the file in `01-source-jobs`
- report the failure in stdout/stderr
- continue to the next file

This avoids silently misrouting jobs on malformed model output.

### Persistence

Before moving a file, the script updates the job JSON with:

- `fitDecision`
- `fitScore`
- `fitReason`
- `fitModel`
- `fitEvaluatedAt`

Optional future fields such as raw prompt/response dumps should be avoided in v1 to keep job files readable.

### Routing

After a successful update:

- `good_fit` jobs move to `02-good-fit`
- `no_good_fit` jobs move to `03-no-good-fit`

Moves should be atomic where possible, using standard filesystem rename/move behavior within the repo.

## Configuration

### Secrets

The script should load credentials from `~/.secrets/deepseek.env`. The exact variable names should be documented in the README and read by the script at runtime. The design assumes a direct DeepSeek API key is present there.

### CLI Flags

Initial CLI surface:

- `--limit N`
  - optional
  - process only the first `N` jobs in deterministic order
- `--source-dir`
  - optional override, default `01-source-jobs`
- `--good-fit-dir`
  - optional override, default `02-good-fit`
- `--no-good-fit-dir`
  - optional override, default `03-no-good-fit`
- `--prompt-file`
  - optional override, default `prompts/job_fit_system_prompt.txt`

Only `--limit` is user-required, but the directory/prompt overrides make the tool easier to test without hardcoding paths everywhere.

## Error Handling

The script should be resilient at the file level rather than failing the entire batch on one bad job. Expected failure cases:

- invalid job JSON
- missing required fields such as `description`
- missing prompt file
- missing or unreadable secrets file
- HTTP/API errors from DeepSeek
- malformed model response
- filesystem move/write errors

Expected handling:

- configuration errors that prevent any useful work should fail fast before processing starts
- per-job issues should be logged and skipped so the batch can continue

## Testing Strategy

Testing should focus on deterministic logic and avoid live DeepSeek calls.

Core test areas:

- job file discovery and deterministic limiting
- prompt/user payload generation
- response parsing and validation
- metadata write-back
- route selection and move behavior
- error handling when the model response is malformed

The API call layer should be isolated so tests can stub the classifier response cleanly.

## Out of Scope For V1

These are intentionally excluded to keep the first version small:

- third classification bucket such as `maybe`
- retry queues or dead-letter folders
- concurrency or parallel API calls
- rate-limit backoff sophistication beyond basic handling
- storing full raw LLM transcripts
- browser UI or dashboard
- reopening/reclassifying files already moved out of `01-source-jobs`

## Design Rationale

This design matches the existing filesystem pipeline instead of introducing new infrastructure. The prompt file gives the user direct control over fit logic without requiring code changes. The binary decision plus persisted explanation supports the intended workflow: run a batch, inspect only `02-good-fit`, and move quickly into application review.

The main risk is output reliability from the LLM. That risk is reduced by enforcing a narrow JSON schema and refusing to move files when the response cannot be validated.
