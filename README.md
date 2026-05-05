# Jobs Pipeline

This repo classifies scraped job JSON files from `01-source-jobs` into:

- `02-good-fit`
- `03-no-good-fit`

The classifier uses DeepSeek chat completions, writes the fit decision back into each JSON file, and then moves the file into the next stage directory.

After classification, a second CLI can open the selected good-fit jobs from `02-good-fit`, open each job's `applyUrl`, open every populated hiring-team `linkedinUrl`, and then move the JSON file into `04-opened-or-applied`.

## Setup

1. Put your DeepSeek API key in `~/.secrets/deepseek.env`:

```env
DEEPSEEK_API_KEY=your-key-here
```

2. Edit [prompts/job_fit_system_prompt.txt](/home/ralph/projects/jobs-pipeline/prompts/job_fit_system_prompt.txt) with your own fit criteria.

## Usage

Run a limited batch:

```bash
python3 scripts/classify_jobs.py --limit 5
```

Run the full batch:

```bash
python3 scripts/classify_jobs.py
```

Open a limited batch of good-fit jobs in your browser:

```bash
python3 scripts/open_jobs.py --limit 10
```

Useful flags:

- `--limit N`: process only the first `N` jobs in deterministic order
- `--prompt-file PATH`: use a different system prompt
- `--env-file PATH`: use a different env file
- `--model MODEL`: override the default model (`deepseek-v4-flash`)
- `--log-level LEVEL`: set the console/file log level (`INFO` by default)
- `--log-file PATH`: override the default per-run log file location
- `--error-artifact-dir PATH`: write per-failure diagnostic JSON artifacts (defaults to `.job-classifier-errors`)

Useful opener flags:

- `--limit N`: open only the first `N` good-fit jobs in deterministic order
- `--source-dir PATH`: override the input directory (defaults to `02-good-fit`)
- `--destination-dir PATH`: override the post-open directory (defaults to `04-opened-or-applied`)

## Diagnostics

The classifier now writes persistent logs by default to `logs/job-classifier/`, with one timestamped `.log` file per run. It also continues streaming logs to the console.

Each run log includes the LLM message trail for live debugging:

- the exact JSON request payload sent to DeepSeek
- the raw HTTP response body returned by DeepSeek
- the extracted model message content used for fit parsing

On model-response failures such as empty content or malformed JSON, it also writes a diagnostic artifact to `.job-classifier-errors/` containing:

- job filename and path
- model name
- error type and message
- system prompt
- rendered user message sent for classification
- request payload when available
- raw model response text when available
- parsed API payload when available

## Output Fields

Each processed job JSON is updated with:

- `fitDecision`
- `fitScore`
- `fitReason`
- `fitModel`
- `fitEvaluatedAt`
