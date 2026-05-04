# Jobs Pipeline

This repo classifies scraped job JSON files from `01-source-jobs` into:

- `02-good-fit`
- `03-no-good-fit`

The classifier uses DeepSeek chat completions, writes the fit decision back into each JSON file, and then moves the file into the next stage directory.

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

Useful flags:

- `--limit N`: process only the first `N` jobs in deterministic order
- `--prompt-file PATH`: use a different system prompt
- `--env-file PATH`: use a different env file
- `--model MODEL`: override the default model (`deepseek-v4-flash`)

## Output Fields

Each processed job JSON is updated with:

- `fitDecision`
- `fitScore`
- `fitReason`
- `fitModel`
- `fitEvaluatedAt`
