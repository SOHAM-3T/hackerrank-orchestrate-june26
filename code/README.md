# Multi-Modal Evidence Review Agent

This folder contains the runnable solution for HackerRank Orchestrate.

## Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r code\requirements.txt
```

Create a `.env` file or export environment variables:

```text
OPENAI_API_KEY=...
MODEL_NAME=gpt-4.1-mini
EVIDENCE_AGENT_MODE=openai
MAX_CONCURRENT_CLAIMS=2
MAX_CONCURRENT_IMAGES=4
TEMPERATURE=0
```

No secrets are stored in the code.

## Run Final Predictions

```powershell
.\.venv\Scripts\python.exe code\main.py
```

This reads `dataset/claims.csv` and writes root-level `output.csv`.

For local smoke tests without an API key:

```powershell
.\.venv\Scripts\python.exe code\main.py --mode heuristic
```

Heuristic mode is schema-valid but not intended for competitive final predictions.

## Evaluation

```powershell
.\.venv\Scripts\python.exe code\evaluation\main.py --mode heuristic
```

For model-backed evaluation:

```powershell
.\.venv\Scripts\python.exe code\evaluation\main.py --mode openai
```

The evaluator writes:

- `code/evaluation/sample_predictions.csv`
- `code/evaluation/evaluation_report.md`

## Architecture

The agent has four stages:

1. Claim Parser: extracts issue, part, severity hints, constraints, and adversarial text.
2. Image Evidence Extractor: normalizes image formats and inspects each image independently.
3. Evidence Adjudicator: compares image observations with the claim, user history, and evidence requirements.
4. Schema Guardian: validates allowed values, column order, booleans, flags, and supporting image IDs.

Execution is sequential inside each claim and parallel only for independent image and claim work. No MCP delegation server is used.

## Image Handling

The dataset includes files whose `.jpg` extension does not match their actual format. The normalizer detects signatures for JPEG, PNG, WebP, and AVIF. Unsupported or disguised files are converted into provider-safe cached images under `.cache/evidence_agent/`.

## Caching

Model responses are cached by prompt version, model name, row payload, and image metadata under `.cache/evidence_agent/model_responses/`.
