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
CONFIDENCE_THRESHOLD=0.4
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

The evaluation report includes:

- Per-field accuracy tables
- Confusion matrices for `claim_status`, `issue_type`, and `severity`
- Per-class accuracy breakdowns
- Per-object-type (car/laptop/package) metric splits
- Exact-match rate across all metric fields
- Evidence requirement coverage statistics
- Risk flag precision, recall, and F1
- Strategy comparison (text-only baseline vs. full pipeline)
- Operational cost and runtime analysis

## Architecture

The agent has six stages:

1. **Claim Parser**: Extracts issue type, object part, severity, constraints, and adversarial text. Uses LLM structured extraction with rule-based fallback when the API is unavailable or extraction fails.
2. **Image Evidence Extractor**: Normalizes image formats and inspects each image independently using a vision-language model (VLM).
3. **Cross-Image Aggregator**: Aggregates all per-image observations before adjudication. Detects conflicting evidence, partial support, object/part consistency, and computes confidence statistics (max, avg, supporting).
4. **Evidence Requirements Matcher**: Loads `evidence_requirements.csv` and matches applicable requirements to the claim by object type, issue family, and part. Evaluates whether each requirement is satisfied by the image evidence.
5. **Confidence-Aware Adjudicator**: Compares image observations, cross-image aggregation, matched requirements, user history, and confidence statistics to determine `claim_status`, `evidence_standard_met`, risk flags, and justifications. Low-confidence evidence triggers `manual_review_required` flags.
6. **Schema Guardian**: Validates allowed values, column order, booleans, flags, and supporting image IDs against the output schema.

```
Claim Parsing (LLM + rules)
  → Per-Image Evidence Extraction (VLM)
    → Cross-Image Aggregation
      → Requirements Matching
        → Confidence-Aware Adjudication
          → Guardian Validation → CSV Output
```

Execution is sequential inside each claim and parallel only for independent image and claim work. No MCP delegation server is used.

## Key Features

### LLM Claim Parsing
- Structured JSON extraction from multilingual claim conversations
- Falls back to keyword-based rules if LLM is unavailable
- Detects adversarial injection patterns in both LLM and rule paths

### Cross-Image Reasoning
- Detects when multiple images conflict on issue type or visible object
- Identifies partial support (some images match, others don't)
- Tracks object and part consistency across all images in a claim

### Confidence-Aware Decisions
- Uses per-image confidence from VLM responses
- Computes max, average, and supporting-evidence confidence
- Configurable threshold (`CONFIDENCE_THRESHOLD`, default 0.4)
- Low confidence triggers `manual_review_required` flag
- Very low confidence can downgrade `supported` to `not_enough_information`

### Explicit Requirement Matching
- Loads all 12 requirements from `evidence_requirements.csv`
- Matches by claim object, issue type, and object part
- Each matched requirement is evaluated (met/unmet with reason)
- Requirement IDs appear in `evidence_standard_met_reason`
- Unmet requirements are cited in `claim_status_justification`

### Rich Explainability
- Justifications reference specific image IDs, confidence scores, cross-image consistency, user history summaries, and matched requirement IDs
- All explanations are capped at 450 characters for output compatibility

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for OpenAI mode |
| `MODEL_NAME` | `gpt-4.1-mini` | Vision-language model to use |
| `EVIDENCE_AGENT_MODE` | `openai` | `openai` or `heuristic` |
| `MAX_CONCURRENT_CLAIMS` | `2` | Parallel claim processing threads |
| `MAX_CONCURRENT_IMAGES` | `4` | Parallel image processing threads per claim |
| `TEMPERATURE` | `0` | Model temperature (0 for deterministic) |
| `MAX_MODEL_RETRIES` | `2` | Retry attempts for failed API calls |
| `CONFIDENCE_THRESHOLD` | `0.4` | Below this, adds `manual_review_required` |

## Image Handling

The dataset includes files whose `.jpg` extension does not match their actual format. The normalizer detects signatures for JPEG, PNG, WebP, and AVIF. Unsupported or disguised files are converted into provider-safe cached images under `.cache/evidence_agent/`.

## Caching

Model responses are cached by prompt version, model name, row payload, and image metadata under `.cache/evidence_agent/model_responses/`. Both vision and text-only calls are cached independently.
