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

The agent has six stages (see `code/evidence_agent/pipeline.py:EvidencePipeline.process_row`).

1. **Claim Parser (`parse_claim`)**: Creates structured fields (`issue_type`, `object_part`, `severity_hint`, `constraints`, `adversarial_text`) from `user_claim`.
   - Uses a text-only structured JSON call when `EVIDENCE_AGENT_MODE` is not `heuristic`.
   - Falls back to keyword/rule extraction (`rule_parse_claim`) if the model is unavailable or returns unknowns.
   - Independently flags prompt-injection style phrases via `INJECTION_PATTERNS`.
2. **Image Evidence Extractor (`extract_image_evidence`)**: For each `image_paths` entry, normalizes/converts the image into a provider-safe cached format and inspects that single image.
   - If `heuristic` mode (or normalization fails), produces a conservative schema-valid observation without pixel understanding.
   - Otherwise, uses a vision structured JSON call to extract visible object/parts, issue type/part, severity, `risk_flags`, an image-grounded description, and a numeric confidence.
3. **Cross-Image Aggregation (`aggregate_evidence`)**: Combines all per-image observations to compute:
   - consistency signals (object/part consistency)
   - conflict signals
   - confidence statistics (`max_confidence`, `avg_confidence`, and “supporting-evidence” confidence)
4. **Evidence Requirements Matching (`match_requirements`)**: Loads `dataset/evidence_requirements.csv` and selects which requirements apply to the claim.
   - Requirements are filtered by `claim_object` and mapped applicability by `issue_type` and/or `object_part`.
   - Each matched requirement is evaluated as met/unmet with a short reason.
5. **Confidence-Aware Adjudication (`adjudicate`)**: Produces final decision fields using:
   - user history risk flags (`dataset/user_history.csv`)
   - cross-image aggregation (conflicts + confidence)
   - per-image risk flags
   - requirement-matching results
   - confidence thresholding (`CONFIDENCE_THRESHOLD`, default `0.4`)
   - Final `claim_status` logic:
     - `not_enough_information` if evidence standard is not met
     - `supported` if evidence is met and supporting images exist (with a possible downgrade for very low supporting confidence)
     - `contradicted` when evidence exists but mismatches are detected (e.g., wrong part/object)
6. **Schema Guardian (`guardian`)**: Coerces/validates output fields against allowed enums, cleans justification text, filters `supporting_image_ids` to those actually present, and enforces `supporting_image_ids=none` for `not_enough_information`.

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

| Variable                | Default        | Description                                 |
| ----------------------- | -------------- | ------------------------------------------- |
| `OPENAI_API_KEY`        | —              | Required for OpenAI mode                    |
| `GOOGLE_API_KEY`        | —              | Required for GEMINI mode                    |
| `MODEL_NAME`            | `gpt-4.1-mini` | Vision-language model to use                |
| `EVIDENCE_AGENT_MODE`   | `openai`       | `openai` or `heuristic`                     |
| `MAX_CONCURRENT_CLAIMS` | `2`            | Parallel claim processing threads           |
| `MAX_CONCURRENT_IMAGES` | `4`            | Parallel image processing threads per claim |
| `TEMPERATURE`           | `0`            | Model temperature (0 for deterministic)     |
| `MAX_MODEL_RETRIES`     | `2`            | Retry attempts for failed API calls         |
| `CONFIDENCE_THRESHOLD`  | `0.4`          | Below this, adds `manual_review_required`   |

## Image Handling

The dataset includes files whose `.jpg` extension does not match their actual format. The normalizer detects signatures for JPEG, PNG, WebP, and AVIF. Unsupported or disguised files are converted into provider-safe cached images under `.cache/evidence_agent/`.

## Caching

Model responses are cached by prompt version, model name, row payload, and image metadata under `.cache/evidence_agent/model_responses/`. Both vision and text-only calls are cached independently.
