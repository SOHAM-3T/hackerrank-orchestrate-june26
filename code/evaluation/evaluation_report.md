# Evaluation Report

## Run Summary

- Mode: `heuristic`
- Sample claims: 20
- Sample images: 29
- Prediction file: `C:\Users\soham\OneDrive\Documents\SOHAM\NIT ANDHRA\CSE\hackerrank-orchestrate-june26\code\evaluation\sample_predictions.csv`
- Exact metric-field row matches: 0/20 (0.0%)
- Invalid schema rows: 0
- Missing required fields: 0

## Field Metrics

| Field | Accuracy |
|---|---:|
| `evidence_standard_met` | 0.900 |
| `issue_type` | 0.500 |
| `object_part` | 0.700 |
| `claim_status` | 0.100 |
| `valid_image` | 0.900 |
| `severity` | 0.100 |

Risk flag precision: 0.283
Risk flag recall: 0.500
Risk flag F1-style score: 0.361

## Confusion Matrices

### `claim_status`

| Expected \ Predicted | `contradicted` | `not_enough_information` | `supported` |
|---|---:|---:|---:|
| `contradicted` | 0 | 5 | 0 |
| `not_enough_information` | 0 | 2 | 0 |
| `supported` | 0 | 13 | 0 |

### `issue_type`

| Expected \ Predicted | `broken_part` | `crack` | `crushed_packaging` | `dent` | `missing_part` | `none` | `scratch` | `stain` | `torn_packaging` | `unknown` | `water_damage` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `broken_part` | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 |
| `crack` | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| `crushed_packaging` | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `dent` | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `missing_part` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `none` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| `scratch` | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 |
| `stain` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| `torn_packaging` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| `unknown` | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `water_damage` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |

### `severity`

| Expected \ Predicted | `high` | `low` | `medium` | `none` | `unknown` |
|---|---:|---:|---:|---:|---:|
| `high` | 0 | 0 | 0 | 0 | 1 |
| `low` | 0 | 0 | 0 | 0 | 4 |
| `medium` | 0 | 0 | 0 | 0 | 11 |
| `none` | 0 | 0 | 0 | 0 | 2 |
| `unknown` | 0 | 0 | 0 | 0 | 2 |

## Per-Class Accuracy

### `claim_status`

| Class | Correct | Total | Accuracy |
|---|---:|---:|---:|
| `contradicted` | 0 | 5 | 0.000 |
| `not_enough_information` | 2 | 2 | 1.000 |
| `supported` | 0 | 13 | 0.000 |

### `issue_type`

| Class | Correct | Total | Accuracy |
|---|---:|---:|---:|
| `broken_part` | 1 | 3 | 0.333 |
| `crack` | 2 | 3 | 0.667 |
| `crushed_packaging` | 1 | 1 | 1.000 |
| `dent` | 3 | 3 | 1.000 |
| `none` | 0 | 2 | 0.000 |
| `scratch` | 1 | 2 | 0.500 |
| `stain` | 1 | 1 | 1.000 |
| `torn_packaging` | 1 | 1 | 1.000 |
| `unknown` | 0 | 3 | 0.000 |
| `water_damage` | 0 | 1 | 0.000 |

### `severity`

| Class | Correct | Total | Accuracy |
|---|---:|---:|---:|
| `high` | 0 | 1 | 0.000 |
| `low` | 0 | 4 | 0.000 |
| `medium` | 0 | 11 | 0.000 |
| `none` | 0 | 2 | 0.000 |
| `unknown` | 2 | 2 | 1.000 |

## Per-Object-Type Breakdown

#### `car` (8 claims)

| Field | Accuracy |
|---|---:|
| `evidence_standard_met` | 0.875 |
| `issue_type` | 0.500 |
| `object_part` | 0.750 |
| `claim_status` | 0.125 |
| `valid_image` | 0.875 |
| `severity` | 0.125 |

#### `laptop` (6 claims)

| Field | Accuracy |
|---|---:|
| `evidence_standard_met` | 1.000 |
| `issue_type` | 0.667 |
| `object_part` | 0.667 |
| `claim_status` | 0.000 |
| `valid_image` | 1.000 |
| `severity` | 0.000 |

#### `package` (6 claims)

| Field | Accuracy |
|---|---:|
| `evidence_standard_met` | 0.833 |
| `issue_type` | 0.333 |
| `object_part` | 0.667 |
| `claim_status` | 0.167 |
| `valid_image` | 0.833 |
| `severity` | 0.167 |

## Exact Match & Requirement Coverage

- Exact match rate (all 6 metric fields correct): **0.0%**
- Predictions referencing evidence requirements: 20/20 (100.0%)

## Strategy Comparison

### Strategy A: Text-only single-pass baseline

This baseline extracts fields from claim text only and marks visual verification as
not enough information. It is included to show why image evidence and staged review
matter.

| Field | Accuracy |
|---|---:|
| `evidence_standard_met` | 0.100 |
| `issue_type` | 0.500 |
| `object_part` | 0.700 |
| `claim_status` | 0.100 |
| `valid_image` | 0.100 |
| `severity` | 0.100 |

Baseline risk flag precision: 0.350
Baseline risk flag recall: 0.269
Baseline risk flag F1-style score: 0.304

### Strategy B: Final staged pipeline

The implemented pipeline separates claim parsing (LLM + rule fallback), per-image
evidence extraction, cross-image aggregation, confidence-aware adjudication with
requirement matching, and schema guarding. This is more explainable, supports bounded
parallel image review, keeps output deterministic, and gives the judge a clearer
story about how visual evidence, evidence requirements, confidence, and user history
interact.

## Operational Analysis

- Final test set: 44 claims and 82 images.
- Model calls in OpenAI mode: approximately one vision call per image plus one text call per claim for LLM claim parsing, so about 82 + 44 = 126 calls for the test set and 29 + 20 = 49 for the sample set.
- Token usage: claim text plus compact JSON instructions per image; image-token accounting depends on provider detail settings and image dimensions. LLM claim parsing uses ~200 input tokens per call.
- Cost: estimate with the selected model's current image and text pricing before submission. Text-only claim parsing adds negligible cost (~$0.01 total).
- Runtime: bounded by image upload/model latency; default concurrency is `MAX_CONCURRENT_CLAIMS=2` and `MAX_CONCURRENT_IMAGES=4`.
- Rate limits: lower concurrency if requests-per-minute or tokens-per-minute errors appear.
- Caching: responses are cached by prompt version, model name, row payload, and image metadata under `.cache/evidence_agent/`.
- Retries: model calls retry with bounded backoff via `MAX_MODEL_RETRIES`.

## Notes

Heuristic mode is only for local smoke testing and schema validation when no API key
is available. Final competitive predictions should use `EVIDENCE_AGENT_MODE=openai`
with a vision-capable model and `OPENAI_API_KEY` set.
