# Evaluation Report

## Run Summary

- Mode: `heuristic`
- Sample claims: 20
- Sample images: 29
- Prediction file: `C:\Users\soham\OneDrive\Documents\SOHAM\NIT ANDHRA\CSE\hackerrank-orchestrate-june26\code\evaluation\sample_predictions.csv`
- Exact metric-field row matches: 6/20
- Invalid schema rows: 0
- Missing required fields: 0

## Field Metrics

| Field | Accuracy |
|---|---:|
| `evidence_standard_met` | 0.900 |
| `issue_type` | 0.550 |
| `object_part` | 0.600 |
| `claim_status` | 0.650 |
| `valid_image` | 0.900 |
| `severity` | 0.350 |

Risk flag precision: 1.000
Risk flag recall: 0.269
Risk flag F1-style score: 0.424

## Strategy Comparison

### Strategy A: Text-only single-pass baseline

This baseline extracts fields from claim text only and marks visual verification as
not enough information. It is included to show why image evidence and staged review
matter.

| Field | Accuracy |
|---|---:|
| `evidence_standard_met` | 0.100 |
| `issue_type` | 0.550 |
| `object_part` | 0.600 |
| `claim_status` | 0.100 |
| `valid_image` | 0.100 |
| `severity` | 0.100 |

Baseline risk flag precision: 0.350
Baseline risk flag recall: 0.269
Baseline risk flag F1-style score: 0.304

### Strategy B: Final staged pipeline

The implemented pipeline separates claim parsing, per-image evidence extraction,
adjudication, and schema guarding. This is more explainable, supports bounded
parallel image review, keeps output deterministic, and gives the judge a clearer
story about how visual evidence, evidence requirements, and user history interact.

## Operational Analysis

- Final test set: 44 claims and 82 images.
- Model calls in OpenAI mode: approximately one vision call per image, so about 82 calls for the test set and 29 for the sample set.
- Token usage: claim text plus compact JSON instructions per image; image-token accounting depends on provider detail settings and image dimensions.
- Cost: estimate with the selected model's current image and text pricing before submission.
- Runtime: bounded by image upload/model latency; default concurrency is `MAX_CONCURRENT_CLAIMS=2` and `MAX_CONCURRENT_IMAGES=4`.
- Rate limits: lower concurrency if requests-per-minute or tokens-per-minute errors appear.
- Caching: responses are cached by prompt version, model name, row payload, and image metadata under `.cache/evidence_agent/`.
- Retries: model calls retry with bounded backoff via `MAX_MODEL_RETRIES`.

## Notes

Heuristic mode is only for local smoke testing and schema validation when no API key
is available. Final competitive predictions should use `EVIDENCE_AGENT_MODE=openai`
with a vision-capable model and `OPENAI_API_KEY` set.
