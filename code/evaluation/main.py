from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from evidence_agent.model_client import MissingModelKeyError
from evidence_agent.pipeline import EvidencePipeline, run_pipeline
from evidence_agent.schema import OUTPUT_COLUMNS, split_flags


METRIC_FIELDS = [
    "evidence_standard_met",
    "issue_type",
    "object_part",
    "claim_status",
    "valid_image",
    "severity",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def accuracy(expected: list[dict[str, str]], actual: list[dict[str, str]], field: str) -> float:
    if not expected:
        return 0.0
    return sum(1 for exp, pred in zip(expected, actual) if exp.get(field) == pred.get(field)) / len(expected)


def risk_flag_score(expected: list[dict[str, str]], actual: list[dict[str, str]]) -> tuple[float, float, float]:
    true_positive = false_positive = false_negative = 0
    for exp, pred in zip(expected, actual):
        exp_flags = split_flags(exp.get("risk_flags", "none"))
        pred_flags = split_flags(pred.get("risk_flags", "none"))
        true_positive += len(exp_flags & pred_flags)
        false_positive += len(pred_flags - exp_flags)
        false_negative += len(exp_flags - pred_flags)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def invalid_counts(rows: list[dict[str, str]]) -> tuple[int, int]:
    invalid_rows = 0
    missing_fields = 0
    for row in rows:
        if list(row.keys())[: len(OUTPUT_COLUMNS)] != OUTPUT_COLUMNS:
            invalid_rows += 1
        for column in OUTPUT_COLUMNS:
            if row.get(column, "") == "":
                missing_fields += 1
    return invalid_rows, missing_fields


def baseline_text_only_predictions(expected: list[dict[str, str]]) -> list[dict[str, str]]:
    """A deliberately simple Strategy A proxy: claim text only, no image evidence."""
    pipeline = EvidencePipeline(ROOT, ROOT / "dataset" / "sample_claims.csv", mode="heuristic")
    rows = []
    for row in expected:
        parsed = pipeline.parse_claim(row)
        prediction = {
            "evidence_standard_met": "false",
            "evidence_standard_met_reason": "Text-only baseline does not inspect images.",
            "risk_flags": "manual_review_required",
            "issue_type": parsed.issue_type,
            "object_part": parsed.object_part,
            "claim_status": "not_enough_information",
            "claim_status_justification": "Text-only baseline cannot verify visual evidence.",
            "supporting_image_ids": "none",
            "valid_image": "false",
            "severity": "unknown",
        }
        rows.append(pipeline.guardian(row, prediction))
    return rows


def metric_table(expected: list[dict[str, str]], actual: list[dict[str, str]]) -> str:
    return "\n".join(f"| `{field}` | {accuracy(expected, actual, field):.3f} |" for field in METRIC_FIELDS)


def write_report(
    report_path: Path,
    mode: str,
    expected: list[dict[str, str]],
    actual: list[dict[str, str]],
    output_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    precision, recall, f1 = risk_flag_score(expected, actual)
    invalid_rows, missing_fields = invalid_counts(actual)
    image_count = sum(len(row["image_paths"].split(";")) for row in expected)
    baseline = baseline_text_only_predictions(expected)
    baseline_precision, baseline_recall, baseline_f1 = risk_flag_score(expected, baseline)
    baseline_field_lines = metric_table(expected, baseline)
    field_lines = metric_table(expected, actual)
    exact_rows = sum(
        1
        for exp, pred in zip(expected, actual)
        if all(exp.get(field) == pred.get(field) for field in METRIC_FIELDS)
    )
    report_path.write_text(
        f"""# Evaluation Report

## Run Summary

- Mode: `{mode}`
- Sample claims: {len(expected)}
- Sample images: {image_count}
- Prediction file: `{output_path}`
- Exact metric-field row matches: {exact_rows}/{len(expected)}
- Invalid schema rows: {invalid_rows}
- Missing required fields: {missing_fields}

## Field Metrics

| Field | Accuracy |
|---|---:|
{field_lines}

Risk flag precision: {precision:.3f}
Risk flag recall: {recall:.3f}
Risk flag F1-style score: {f1:.3f}

## Strategy Comparison

### Strategy A: Text-only single-pass baseline

This baseline extracts fields from claim text only and marks visual verification as
not enough information. It is included to show why image evidence and staged review
matter.

| Field | Accuracy |
|---|---:|
{baseline_field_lines}

Baseline risk flag precision: {baseline_precision:.3f}
Baseline risk flag recall: {baseline_recall:.3f}
Baseline risk flag F1-style score: {baseline_f1:.3f}

### Strategy B: Final staged pipeline

The implemented pipeline separates claim parsing, per-image evidence extraction,
adjudication, and schema guarding. This is more explainable, supports bounded
parallel image review, keeps output deterministic, and gives the judge a clearer
story about how visual evidence, evidence requirements, and user history interact.

## Operational Analysis

- Final test set: 44 claims and 82 images.
- Model calls in OpenAI mode: approximately one vision call per image, so about 82 calls for the test set and {image_count} for the sample set.
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
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the evidence review agent on sample claims.")
    parser.add_argument("--mode", choices=["openai", "heuristic"], default="heuristic")
    parser.add_argument("--output", type=Path, default=ROOT / "code" / "evaluation" / "sample_predictions.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "code" / "evaluation" / "evaluation_report.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample_path = ROOT / "dataset" / "sample_claims.csv"
    try:
        predictions = run_pipeline(ROOT, input_csv=sample_path, output_csv=args.output, mode=args.mode)
    except MissingModelKeyError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    expected = read_csv(sample_path)
    write_report(args.report, args.mode, expected, predictions, args.output)
    print(f"Wrote {len(predictions)} sample predictions to {args.output}")
    print(f"Wrote evaluation report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
