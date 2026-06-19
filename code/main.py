from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from evidence_agent.model_client import MissingModelKeyError
from evidence_agent.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-modal evidence review agent.")
    parser.add_argument("--input", type=Path, default=ROOT / "dataset" / "claims.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "output.csv")
    parser.add_argument(
        "--mode",
        choices=["openai", "heuristic"],
        default=None,
        help="Use openai for final VLM runs, heuristic for local smoke tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = run_pipeline(ROOT, input_csv=args.input, output_csv=args.output, mode=args.mode)
    except MissingModelKeyError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {len(rows)} prediction rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
