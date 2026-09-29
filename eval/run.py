"""Run an evaluation tier and record the result -- Eval Design §2, §6.

    uv run python -m eval.run --tier deterministic          # free, no model, no network
    uv run python -m eval.run --tier live                   # costs model calls
    uv run python -m eval.run --tier deterministic --json results.json

The two tiers exist because they fail for different reasons and cost different money, which is also
why the CI gates are split the same way: the deterministic tier guards every push, and the live tier
runs on a schedule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval import corpora


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tier", choices=["deterministic", "live"], default="deterministic")
    parser.add_argument("--json", type=Path, help="also write the full result as JSON")
    parser.add_argument("--batches", type=int, default=0,
                        help="live tier only: cap the batches audited, to bound the spend")
    args = parser.parse_args()

    print(f"golden datasets: {corpora.summary()}")
    print(f"running the {args.tier} tier\n")

    if args.tier == "deterministic":
        from eval.runners import deterministic

        result = deterministic.run()
    else:
        from eval.runners import live

        result = live.run(batch_cap=args.batches or None)

    metrics = result.pop("_metrics")
    print()
    for metric in metrics:
        print(metric.line())
        if metric.advisory:
            print("         advisory only -- does not gate")

    print(f"\n{result['tier']}: {'PASS' if result['passed'] else 'FAIL'} in {result['seconds']}s")
    if result["failed"]:
        print(f"  failed: {', '.join(result['failed'])}")

    if args.json:
        args.json.write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(f"  -> {args.json}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
