"""Run evaluation from CLI."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.evaluator import Evaluator


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gold", required=True)
    args = parser.parse_args()

    evaluator = Evaluator()
    start = time.time()
    pred_rows = _read_jsonl(args.pred)
    gold_rows = _read_jsonl(args.gold)
    report = evaluator.evaluate(pred_rows, gold_rows, latency_s=time.time() - start)
    evaluator.save_report(report, "data/reports/evaluation_report.json", "data/reports/evaluation_report.md")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
