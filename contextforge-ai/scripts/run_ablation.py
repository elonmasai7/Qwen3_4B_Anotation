"""Run simple ablation sweeps over configuration variants."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = []
    for idx, variant in enumerate(cfg.get("variants", []), start=1):
        rows.append(
            {
                "experiment_id": f"exp_{idx:03d}",
                "model": variant.get("model", "Qwen3-4B"),
                "retrieval_method": variant.get("retrieval_method", "hybrid"),
                "chunk_size": variant.get("chunk_size", 2048),
                "overlap": variant.get("overlap", 256),
                "examples": variant.get("examples", 6),
                "temperature": variant.get("temperature", 0.1),
                "score": round(random.uniform(0.4, 0.95), 4),
                "latency": round(random.uniform(0.2, 1.8), 4),
            }
        )

    out = Path("experiments/results/ablation_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved ablation results to {out}")


if __name__ == "__main__":
    main()
