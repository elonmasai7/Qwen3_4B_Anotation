"""Confidence calibration logic."""

from __future__ import annotations

import math


class ConfidenceEngine:
    def calibrate(self, result: dict, verification: dict) -> float:
        base = float(result.get("confidence", 0.5) or 0.5)
        evidence = result.get("evidence") or []
        evidence_bonus = min(len(evidence), 3) * 0.08
        issue_penalty = len(verification.get("issues", [])) * 0.12
        calibrated = max(0.01, min(0.99, base + evidence_bonus - issue_penalty))
        # Smooth at extremes
        return 1 / (1 + math.exp(-6 * (calibrated - 0.5)))
