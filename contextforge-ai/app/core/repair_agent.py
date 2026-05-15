"""Repairs weak annotations."""

from __future__ import annotations


class RepairAgent:
    def repair(self, result: dict, issues: list[str], label_options: list[str]) -> dict:
        fixed = dict(result)
        if "missing_label" in issues or "wrong_schema" in issues:
            fixed["label"] = label_options[0] if label_options else fixed.get("label", "unknown")
        if "missing_evidence" in issues:
            fixed["evidence"] = [fixed.get("rationale", "evidence unavailable")]
        if "low_confidence" in issues:
            fixed["confidence"] = max(float(fixed.get("confidence", 0.0) or 0.0), 0.35)
        fixed.setdefault("uncertainty_reason", "repaired_output")
        return fixed
