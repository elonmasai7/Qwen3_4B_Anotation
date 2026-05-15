"""Verifier that validates output quality and schema compliance."""

from __future__ import annotations


class VerifierAgent:
    def verify(self, result: dict, label_options: list[str]) -> dict:
        issues: list[str] = []
        if not isinstance(result, dict):
            issues.append("invalid_json")
            return {"valid": False, "issues": issues}

        label = result.get("label")
        if not label:
            issues.append("missing_label")
        if label_options and label not in label_options:
            issues.append("wrong_schema")
        if not result.get("evidence"):
            issues.append("missing_evidence")
        confidence = float(result.get("confidence", 0.0) or 0.0)
        if confidence < 0.2:
            issues.append("low_confidence")

        return {"valid": len(issues) == 0, "issues": issues}
