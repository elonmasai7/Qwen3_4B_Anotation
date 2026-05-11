from typing import Any
import re
from common.types import (
    AnnotationResult, AnnotationLabel, AnnotationStatus, VerificationResult,
    DataRow,
)
from common.logging import get_logger

logger = get_logger(__name__)


class ConsistencyChecker:
    def __init__(self, ontology: dict[str, Any] | None = None):
        self.ontology = ontology or {}

    def verify(self, annotation: AnnotationResult, data: DataRow) -> VerificationResult:
        issues: list[str] = []
        conflicts: list[dict[str, Any]] = []

        if not annotation.labels:
            issues.append("No labels provided")
            return VerificationResult(is_valid=False, issues=issues)

        for i, label in enumerate(annotation.labels):
            label_issues = self._check_label_consistency(label, data, i)
            issues.extend(label_issues)

        conflict_results = self._detect_conflicts(annotation.labels)
        conflicts.extend(conflict_results)

        schema_valid = self._validate_schema(annotation.labels)
        if not schema_valid:
            issues.append("Schema validation failed")

        is_valid = len(issues) == 0 and len(conflicts) == 0

        return VerificationResult(
            is_valid=is_valid,
            issues=issues,
            conflicts=conflicts,
        )

    def _check_label_consistency(
        self,
        label: AnnotationLabel,
        data: DataRow,
        label_idx: int,
    ) -> list[str]:
        issues: list[str] = []

        if label.confidence < 0 or label.confidence > 1:
            issues.append(f"Label {label_idx}: Confidence out of range [0,1]")

        if not label.rationale:
            issues.append(f"Label {label_idx}: Missing rationale")

        if label.confidence > 0.9 and not label.evidence_spans:
            issues.append(f"Label {label_idx}: High confidence without evidence spans")

        if label.evidence_spans:
            for start, end in label.evidence_spans:
                if start < 0 or end > len(data.content):
                    issues.append(f"Label {label_idx}: Invalid evidence span [{start}, {end}]")

        return issues

    def _detect_conflicts(self, labels: list[AnnotationLabel]) -> list[dict[str, Any]]:
        conflicts: list[dict[str, Any]] = []

        if len(labels) < 2:
            return conflicts

        label_values = [str(label.value) for label in labels]
        if len(set(label_values)) > 1:
            conflicts.append({
                "type": "label_conflict",
                "labels": label_values,
                "description": "Multiple different labels provided",
            })

        for i, label in enumerate(labels):
            if label.alternative_hypotheses:
                for alt in label.alternative_hypotheses:
                    if str(label.value).lower() == alt.lower():
                        conflicts.append({
                            "type": "self_contradiction",
                            "label_index": i,
                            "description": f"Alternative '{alt}' contradicts main label",
                        })

        return conflicts

    def _validate_schema(self, labels: list[AnnotationLabel]) -> bool:
        required_fields = ["value", "confidence", "rationale"]
        for label in labels:
            for field in required_fields:
                if not hasattr(label, field):
                    return False
        return True


class AdversarialChecker:
    def __init__(self):
        self.injection_patterns = [
            r"ignore\s+(previous|above|all)\s+(instructions?|prompts?|rules?)",
            r"system\s*:\s*",
            r"you\s+are\s+(now|a)",
            r"forget\s+(everything|all|what)",
            r"\x00|\x01|\x02",
            r"<script|>",
        ]
        self.distractor_patterns = [
            r"(?i)(fake|fictional|hypothetical)",
            r"(?i)(ignore this|disregard)",
        ]

    def verify(self, data: DataRow, annotation: AnnotationResult) -> VerificationResult:
        issues: list[str] = []

        injection_issues = self._check_prompt_injection(data.content)
        issues.extend(injection_issues)

        context_issues = self._check_contradictory_context(data.content)
        issues.extend(context_issues)

        distractor_issues = self._check_distractors(data.content)
        issues.extend(distractor_issues)

        hallucination_issues = self._check_hallucination(annotation, data)
        issues.extend(hallucination_issues)

        is_valid = len(issues) == 0

        return VerificationResult(is_valid=is_valid, issues=issues)

    def _check_prompt_injection(self, text: str) -> list[str]:
        issues: list[str] = []
        for pattern in self.injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f"Potential prompt injection detected: {pattern}")
        return issues

    def _check_contradictory_context(self, text: str) -> list[str]:
        issues: list[str] = []
        contradictory_pairs = [
            (r"\bpositive\b", r"\bnegative\b"),
            (r"\btrue\b", r"\bfalse\b"),
            (r"\bagree\b", r"\bdisagree\b"),
        ]

        for pattern1, pattern2 in contradictory_pairs:
            has_1 = re.search(pattern1, text, re.IGNORECASE)
            has_2 = re.search(pattern2, text, re.IGNORECASE)
            if has_1 and has_2:
                issues.append("Context contains contradictory statements")

        return issues

    def _check_distractors(self, text: str) -> list[str]:
        issues: list[str] = []
        for pattern in self.distractor_patterns:
            if re.search(pattern, text):
                issues.append(f"Potential distractor detected: {pattern}")
        return issues

    def _check_hallucination(
        self,
        annotation: AnnotationResult,
        data: DataRow,
    ) -> list[str]:
        issues: list[str] = []

        for label in annotation.labels:
            if label.evidence_spans:
                for start, end in label.evidence_spans:
                    if start >= len(data.content) or end > len(data.content):
                        issues.append("Evidence span points outside document bounds")
                        break

        return issues


class RepairEngine:
    def __init__(self):
        self.consistency_checker = ConsistencyChecker()
        self.adversarial_checker = AdversarialChecker()

    def repair(
        self,
        data: DataRow,
        annotation: AnnotationResult,
    ) -> AnnotationResult:
        verification = self.adversarial_checker.verify(data, annotation)

        if not verification.is_valid:
            logger.info("repairing_annotation", issues=verification.issues)
            annotation = self._repair_issues(annotation, verification.issues)

        consistency = self.consistency_checker.verify(annotation, data)
        if not consistency.is_valid:
            annotation = self._repair_consistency(annotation, consistency.issues)

        annotation.status = AnnotationStatus.VERIFIED
        return annotation

    def _repair_issues(
        self,
        annotation: AnnotationResult,
        issues: list[str],
    ) -> AnnotationResult:
        for label in annotation.labels:
            if "hallucination" in str(issues).lower():
                label.evidence_spans = []
            if "confidence" in str(issues).lower():
                label.confidence = min(label.confidence, 0.8)

        return annotation

    def _repair_consistency(
        self,
        annotation: AnnotationResult,
        issues: list[str],
    ) -> AnnotationResult:
        for issue in issues:
            if "confidence" in issue.lower():
                for label in annotation.labels:
                    if label.confidence > 0.95:
                        label.confidence = 0.95

            if "evidence" in issue.lower():
                for label in annotation.labels:
                    label.evidence_spans = []

        return annotation


class QualityVerifier:
    def __init__(self):
        self.consistency_checker = ConsistencyChecker()
        self.adversarial_checker = AdversarialChecker()
        self.repair_engine = RepairEngine()

    def verify_and_repair(
        self,
        data: DataRow,
        annotation: AnnotationResult,
    ) -> AnnotationResult:
        verification = self.adversarial_checker.verify(data, annotation)

        if not verification.is_valid:
            annotation = self.repair_engine.repair(data, annotation)

        consistency = self.consistency_checker.verify(annotation, data)
        if not consistency.is_valid:
            annotation = self._apply_fixes(annotation, consistency)

        return annotation

    def _apply_fixes(
        self,
        annotation: AnnotationResult,
        consistency: VerificationResult,
    ) -> AnnotationResult:
        for issue in consistency.issues:
            if "confidence" in issue.lower():
                for label in annotation.labels:
                    label.confidence = max(0.1, min(0.95, label.confidence))

        return annotation