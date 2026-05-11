from typing import Any
import asyncio
import json
from datetime import datetime, timezone
from common.types import (
    AnnotationLabel, AnnotationResult, AnnotationStatus,
    DataRow,
)
from common.config import get_settings
from common.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class AnnotationReasoner:
    def __init__(self, model_client: Any = None):
        self.model_client = model_client

    async def reason(
        self,
        data: DataRow,
        prompt: str,
        max_retries: int | None = None,
    ) -> AnnotationResult:
        retries = max_retries if max_retries is not None else settings.annotation.max_retries
        start_time = datetime.now(timezone.utc)

        for attempt in range(retries):
            try:
                output = await self._generate(prompt)
                parsed = self._parse_output(output)

                label = self._create_label(parsed)
                status = AnnotationStatus.COMPLETED

                processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

                return AnnotationResult(
                    data_id=data.id,
                    labels=[label],
                    status=status,
                    processing_time_ms=processing_time,
                )
            except Exception as e:
                logger.warning(
                    "annotation_attempt_failed",
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt == retries - 1:
                    return AnnotationResult(
                        data_id=data.id,
                        labels=[],
                        status=AnnotationStatus.FAILED,
                        error_message=str(e),
                    )

        return AnnotationResult(
            data_id=data.id,
            labels=[],
            status=AnnotationStatus.FAILED,
        )

    async def _generate(self, prompt: str) -> str:
        if self.model_client:
            return await self.model_client.generate(prompt)
        return await self._simulate_generation(prompt)

    async def _simulate_generation(self, prompt: str) -> str:
        await asyncio.sleep(0.1)
        return json.dumps({
            "label": "sample_label",
            "confidence": 0.85,
            "rationale": "This is a simulated response for testing.",
            "evidence_spans": [[0, 50]],
            "alternatives": ["alternative_1"],
        })

    def _parse_output(self, output: str) -> dict[str, Any]:
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return self._fallback_parse(output)

    def _fallback_parse(self, output: str) -> dict[str, Any]:
        return {
            "label": output.strip()[:100],
            "confidence": 0.5,
            "rationale": "Parse failed, using raw output",
            "evidence_spans": [],
            "alternatives": [],
        }

    def _create_label(self, parsed: dict[str, Any]) -> AnnotationLabel:
        return AnnotationLabel(
            value=parsed.get("label", "unknown"),
            confidence=parsed.get("confidence", 0.5),
            rationale=parsed.get("rationale", ""),
            evidence_spans=parsed.get("evidence_spans", []),
            alternative_hypotheses=parsed.get("alternatives", []),
        )


class MultiPassReasoner:
    def __init__(self, reasoner: AnnotationReasoner):
        self.reasoner = reasoner
        self.passes = ["draft", "self_critique", "repair", "confidence_calibration", "schema_validation"]

    async def reason(
        self,
        data: DataRow,
        prompt: str,
    ) -> AnnotationResult:
        current_prompt = prompt
        intermediate_results: list[dict[str, Any]] = []

        for pass_name in self.passes:
            logger.info("reasoning_pass", pass_name=pass_name, data_id=data.id)

            result = await self.reasoner.reason(data, current_prompt)
            intermediate_results.append({
                "pass": pass_name,
                "result": result.labels[0].__dict__ if result.labels else {},
            })

            if pass_name == "self_critique" and result.labels:
                current_prompt = self._add_critique_to_prompt(prompt, result.labels[0])

            elif pass_name == "repair" and result.labels:
                current_prompt = self._add_repair_guidance(prompt, result.labels[0])

            elif pass_name == "confidence_calibration":
                result = self._calibrate_confidence(result)

            elif pass_name == "schema_validation":
                result = self._validate_schema(result)

        final_result = intermediate_results[-1]["result"] if intermediate_results else {}

        return AnnotationResult(
            data_id=data.id,
            labels=[AnnotationLabel(**final_result)] if final_result else [],
            status=AnnotationStatus.COMPLETED,
        )

    def _add_critique_to_prompt(self, prompt: str, label: AnnotationLabel) -> str:
        critique = f"""
        Previous analysis: {label.rationale}
        Confidence: {label.confidence}

        Review and improve your response. Consider:
        - Is the evidence strong enough?
        - Are there alternative explanations?
        - Is the confidence appropriately calibrated?
        """
        return prompt + critique

    def _add_repair_guidance(self, prompt: str, label: AnnotationLabel) -> str:
        guidance = f"""
        Issues to address: {label.rationale}

        Please provide a corrected and improved response.
        """
        return prompt + guidance

    def _calibrate_confidence(self, result: AnnotationResult) -> AnnotationResult:
        if not result.labels:
            return result

        for label in result.labels:
            if label.confidence > 0.95:
                label.confidence = min(label.confidence, 0.95)
            elif label.confidence < 0.1:
                label.confidence = max(label.confidence, 0.1)

        return result

    def _validate_schema(self, result: AnnotationResult) -> AnnotationResult:
        if not result.labels:
            return result

        for label in result.labels:
            if not label.rationale:
                label.rationale = "No rationale provided"
            if not label.evidence_spans:
                logger.warning("missing_evidence_spans", data_id=result.data_id)

        return result


class SelfConsistencyEngine:
    def __init__(
        self,
        reasoner: AnnotationReasoner,
        num_branches: int | None = None,
    ):
        self.reasoner = reasoner
        self.num_branches = num_branches or settings.annotation.num_branches

    async def annotate(
        self,
        data: DataRow,
        prompt: str,
        voting_strategy: str = "majority",
    ) -> AnnotationResult:
        tasks = [
            self.reasoner.reason(data, prompt)
            for _ in range(self.num_branches)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = [r for r in results if isinstance(r, AnnotationResult) and r.labels]

        if not valid_results:
            return AnnotationResult(
                data_id=data.id,
                labels=[],
                status=AnnotationStatus.FAILED,
                error_message="All branches failed",
            )

        final_label = self._aggregate_labels(valid_results, voting_strategy)

        return AnnotationResult(
            data_id=data.id,
            labels=[final_label],
            status=AnnotationStatus.COMPLETED,
            processing_time_ms=valid_results[0].processing_time_ms,
        )

    def _aggregate_labels(
        self,
        results: list[AnnotationResult],
        strategy: str,
    ) -> AnnotationLabel:
        if strategy == "majority":
            return self._majority_vote(results)
        elif strategy == "weighted":
            return self._weighted_vote(results)
        elif strategy == "entropy":
            return self._entropy_weighted(results)
        elif strategy == "confidence":
            return self._confidence_weighted(results)
        else:
            return self._majority_vote(results)

    def _majority_vote(self, results: list[AnnotationResult]) -> AnnotationLabel:
        label_counts: dict[str, int] = {}
        for r in results:
            if r.labels:
                label_val = str(r.labels[0].value)
                label_counts[label_val] = label_counts.get(label_val, 0) + 1

        majority_label = max(label_counts, key=label_counts.get)
        confidence = label_counts[majority_label] / len(results)

        return AnnotationLabel(
            value=majority_label,
            confidence=confidence,
            rationale="Aggregated via majority voting",
        )

    def _weighted_vote(self, results: list[AnnotationResult]) -> AnnotationLabel:
        label_scores: dict[str, float] = {}
        for r in results:
            if r.labels:
                label_val = str(r.labels[0].value)
                label_scores[label_val] = label_scores.get(label_val, 0) + r.labels[0].confidence

        best_label = max(label_scores, key=label_scores.get)
        confidence = label_scores[best_label] / len(results)

        return AnnotationLabel(
            value=best_label,
            confidence=min(confidence, 1.0),
            rationale="Aggregated via weighted voting",
        )

    def _entropy_weighted(self, results: list[AnnotationResult]) -> AnnotationLabel:
        label_entropies: dict[str, list[float]] = {}

        for r in results:
            if r.labels:
                label_val = str(r.labels[0].value)
                if label_val not in label_entropies:
                    label_entropies[label_val] = []
                label_entropies[label_val].append(r.labels[0].confidence)

        label_avg_conf = {
            label: sum(confs) / len(confs)
            for label, confs in label_entropies.items()
        }

        best_label = max(label_avg_conf, key=label_avg_conf.get)
        confidence = label_avg_conf[best_label]

        return AnnotationLabel(
            value=best_label,
            confidence=confidence,
            rationale="Aggregated via entropy-weighted voting",
        )

    def _confidence_weighted(self, results: list[AnnotationResult]) -> AnnotationLabel:
        label_weights: dict[str, float] = {}
        label_total_conf: dict[str, float] = {}

        for r in results:
            if r.labels:
                label_val = str(r.labels[0].value)
                conf = r.labels[0].confidence
                weight = conf * conf

                label_weights[label_val] = label_weights.get(label_val, 0) + weight
                label_total_conf[label_val] = label_total_conf.get(label_val, 0) + conf

        best_label = max(label_weights, key=label_weights.get)
        confidence = label_total_conf[best_label] / len(results)

        return AnnotationLabel(
            value=best_label,
            confidence=confidence,
            rationale="Aggregated via confidence-weighted voting",
        )