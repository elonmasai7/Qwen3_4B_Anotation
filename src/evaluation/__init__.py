from typing import Any
import time
from collections import defaultdict
from common.types import (
    AnnotationResult, EvaluationMetrics, ExperimentConfig,
)
from common.logging import get_logger

logger = get_logger(__name__)


class MetricsCalculator:
    @staticmethod
    def calculate_accuracy(
        predictions: list[AnnotationResult],
        ground_truths: list[Any],
    ) -> float:
        if not predictions or not ground_truths:
            return 0.0

        correct = 0
        for pred, gt in zip(predictions, ground_truths):
            if pred.labels:
                pred_label = str(pred.labels[0].value)
                if pred_label == str(gt):
                    correct += 1

        return correct / len(ground_truths)

    @staticmethod
    def calculate_precision(
        predictions: list[AnnotationResult],
        ground_truths: list[Any],
    ) -> float:
        if not predictions:
            return 0.0

        true_positives = 0
        false_positives = 0

        for pred, gt in zip(predictions, ground_truths):
            if pred.labels:
                pred_label = str(pred.labels[0].value)
                if pred_label == str(gt):
                    true_positives += 1
                else:
                    false_positives += 1

        denom = true_positives + false_positives
        return true_positives / denom if denom > 0 else 0.0

    @staticmethod
    def calculate_recall(
        predictions: list[AnnotationResult],
        ground_truths: list[Any],
    ) -> float:
        if not ground_truths:
            return 0.0

        true_positives = 0
        false_negatives = 0

        for pred, gt in zip(predictions, ground_truths):
            gt_val = str(gt)
            if gt_val:
                if pred.labels and str(pred.labels[0].value) == gt_val:
                    true_positives += 1
                else:
                    false_negatives += 1

        denom = true_positives + false_negatives
        return true_positives / denom if denom > 0 else 0.0

    @staticmethod
    def calculate_f1(
        predictions: list[AnnotationResult],
        ground_truths: list[Any],
    ) -> float:
        precision = MetricsCalculator.calculate_precision(predictions, ground_truths)
        recall = MetricsCalculator.calculate_recall(predictions, ground_truths)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def calculate_macro_f1(
        predictions: list[AnnotationResult],
        ground_truths: list[Any],
    ) -> float:
        label_groups: dict[str, tuple[list[AnnotationResult], list[Any]]] = defaultdict(lambda: ([], []))

        for pred, gt in zip(predictions, ground_truths):
            label = str(gt)
            label_groups[label][0].append(pred)
            label_groups[label][1].append(gt)

        f1_scores = []
        for label, (preds, gts) in label_groups.items():
            f1 = MetricsCalculator.calculate_f1(preds, gts)
            f1_scores.append(f1)

        return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    @staticmethod
    def calculate_micro_f1(
        predictions: list[AnnotationResult],
        ground_truths: list[Any],
    ) -> float:
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for pred, gt in zip(predictions, ground_truths):
            pred_val = str(pred.labels[0].value) if pred.labels else ""
            gt_val = str(gt)

            if pred_val == gt_val and pred_val:
                true_positives += 1
            elif pred_val != gt_val:
                if pred_val:
                    false_positives += 1
                if gt_val:
                    false_negatives += 1

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def calculate_exact_match(
        predictions: list[AnnotationResult],
        ground_truths: list[Any],
    ) -> float:
        if not predictions or not ground_truths:
            return 0.0

        exact_matches = 0
        for pred, gt in zip(predictions, ground_truths):
            if pred.labels:
                pred_str = str(pred.labels[0].value).strip().lower()
                gt_str = str(gt).strip().lower()
                if pred_str == gt_str:
                    exact_matches += 1

        return exact_matches / len(ground_truths)

    @staticmethod
    def calculate_calibration_error(
        predictions: list[AnnotationResult],
        ground_truths: list[Any],
    ) -> float:
        bins: dict[tuple[float, float], tuple[list[float], list[int]]] = defaultdict(lambda: ([], []))

        bin_size = 0.1
        for pred, gt in zip(predictions, ground_truths):
            if pred.labels:
                conf = pred.labels[0].confidence
                bin_key = (int(conf / bin_size) * bin_size, (int(conf / bin_size) + 1) * bin_size)
                is_correct = 1 if str(pred.labels[0].value) == str(gt) else 0
                bins[bin_key][0].append(conf)
                bins[bin_key][1].append(is_correct)

        total_error = 0.0
        for (bin_start, _), (confs, correct) in bins.items():
            if confs:
                avg_confidence = sum(confs) / len(confs)
                accuracy = sum(correct) / len(correct)
                total_error += abs(avg_confidence - accuracy) * len(confs)

        total_samples = sum(len(v[0]) for v in bins.values())
        return total_error / total_samples if total_samples > 0 else 0.0


class LatencyTracker:
    def __init__(self) -> None:
        self.latencies: list[float] = []
        self.start_time: float | None = None

    def start(self) -> None:
        self.start_time = time.time()

    def stop(self) -> float:
        if self.start_time is None:
            return 0.0
        latency = (time.time() - self.start_time) * 1000
        self.latencies.append(latency)
        self.start_time = None
        return latency

    def get_average(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    def get_percentile(self, percentile: int) -> float:
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * percentile / 100)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]


class EvaluationEngine:
    def __init__(self) -> None:
        self.metrics_calc = MetricsCalculator()
        self.latency_tracker = LatencyTracker()
        self.total_cost = 0.0
        self.total_samples = 0

    def evaluate(
        self,
        predictions: list[AnnotationResult],
        ground_truths: list[Any],
        cost: float | None = None,
    ) -> EvaluationMetrics:
        accuracy = self.metrics_calc.calculate_accuracy(predictions, ground_truths)
        precision = self.metrics_calc.calculate_precision(predictions, ground_truths)
        recall = self.metrics_calc.calculate_recall(predictions, ground_truths)
        f1 = self.metrics_calc.calculate_f1(predictions, ground_truths)
        macro_f1 = self.metrics_calc.calculate_macro_f1(predictions, ground_truths)
        micro_f1 = self.metrics_calc.calculate_micro_f1(predictions, ground_truths)
        exact_match = self.metrics_calc.calculate_exact_match(predictions, ground_truths)
        calibration_error = self.metrics_calc.calculate_calibration_error(predictions, ground_truths)

        avg_latency = self.latency_tracker.get_average()
        throughput = 1000 / avg_latency if avg_latency > 0 else 0

        cost_per_sample = self.total_cost / self.total_samples if self.total_samples > 0 else 0

        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            macro_f1=macro_f1,
            micro_f1=micro_f1,
            exact_match=exact_match,
            calibration_error=calibration_error,
            latency_ms=avg_latency,
            throughput=throughput,
            cost_per_sample=cost_per_sample,
        )

    def compare_experiments(
        self,
        experiments: list[tuple[ExperimentConfig, EvaluationMetrics]],
    ) -> dict[str, Any]:
        comparison = {
            "experiments": [],
            "best_by_metric": {},
        }

        best_scores = {
            "accuracy": (None, 0.0),
            "f1": (None, 0.0),
            "latency_ms": (None, float("inf")),
            "cost_per_sample": (None, float("inf")),
        }

        for config, metrics in experiments:
            exp_data = {
                "id": config.id,
                "name": config.name,
                "metrics": metrics.model_dump(),
            }
            comparison["experiments"].append(exp_data)

            for metric_name, (best_id, best_val) in best_scores.items():
                metric_val = getattr(metrics, metric_name, None)
                if metric_val is not None:
                    is_better = (
                        metric_val > best_val
                        if metric_name in ["accuracy", "f1"]
                        else metric_val < best_val
                    )
                    if is_better or best_id is None:
                        best_scores[metric_name] = (config.id, metric_val)

        comparison["best_by_metric"] = {
            k: {"id": v[0], "value": v[1]}
            for k, v in best_scores.items()
        }

        return comparison