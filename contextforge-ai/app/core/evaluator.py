"""Evaluation metrics for predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Evaluator:
    @staticmethod
    def _confusion(y_true: list[str], y_pred: list[str], labels: list[str]) -> np.ndarray:
        index = {label: i for i, label in enumerate(labels)}
        matrix = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(y_true, y_pred, strict=False):
            matrix[index[t], index[p]] += 1
        return matrix

    @staticmethod
    def _macro_precision_recall_f1(cm: np.ndarray) -> tuple[float, float, float]:
        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []
        for i in range(cm.shape[0]):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))

    def evaluate(self, pred_rows: list[dict], gold_rows: list[dict], latency_s: float = 0.0) -> dict:
        gold_map = {row["id"]: row for row in gold_rows}
        y_true: list[str] = []
        y_pred: list[str] = []
        confidences: list[float] = []

        for pred in pred_rows:
            sample_id = pred["id"]
            if sample_id not in gold_map:
                continue
            y_pred.append(str(pred.get("label", "")))
            y_true.append(str(gold_map[sample_id].get("label", "")))
            confidences.append(float(pred.get("confidence", 0.0) or 0.0))

        labels = sorted(set(y_true) | set(y_pred))
        cm = self._confusion(y_true, y_pred, labels)
        macro_precision, macro_recall, macro_f1 = self._macro_precision_recall_f1(cm)
        micro_tp = sum(cm[i, i] for i in range(len(labels)))
        total = int(cm.sum())
        micro_precision = micro_tp / max(total, 1)
        micro_recall = micro_tp / max(total, 1)
        micro_f1 = (
            0.0
            if (micro_precision + micro_recall) == 0
            else 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        )
        accuracy = float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else 0.0
        report = {
            "accuracy": accuracy,
            "precision": macro_precision,
            "recall": macro_recall,
            "macro_f1": macro_f1,
            "micro_f1": float(micro_f1),
            "exact_match": accuracy,
            "confusion_matrix": {
                "labels": labels,
                "matrix": cm.tolist(),
            },
            "average_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "latency_per_sample": float(latency_s / max(len(pred_rows), 1)),
            "throughput": float(len(pred_rows) / max(latency_s, 1e-9)),
        }
        return report

    def save_report(self, report: dict, json_path: str, md_path: str) -> None:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(json.dumps(report, indent=2), encoding="utf-8")

        lines = [
            "# Evaluation Report",
            "",
            f"- Accuracy: {report['accuracy']:.4f}",
            f"- Precision: {report['precision']:.4f}",
            f"- Recall: {report['recall']:.4f}",
            f"- Macro F1: {report['macro_f1']:.4f}",
            f"- Micro F1: {report['micro_f1']:.4f}",
            f"- Exact Match: {report['exact_match']:.4f}",
            f"- Avg Confidence: {report['average_confidence']:.4f}",
            f"- Latency per Sample: {report['latency_per_sample']:.4f}s",
            f"- Throughput: {report['throughput']:.2f} samples/s",
        ]
        Path(md_path).write_text("\n".join(lines), encoding="utf-8")
