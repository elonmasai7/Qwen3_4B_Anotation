"""Timing and throughput metrics helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RunMetrics:
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    total_samples: int = 0

    def finish(self, total_samples: int) -> None:
        self.finished_at = time.time()
        self.total_samples = total_samples

    @property
    def latency(self) -> float:
        end = self.finished_at if self.finished_at else time.time()
        return max(end - self.started_at, 1e-9)

    @property
    def throughput(self) -> float:
        return float(self.total_samples) / self.latency
