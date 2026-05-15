"""Queue-backed async annotation jobs using database persistence."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

from sqlalchemy import select

from app.core.annotation_pipeline import AnnotationPipeline
from app.db.database import SessionLocal
from app.db.models import AnnotationJobRecord


class AnnotationJobQueue:
    def __init__(self) -> None:
        self.pipeline = AnnotationPipeline()

    def enqueue(self, sample: dict, example_pool: list[dict] | None = None) -> str:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        payload = {
            "sample": sample,
            "example_pool": example_pool or [],
        }
        with SessionLocal() as db:
            db.add(
                AnnotationJobRecord(
                    job_id=job_id,
                    status="queued",
                    payload=payload,
                    result={},
                    error="",
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            )
            db.commit()
        return job_id

    def get_job(self, job_id: str) -> dict | None:
        with SessionLocal() as db:
            record = db.scalar(select(AnnotationJobRecord).where(AnnotationJobRecord.job_id == job_id))
            if not record:
                return None
            return {
                "job_id": record.job_id,
                "status": record.status,
                "result": record.result,
                "error": record.error,
            }

    async def run_worker_loop(self, poll_interval_s: float = 0.2) -> None:
        while True:
            processed = await self._process_once()
            if not processed:
                await asyncio.sleep(poll_interval_s)

    async def _process_once(self) -> bool:
        with SessionLocal() as db:
            record = db.scalar(
                select(AnnotationJobRecord)
                .where(AnnotationJobRecord.status == "queued")
                .order_by(AnnotationJobRecord.created_at.asc())
                .limit(1)
            )
            if record is None:
                return False

            record.status = "running"
            record.updated_at = datetime.now(UTC)
            db.commit()

            sample = record.payload.get("sample", {})
            example_pool = record.payload.get("example_pool", [])

            try:
                result, latency = self.pipeline.annotate_sample(sample, example_pool)
                record.status = "completed"
                record.result = {
                    "prediction": result,
                    "latency": latency,
                }
                record.error = ""
            except Exception as exc:  # pragma: no cover
                record.status = "failed"
                record.result = {}
                record.error = str(exc)

            record.updated_at = datetime.now(UTC)
            db.commit()
            return True
