"""Relational models for datasets, annotations, and experiments."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class DatasetRecord(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    source_path: Mapped[str] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    annotations: Mapped[list["AnnotationRecord"]] = relationship(back_populates="dataset")


class AnnotationRecord(Base):
    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), index=True)
    sample_id: Mapped[str] = mapped_column(String(128), index=True)
    label: Mapped[str] = mapped_column(String(128))
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    evidence: Mapped[list[str]] = mapped_column(JSON, default=list)
    rationale: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    dataset: Mapped[DatasetRecord] = relationship(back_populates="annotations")


class ExperimentRecord(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(String(64), unique=True)
    model: Mapped[str] = mapped_column(String(255), default="Qwen3-4B")
    retrieval_method: Mapped[str] = mapped_column(String(64), default="hybrid")
    chunk_size: Mapped[int] = mapped_column(Integer, default=2048)
    overlap: Mapped[int] = mapped_column(Integer, default=256)
    examples: Mapped[int] = mapped_column(Integer, default=6)
    temperature: Mapped[float] = mapped_column(Float, default=0.1)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    latency: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AnnotationJobRecord(Base):
    __tablename__ = "annotation_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
