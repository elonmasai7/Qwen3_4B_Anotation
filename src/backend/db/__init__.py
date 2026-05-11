from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text,
    JSON, ForeignKey, Index, Enum as SQLEnum,
)
from sqlalchemy.orm import DeclarativeBase, relationship
from datetime import datetime, timezone
import uuid
import enum


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class DatasetStatus(str, enum.Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class AnnotationStatusDB(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    source = Column(String, nullable=True)
    format = Column(String, nullable=False)
    status = Column(SQLEnum(DatasetStatus), default=DatasetStatus.UPLOADING)
    row_count = Column(Integer, default=0)
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    data_rows = relationship("DataRow", back_populates="dataset")
    experiments = relationship("Experiment", back_populates="dataset")


class DataRow(Base):
    __tablename__ = "data_rows"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False)
    content = Column(Text, nullable=False)
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=utcnow)

    dataset = relationship("Dataset", back_populates="data_rows")
    annotations = relationship("Annotation", back_populates="data_row")

    __table_args__ = (Index("idx_dataset_id", "dataset_id"),)


class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    data_row_id = Column(String, ForeignKey("data_rows.id"), nullable=False)
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=True)
    labels = Column(JSON, default=list)
    status = Column(SQLEnum(AnnotationStatusDB), default=AnnotationStatusDB.PENDING)
    confidence = Column(Float, default=0.0)
    rationale = Column(Text, nullable=True)
    evidence_spans = Column(JSON, default=list)
    processing_time_ms = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    data_row = relationship("DataRow", back_populates="annotations")

    __table_args__ = (
        Index("idx_data_row_id", "data_row_id"),
        Index("idx_experiment_id", "experiment_id"),
    )


class PromptTemplateDB(Base):
    __tablename__ = "prompt_templates"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    instruction = Column(Text, nullable=False)
    task_definition = Column(Text, nullable=False)
    output_schema = Column(Text, nullable=False)
    cot_scaffold = Column(Text, nullable=True)
    examples = Column(JSON, default=list)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    config = Column(JSON, default=dict)
    status = Column(String, default="pending")
    metrics = Column(JSON, default=dict)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
    completed_at = Column(DateTime, nullable=True)

    dataset = relationship("Dataset", back_populates="experiments")
    runs = relationship("ExperimentRun", back_populates="experiment")


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=False)
    prompt_template = Column(Text, nullable=False)
    metrics = Column(JSON, default=dict)
    status = Column(String, default="running")
    started_at = Column(DateTime, default=utcnow)
    completed_at = Column(DateTime, nullable=True)

    experiment = relationship("Experiment", back_populates="runs")


class LeaderboardSubmission(Base):
    __tablename__ = "leaderboard_submissions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=False)
    metrics = Column(JSON, default=dict)
    rank = Column(Integer, nullable=True)
    submitted_at = Column(DateTime, default=utcnow)


def create_tables(engine):
    Base.metadata.create_all(engine)


def get_db_url(settings) -> str:
    return f"postgresql+asyncpg://{settings.database.user}:{settings.database.password}@{settings.database.host}:{settings.database.port}/{settings.database.name}"