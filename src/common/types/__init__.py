from typing import Any, Literal, TypeAlias
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import uuid

class DataFormat(str, Enum):
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    TSV = "tsv"
    PARQUET = "parquet"
    XML = "xml"
    YAML = "yaml"

class AnnotationStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"

class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ChunkStrategy(str, Enum):
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    RECURSIVE = "recursive"
    SLIDING = "sliding"

class VotingStrategy(str, Enum):
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    ENTROPY = "entropy"
    CONFIDENCE = "confidence"

DatasetId: TypeAlias = str
AnnotationId: TypeAlias = str
PromptId: TypeAlias = str
ExperimentId: TypeAlias = str

class Metadata(BaseModel):
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1
    tags: list[str] = Field(default_factory=list)
    source: str | None = None
    owner: str | None = None

class DataRow(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: Metadata = Field(default_factory=Metadata)
    raw_data: dict[str, Any] = Field(default_factory=dict)

class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    start_idx: int
    end_idx: int
    chunk_index: int
    embedding: list[float] | None = None
    importance_score: float | None = None

class Example(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input_text: str
    output: Any
    rationale: str | None = None
    is_counterexample: bool = False
    embedding: list[float] | None = None

class AnnotationLabel(BaseModel):
    value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    evidence_spans: list[tuple[int, int]] = Field(default_factory=list)
    alternative_hypotheses: list[str] = Field(default_factory=list)

class AnnotationResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    data_id: str
    labels: list[AnnotationLabel]
    status: AnnotationStatus = AnnotationStatus.PENDING
    processing_time_ms: float | None = None
    error_message: str | None = None

class VerificationResult(BaseModel):
    is_valid: bool
    issues: list[str] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    repaired_output: dict[str, Any] | None = None

class EvaluationMetrics(BaseModel):
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    macro_f1: float | None = None
    micro_f1: float | None = None
    exact_match: float | None = None
    calibration_error: float | None = None
    latency_ms: float | None = None
    throughput: float | None = None
    cost_per_sample: float | None = None

class ExperimentConfig(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    prompt_template: str
    num_examples: int = 5
    chunk_strategy: ChunkStrategy = ChunkStrategy.SEMANTIC
    voting_strategy: VotingStrategy = VotingStrategy.MAJORITY
    temperature: float = 0.1
    top_p: float = 0.95
    max_tokens: int = 2048
    num_branches: int = 3
    retrieval_method: str = "hybrid"
    created_at: datetime = Field(default_factory=datetime.utcnow)

class PromptTemplate(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    instruction: str
    task_definition: str
    output_schema: str
    cot_scaffold: str
    examples: list[Example] = Field(default_factory=list)

class LeaderboardSubmission(BaseModel):
    experiment_id: str
    metrics: EvaluationMetrics
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)