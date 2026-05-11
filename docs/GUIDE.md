# Annotation Platform — Comprehensive Guide

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Project Structure](#3-project-structure)
4. [Quick Start](#4-quick-start)
5. [Configuration](#5-configuration)
6. [Data Ingestion](#6-data-ingestion)
7. [Annotation Pipeline](#7-annotation-pipeline)
8. [Verification & Quality](#8-verification--quality)
9. [Evaluation](#9-evaluation)
10. [Active Learning](#10-active-learning)
11. [Agent System](#11-agent-system)
12. [API Reference](#12-api-reference)
13. [UI Dashboard](#13-ui-dashboard)
14. [Docker Deployment](#14-docker-deployment)
15. [Testing](#15-testing)
16. [CI/CD](#16-cicd)

---

## 1. Overview

The **Qwen3-4B Automatic Data Annotation Platform** is a production-grade system for automating data annotation using the Qwen3-4B language model. It is designed for long-context scenarios with in-context learning (ICL)-based annotation, quality verification, and leaderboard-ready submissions.

### Core Capabilities

| Capability | Description |
|---|---|
| Multi-Format Ingestion | JSON, JSONL, CSV, TSV, Parquet, XML, YAML with streaming and validation |
| Long-Context Processing | 100k+ token contexts with semantic/hierarchical chunking and memory compression |
| ICL Prompt Construction | Dynamic example retrieval with cosine similarity, BM25, reranking, and hybrid search |
| Multi-Pass Reasoning | Draft → Self-Critique → Repair → Confidence Calibration → Schema Validation |
| Self-Consistency | N-branch inference with majority/entropy/confidence-weighted voting |
| Auto Verification | Consistency checks, adversarial detection, and automatic repair |
| Active Learning | Uncertainty sampling, disagreement sampling, weak label detection |
| Evaluation Suite | Accuracy, precision, recall, F1, macro/micro F1, exact match, calibration error |

### Dataset

The platform ships with `data/FINAL_DATASET.csv` (6,557 rows), a face anti-spoofing dataset with REAL/FAKE labels for image classification. This dataset can be used to test the ingestion, annotation, and evaluation pipelines.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      UI Layer (Jinja2 + JS)                      │
│  Dashboard | Annotation | Datasets | Experiments | Prompts | LB  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    FastAPI Backend (src/main.py)                 │
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────┐ │
│  │ Cache    │ │ Messaging│ │ Storage  │ │ DB       │ │ API   │ │
│  │ (Redis)  │ │ (Kafka)  │ │ (MinIO)  │ │(Postgres)│ │ REST  │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └───────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                Annotation Intelligence Engine                    │
│                                                                  │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Prompt Constructor│  │ Multi-Pass   │  │ Self-Consistency  │  │
│  │ (ICL + Retrieval) │  │ Reasoner     │  │ Engine (N-branch) │  │
│  └─────────────────┘  └──────────────┘  └───────────────────┘  │
│                                                                  │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Quality Verifier │  │ Evaluation   │  │ Active Learning   │  │
│  │ (Adversarial +   │  │ Engine       │  │ Engine            │  │
│  │  Consistency)    │  │              │  │                   │  │
│  └─────────────────┘  └──────────────┘  └───────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    Qwen3-4B Inference Layer                      │
│                                                                  │
│  ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌───────────────┐   │
│  │ vLLM     │ │ Quantization│ │ KV Cache │ │ GPU Scheduler │   │
│  │ Engine   │ │ (FP16/INT8) │ │ Optimizer│ │               │   │
│  └──────────┘ └────────────┘ └──────────┘ └───────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Request → FastAPI → Annotation Reasoner → Multi-Pass Pipeline
                                                      │
                                                      ▼
                                              Self-Consistency
                                              (N parallel runs)
                                                      │
                                                      ▼
                                              Quality Verifier
                                              (Adversarial + Consistency)
                                                      │
                                                      ▼
                                              Evaluation Engine
                                              (Metrics Computation)
                                                      │
                                                      ▼
                                              Response to User
```

---

## 3. Project Structure

```
annotation-platform/
├── config/                        # YAML configuration files
│   ├── base.yaml                  # Main configuration
│   ├── environments/              # Environment-specific configs
│   ├── prompts/                   # Prompt templates
│   ├── schemas/                   # Output schemas
│   └── templates/                 # Annotation templates
├── data/
│   └── FINAL_DATASET.csv          # Sample dataset (6557 rows, REAL/FAKE)
├── docker/
│   ├── Dockerfile                 # Production Docker image
│   └── docker-compose.yaml        # Full stack (API + Postgres + Redis + Kafka + MinIO + Prometheus)
├── docs/
│   ├── GUIDE.md                   # This guide
│   ├── api/                       # API documentation
│   ├── architecture/              # Architecture docs
│   └── guides/                    # Usage guides
├── k8s/                           # Kubernetes manifests (stub)
├── scripts/
│   └── setup.sh                   # One-shot development setup
├── src/                           # Main source code
│   ├── main.py                    # FastAPI entry point with lifespan management
│   ├── agents/                    # 12 autonomous specialist agents
│   │   ├── BaseAgent              # Abstract base for all agents
│   │   ├── ResearchEngineer       # Research tasks
│   │   ├── PromptEngineer         # Prompt optimization
│   │   ├── RetrievalEngineer      # Retrieval strategy
│   │   ├── EvaluationEngineer     # Metric computation
│   │   ├── BackendEngineer        # API/service health
│   │   ├── MLEngineer             # Model optimization
│   │   ├── MLOpsEngineer          # Deployment/CI
│   │   ├── QAEngineer             # Testing
│   │   ├── SecurityEngineer       # Vulnerability scanning
│   │   ├── DocumentationEngineer  # Doc generation
│   │   ├── OptimizationEngineer   # Performance optimization
│   │   ├── ExperimentScientist    # Experiment design
│   │   └── AgentOrchestrator      # Multi-agent pipeline runner
│   ├── annotation/                # Core annotation engine
│   │   ├── memory_engine/         # Long-context memory management
│   │   ├── prompt_constructor/    # ICL prompt construction + dynamic retrieval
│   │   │   ├── PromptConstructor  # Assembles prompts from templates + examples + chunks
│   │   │   ├── DynamicExampleRetriever  # Semantic/BM25/hybrid retrieval
│   │   │   └── PromptOptimizer    # Tests prompt variations, picks best
│   │   ├── reasoner/              # Multi-pass reasoning + self-consistency
│   │   │   ├── AnnotationReasoner # Single-pass LLM annotation
│   │   │   ├── MultiPassReasoner  # 5-pass pipeline (draft→critique→repair→calibrate→validate)
│   │   │   └── SelfConsistencyEngine  # N-branch inference + voting
│   │   └── self_consistency/      # Re-exports from reasoner
│   ├── backend/                   # FastAPI backend services
│   │   ├── api/                   # REST API endpoints (mounted at /api/v1)
│   │   ├── cache/                 # Redis cache manager
│   │   ├── db/                    # SQLAlchemy ORM models (Dataset, DataRow, Annotation, Experiment, etc.)
│   │   ├── messaging/             # Kafka producer/consumer for async tasks
│   │   └── storage/               # MinIO object storage client
│   ├── common/                    # Shared infrastructure
│   │   ├── types/                 # Pydantic models, enums, type aliases
│   │   ├── config/                # Hierarchical Pydantic Settings from YAML + env vars
│   │   └── logging/               # Structured logging (structlog + JSON)
│   ├── evaluation/                # Metrics and benchmarks
│   │   ├── MetricsCalculator      # Accuracy, precision, recall, F1, calibration error
│   │   ├── LatencyTracker         # Perf tracking
│   │   └── EvaluationEngine       # Full eval + experiment comparison
│   ├── ingestion/                 # Data loading pipeline
│   │   ├── loaders/               # Multi-format loaders (JSON, JSONL, CSV, TSV, Parquet, XML, YAML)
│   │   ├── normalizers/           # Data normalization
│   │   ├── processors/            # Chunking strategies (semantic, hierarchical, recursive, sliding), memory compression, summarization
│   │   └── validators/            # Data validation
│   ├── optimization/              # Inference optimization
│   │   ├── VLLMEngine             # vLLM integration (batch/async generation)
│   │   ├── TokenBudgetPlanner     # Token allocation across prompt parts
│   │   ├── QuantizationManager    # FP16/INT8/INT4 precision
│   │   ├── SpeculativeDecoding    # Draft model speculation
│   │   ├── KVCacheOptimizer       # LRU KV cache
│   │   ├── AsyncBatcher           # Dynamic prompt batching
│   │   └── GPUManager             # GPU load tracking and selection
│   ├── training/                  # Active learning + prompt evolution
│   │   ├── ActiveLearningEngine   # Uncertainty/disagreement/weak label sampling
│   │   ├── SyntheticExampleGenerator  # Text perturbation + counterexample generation
│   │   ├── PromptEvolutionEngine  # Genetic algorithm prompt search
│   │   └── GeneticPromptSearch    # Convenience wrapper
│   ├── ui/                        # Web dashboard
│   │   ├── routes.py              # Jinja2 page routes
│   │   ├── templates/             # HTML templates (base, dashboard, annotation, datasets, experiments, prompts, leaderboard)
│   │   └── static/                # CSS (dark/light theme) + JS (SPA-like frontend)
│   └── verification/              # Quality assurance
│       ├── ConsistencyChecker     # Label consistency, conflict detection, schema validation
│       ├── AdversarialChecker     # Prompt injection, contradiction, hallucination detection
│       ├── RepairEngine           # Automated issue repair
│       └── QualityVerifier        # Orchestrates check + repair
├── tests/
│   ├── unit/
│   │   └── test_core.py           # 83 unit tests across all modules
│   ├── integration/
│   │   └── test_ingestion.py      # 5 integration tests for ingestion pipeline
│   ├── e2e/                       # End-to-end tests (stub)
│   └── benchmarks/                # Performance benchmarks (stub)
├── .github/workflows/
│   └── ci.yml                     # GitHub Actions CI
├── conftest.py                    # Pytest fixtures
├── pytest.ini                     # Pytest config
├── pyproject.toml                 # Project metadata and dependencies
└── README.md                      # Project overview
```

---

## 4. Quick Start

### Prerequisites

- Python 3.11+
- pip
- (Optional) Docker and Docker Compose for full stack

### Installation

```bash
# Clone the repository
git clone https://github.com/elonmasai7/Qwen3_4B_Anotation.git
cd annotation-platform

# Install dependencies
pip install -e .

# Or with dev dependencies (testing, linting)
pip install -e ".[dev]"
```

### Running the API Server

```bash
# Start the FastAPI server
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Or directly
python -m src.main
```

The server starts at `http://localhost:8000`. API docs at `http://localhost:8000/api/v1/docs`.

### Verifying It Works

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Run an annotation
curl -X POST http://localhost:8000/api/v1/annotate \
  -H "Content-Type: application/json" \
  -d '{"data_id": "test1", "content": "This is a sample document.", "prompt_template": "classify this text"}'

# Upload a dataset
curl -X POST http://localhost:8000/api/v1/datasets/upload \
  -F "file=@data/FINAL_DATASET.csv"
```

### Running with Docker

```bash
docker-compose -f docker/docker-compose.yaml up
```

---

## 5. Configuration

Configuration is managed via `config/base.yaml` with overrides from environment variables (via Pydantic Settings).

### Core Sections

```yaml
platform:
  name: "Annotation Platform"
  version: "1.0.0"
  environment: "development"     # development | staging | production

database:                        # PostgreSQL
  host: localhost
  port: 5432
  name: annotation_db
  pool_size: 20
  max_overflow: 10

redis:                           # Redis cache
  host: localhost
  port: 6379
  db: 0

kafka:                           # Kafka messaging
  bootstrap_servers: localhost:9092
  consumer_group: annotation-platform

minio:                           # MinIO object storage
  endpoint: localhost:9000
  access_key: minioadmin
  secret_key: minioadmin
  bucket: annotation-data

model:                           # LLM configuration
  name: "Qwen/Qwen3-4B"
  max_tokens: 8192
  temperature: 0.1
  top_p: 0.95

vllm:                            # vLLM serving
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.85
  max_num_seqs: 256
  enforce_eager: false

annotation:                      # Annotation pipeline
  num_branches: 3
  confidence_threshold: 0.8
  max_retries: 3

retrieval:                       # Example retrieval
  top_k: 5
  embedding_model: "BAAI/bge-small-en-v1.5"
  similarity_threshold: 0.7

logging:                         # Structured logging
  level: "INFO"
  format: "json"                 # json | standard

monitoring:                      # Prometheus metrics
  enabled: true
  metrics_port: 9090
```

Environment variables override any field using the pattern `SECTION_FIELD` (e.g., `DATABASE_HOST=prod-db.example.com`).

---

## 6. Data Ingestion

### Supported Formats

| Format | Loader | Extension |
|---|---|---|
| JSON | `JSONLoader` | `.json` |
| JSONL | `JSONLLoader` | `.jsonl` |
| CSV | `CSVLoader` | `.csv` |
| TSV | `TSVLoader` | `.tsv` |
| Parquet | `ParquetLoader` | `.parquet` |
| XML | `XMLLoader` | `.xml` |
| YAML | `YAMLLoader` | `.yaml`, `.yml` |

### Using Loaders Programmatically

```python
from pathlib import Path
from src.ingestion.loaders import get_loader, CSVLoader

# Auto-detect loader by file extension
loader = get_loader(Path("data/FINAL_DATASET.csv"))

# Validate the file
valid = await loader.validate()

# Load all rows (async streaming)
async for row in loader.load():
    print(row.id, row.content, row.raw_data)

# Or load in batches
async for batch in loader.stream_load(batch_size=100):
    print(f"Got batch of {len(batch)} rows")
```

### CSVLoader Customization

```python
# Map columns for content and id
loader = CSVLoader(
    file_path=Path("data/FINAL_DATASET.csv"),
    text_column="image_url",    # Column to use as DataRow.content
    id_column="image_id",       # Column to use as DataRow.id
)
```

### Processing Pipeline

```python
from src.ingestion.processors import ContextProcessor, ChunkingConfig, MemoryCompressor
from src.ingestion.normalizers import DataNormalizer
from src.ingestion.validators import DataValidator

normalizer = DataNormalizer()
validator = DataValidator()
chunker = ContextProcessor()

row = await loader.load().__anext__()
normalized = await normalizer.normalize(row)
issues = await validator.validate(normalized)
chunks = await chunker.process(normalized)  # Semantic/hierarchical/recursive/sliding
```

### Chunking Strategies

| Strategy | Description |
|---|---|
| `SEMANTIC` | Sentence-splitting with token budget, embeddings, importance scoring |
| `HIERARCHICAL` | Paragraph-level splitting, recursively splits oversize sections |
| `RECURSIVE` | Recursively halves oversized text at approximate midpoints |
| `SLIDING` | Fixed-size token windows with configurable overlap |

---

## 7. Annotation Pipeline

### Prompt Constructor

Builds ICL prompts from templates, retrieved examples, and context chunks:

```python
from src.annotation.prompt_constructor import PromptConstructor, PromptConfig, DynamicExampleRetriever

constructor = PromptConstructor()
retriever = DynamicExampleRetriever()

template = PromptTemplate(
    name="classification",
    instruction="Classify the following text",
    task_definition="Determine if the text is positive or negative",
    output_schema='{"label": "positive|negative", "confidence": 0.0-1.0}',
    cot_scaffold="Think step by step: 1) Analyze the content 2) Determine sentiment",
    examples=[...],
)

examples = retriever.retrieve_similar(input_text, template.examples, top_k=3)
prompt = constructor.construct(template, input_text, retrieved_chunks=chunks, examples=examples)
```

### Multi-Pass Reasoner

Runs a 5-pass pipeline that iteratively improves annotations:

```python
from src.annotation.reasoner import AnnotationReasoner, MultiPassReasoner, SelfConsistencyEngine

reasoner = AnnotationReasoner()
multi_pass = MultiPassReasoner(reasoner)
consistency = SelfConsistencyEngine(reasoner, num_branches=3)

# Single pass
result = await reasoner.reason(data_row, prompt)

# Multi-pass (draft → self_critique → repair → calibrate → validate)
result = await multi_pass.reason(data_row, prompt)

# Self-consistency (N parallel branches + voting)
result = await consistency.annotate(data_row, prompt, voting_strategy="majority")
```

### Voting Strategies

| Strategy | Description |
|---|---|
| `majority` | Counts label occurrences across branches |
| `weighted` | Sums confidence scores per label |
| `entropy` | Averages confidence per label |
| `confidence` | Squares confidence as weight |

---

## 8. Verification & Quality

```python
from src.verification import QualityVerifier, ConsistencyChecker, AdversarialChecker

verifier = QualityVerifier()

# Full pipeline: adversarial check → repair → consistency check → fix
verified = await verifier.verify_and_repair(data_row, annotation_result)

# Individual checks
consistency = ConsistencyChecker()
result = consistency.verify(annotation, data)
# Checks: label consistency, confidence range, rationale presence, evidence spans, conflicts

adversarial = AdversarialChecker()
result = adversarial.verify(data, annotation)
# Checks: prompt injection, contradictory context, distractors, hallucination
```

---

## 9. Evaluation

```python
from src.evaluation import EvaluationEngine, MetricsCalculator

engine = EvaluationEngine()
metrics = engine.evaluate(predictions, ground_truths, cost=0.05)

# Available metrics
print(f"Accuracy: {metrics.accuracy}")
print(f"Precision: {metrics.precision}")
print(f"Recall: {metrics.recall}")
print(f"F1: {metrics.f1}")
print(f"Macro F1: {metrics.macro_f1}")
print(f"Micro F1: {metrics.micro_f1}")
print(f"Exact Match: {metrics.exact_match}")
print(f"Calibration Error: {metrics.calibration_error}")
print(f"Latency: {metrics.latency_ms}ms")
```

---

## 10. Active Learning

```python
from src.training import ActiveLearningEngine, SyntheticExampleGenerator

engine = ActiveLearningEngine(uncertainty_threshold=0.3)

# Select samples for human review
uncertain = engine.select_uncertain_samples(annotations, data, num_samples=10)
disagreement = engine.select_disagreement_samples(annotations, data, num_samples=10)
weak = engine.select_weak_label_samples(annotations, data, num_samples=10)

# Generate training examples
generator = SyntheticExampleGenerator()
new_examples = generator.generate(existing_examples, num_new=5)
counterexamples = generator.generate_counterexamples(positive_examples, num=3)
```

---

## 11. Agent System

The platform includes 12 autonomous specialist agents coordinated by an `AgentOrchestrator`:

```python
from src.agents import AgentOrchestrator

orchestrator = AgentOrchestrator()

# Run a single agent
result = await orchestrator.run_agent("research", {"query": "optimize ICL prompts"})

# Run a pipeline of agents
results = await orchestrator.run_pipeline([
    ("research", {"query": "best practices for few-shot"}),
    ("prompt", {"task": "optimize prompt for classification"}),
    ("evaluation", {"metrics": {"accuracy": 0.85, "f1": 0.82}}),
])
```

Current agent stubs return mock responses. They will be wired to real LLM inference via the VLLMEngine in production.

---

## 12. API Reference

All API endpoints are mounted at `/api/v1`:

### Annotation

```
POST /api/v1/annotate
```

```json
{
  "data_id": "doc_001",
  "content": "Document text to annotate",
  "prompt_template": "classify this text",
  "metadata": {}
}
```

Response:
```json
{
  "annotation_id": "uuid",
  "status": "completed",
  "labels": [
    {
      "value": "sample_label",
      "confidence": 0.85,
      "rationale": "Analysis based on...",
      "evidence_spans": [[0, 50]]
    }
  ],
  "processing_time_ms": 150.0
}
```

### Datasets

```
POST /api/v1/datasets/upload          # Upload file (multipart)
GET  /api/v1/datasets                 # List datasets
GET  /api/v1/datasets/{dataset_id}    # Get dataset info
```

### Experiments

```
POST /api/v1/experiments              # Create experiment
GET  /api/v1/experiments              # List experiments
GET  /api/v1/experiments/{id}         # Get experiment with metrics
```

### Leaderboard

```
GET  /api/v1/leaderboard              # Get rankings
POST /api/v1/leaderboard/submit       # Submit entry
```

### Prompts

```
GET  /api/v1/prompts                  # List prompts
POST /api/v1/prompts                  # Create prompt
```

### System

```
GET  /api/v1/                         # API root
GET  /api/v1/health                   # Health check
```

Interactive API docs available at `http://localhost:8000/api/v1/docs` (auto-generated by FastAPI).

---

## 13. UI Dashboard

The web UI is available at `http://localhost:8000/ui/` with pages:

| Page | Route | Description |
|---|---|---|
| Dashboard | `/ui/` | Stats overview, recent experiments, leaderboard |
| Annotation | `/ui/annotation` | Submit documents for annotation |
| Datasets | `/ui/datasets` | Upload and manage datasets |
| Experiments | `/ui/experiments` | Create and monitor experiments |
| Prompts | `/ui/prompts` | Manage prompt templates |
| Leaderboard | `/ui/leaderboard` | View rankings and submit entries |

The UI features a dark theme by default with light theme support via `prefers-color-scheme`, responsive design, and keyboard accessibility.

---

## 14. Docker Deployment

### Services

| Service | Image | Purpose |
|---|---|---|
| `api` | Built from `docker/Dockerfile` | FastAPI application |
| `postgres` | `postgres:16` | Primary database |
| `redis` | `redis:7-alpine` | Cache and rate limiting |
| `kafka` | `bitnami/kafka:3.6` | Async annotation task queue |
| `minio` | `minio/minio:latest` | Object storage for datasets/files |
| `prometheus` | `prom/prometheus:latest` | Metrics collection |

### Usage

```bash
# Start all services
docker-compose -f docker/docker-compose.yaml up -d

# View logs
docker-compose -f docker/docker-compose.yaml logs -f api

# Scale the API
docker-compose -f docker/docker-compose.yaml up -d --scale api=3

# Stop all services
docker-compose -f docker/docker-compose.yaml down
```

### Environment Variables

The API service is configured via environment variables that override `config/base.yaml`:

| Variable | Default | Description |
|---|---|---|
| `PLATFORM_ENVIRONMENT` | `development` | Runtime environment |
| `DATABASE_HOST` | `postgres` | PostgreSQL host |
| `REDIS_HOST` | `redis` | Redis host |
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka:9092` | Kafka broker |
| `MINIO_ENDPOINT` | `minio:9000` | MinIO endpoint |

---

## 15. Testing

```bash
# Run all tests
PYTHONPATH=src pytest tests/ -v

# Run only unit tests
PYTHONPATH=src pytest tests/unit/ -v

# Run only integration tests
PYTHONPATH=src pytest tests/integration/ -v

# Run with coverage
PYTHONPATH=src pytest tests/ --cov=src

# Lint check
ruff check src/

# Type check (partial - needs full deps)
mypy src/
```

---

## 16. CI/CD

The project includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs on push/PR to `main`:

- Sets up Python 3.11
- Installs dependencies with dev extras
- Runs `ruff check src/`
- Runs `pytest tests/ -v`
- Verifies the ingestion pipeline with `FINAL_DATASET.csv`
