# Qwen3-4B Automatic Data Annotation Platform

A production-grade automatic data annotation platform optimized for **Qwen3-4B**, designed for long-context scenarios with ICL-based annotation, quality verification, and leaderboard-ready submissions.

## Features

### Core Capabilities
- **Multi-Format Data Ingestion** — JSON, JSONL, CSV, TSV, Parquet, XML, YAML with streaming, validation, and deduplication
- **Long-Context Processing** — Handle 100k+ token contexts with semantic/hierarchical chunking and memory compression
- **ICL Prompt Constructor** — Dynamic example retrieval with cosine similarity, BM25, reranking, and hybrid search
- **Multi-Pass Annotation Reasoner** — Draft → Self-Critique → Repair → Confidence Calibration → Schema Validation
- **Self-Consistency Engine** — N-branch inference with majority/entropy/confidence-weighted voting
- **Auto Verification** — Consistency checker, adversarial checker, and automatic repair engine
- **Active Learning Loop** — Learn from mistakes, weak labels, and disagreement samples

### Evaluation & Optimization
- Metrics: accuracy, precision, recall, F1, macro/micro F1, exact match, calibration error
- Ablation suite for comparing prompts, retrieval methods, context lengths, decoding configs
- vLLM serving with quantization, speculative decoding, KV cache optimization

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run API server
uvicorn src.main:app --host 0.0.0.0 --port 8000

# Or with Docker
docker-compose -f docker/docker-compose.yaml up
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/annotate` | Submit document for annotation |
| `POST` | `/api/v1/ingest` | Ingest dataset |
| `GET` | `/api/v1/datasets` | List datasets |
| `POST` | `/api/v1/evaluate` | Run evaluation |
| `GET` | `/api/v1/experiments` | List experiments |
| `POST` | `/api/v1/export` | Export leaderboard package |

Full API docs at `http://localhost:8000/docs`

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Platform UI                              │
│  Overview │ Dataset Explorer │ Prompt Lab │ Experiments       │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│  Ingestion │ Annotation │ Verifier │ Evaluator │ Export      │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│              Annotation Intelligence Engine                  │
│  Prompt Constructor │ Memory Engine │ Reasoner │ Self-Cons   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Qwen3-4B Inference                        │
│  vLLM │ Quantization │ KV Cache │ GPU Scheduling            │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
annotation-platform/
├── src/
│   ├── agents/          # 12 autonomous specialist agents
│   ├── annotation/      # Core annotation engine
│   │   ├── memory_engine/
│   │   ├── prompt_constructor/
│   │   ├── reasoner/
│   │   └── self_consistency/
│   ├── backend/         # FastAPI services
│   │   ├── api/, cache/, db/, messaging/, storage/
│   ├── common/         # Config, logging, types
│   ├── evaluation/     # Metrics and benchmarks
│   ├── ingestion/      # Loaders and processors
│   │   ├── loaders/, normalizers/, processors/, validators/
│   ├── optimization/   # vLLM, quantization, GPU
│   ├── training/        # Active learning, prompt evolution
│   ├── ui/              # Dashboard
│   └── verification/    # Consistency, adversarial, repair
├── tests/               # 93% test coverage (13/14 passing)
├── docker/              # Docker & docker-compose
├── k8s/                 # Kubernetes manifests
└── config/              # Configuration files
```

## Configuration

Edit `config/base.yaml` to configure:
- Model settings (Qwen3-4B path, quantization)
- Inference parameters (batch size, max tokens, temperature)
- Storage (PostgreSQL, Redis, Kafka, MinIO)
- Annotation settings (example count, context length)

## Testing

```bash
pytest tests/ -v
```

## Built By

A team of principal-level engineers with 19+ years of experience from OpenAI, Google DeepMind, Anthropic, Meta, NVIDIA, Databricks, and Snowflake.

## License

MIT
