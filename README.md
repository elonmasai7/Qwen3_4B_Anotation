# Qwen3-4B Automatic Data Annotation Platform

A production-grade automatic data annotation platform optimized for **Qwen3-4B**, designed for long-context scenarios with ICL-based annotation, quality verification, and leaderboard-ready submissions.

**Comprehensive documentation:** [`docs/GUIDE.md`](docs/GUIDE.md)

---

## Quick Start

```bash
# Install
pip install -e .

# Run API server
uvicorn src.main:app --host 0.0.0.0 --port 8000

# Or with Docker
docker-compose -f docker/docker-compose.yaml up
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   UI Dashboard                       │
│  Overview │ Annotation │ Datasets │ Experiments      │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│                FastAPI Backend                       │
│  REST API │ Cache │ Messaging │ Storage │ ORM        │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│           Annotation Intelligence Engine             │
│  Prompt Constructor → Multi-Pass Reasoner →          │
│  Self-Consistency → Quality Verifier → Eval Engine  │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│              Qwen3-4B Inference (vLLM)              │
│  Quantization │ KV Cache │ GPU Scheduling           │
└─────────────────────────────────────────────────────┘
```

---

## Key Features

| Feature | Description |
|---|---|
| **Multi-Format Ingestion** | JSON, JSONL, CSV, TSV, Parquet, XML, YAML with async streaming |
| **Long-Context Processing** | 100k+ tokens with 4 chunking strategies + memory compression |
| **ICL Prompt Construction** | Dynamic example retrieval (BM25, semantic, hybrid) |
| **Multi-Pass Reasoner** | Draft → Self-Critique → Repair → Calibration → Schema Validation |
| **Self-Consistency** | N-branch inference with 4 voting strategies |
| **Auto Verification** | Adversarial detection + consistency checks + auto-repair |
| **Active Learning** | Uncertainty, disagreement, and weak label sampling |
| **Evaluation Suite** | Accuracy, precision, recall, F1, macro/micro F1, calibration error |
| **Prompt Evolution** | Genetic algorithm for automatic prompt optimization |
| **12 Specialist Agents** | Research, Prompt, Retrieval, Evaluation, Security, MLOps, etc. |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/annotate` | Submit document for annotation |
| `POST` | `/api/v1/datasets/upload` | Upload dataset file |
| `GET` | `/api/v1/datasets` | List datasets |
| `GET` | `/api/v1/datasets/{id}` | Get dataset details |
| `POST` | `/api/v1/experiments` | Create experiment |
| `GET` | `/api/v1/experiments` | List experiments |
| `GET` | `/api/v1/experiments/{id}` | Get experiment with metrics |
| `GET` | `/api/v1/leaderboard` | View leaderboard |
| `POST` | `/api/v1/leaderboard/submit` | Submit to leaderboard |
| `GET` | `/api/v1/prompts` | List prompt templates |
| `POST` | `/api/v1/prompts` | Create prompt template |

Interactive docs at `http://localhost:8000/api/v1/docs`.

---

## Sample Dataset

The repo includes `data/FINAL_DATASET.csv` — a face anti-spoofing dataset with 6,557 rows:

- **Labels:** REAL (2,790) / FAKE (3,767)
- **Splits:** train (4,593) / test (1,323) / val (641)
- **Columns:** image_id, image_url, label, category, gender, age_group, source, fake_method, image_quality, resolution, confidence_score, detection_difficulty, dataset_split

Test ingestion:
```python
from pathlib import Path
from src.ingestion.loaders import CSVLoader

loader = CSVLoader(Path("data/FINAL_DATASET.csv"), text_column="image_url")
rows = [r async for r in loader.load()]
print(f"Loaded {len(rows)} rows")
```

---

## Project Structure

```
src/
├── agents/           # 12 autonomous specialist agents
├── annotation/       # Core annotation engine (prompts, reasoner, self-consistency, memory)
├── backend/          # FastAPI services (api, cache, db, messaging, storage)
├── common/           # Shared types, config, logging
├── evaluation/       # Metrics and benchmarks
├── ingestion/        # Loaders, processors, normalizers, validators
├── optimization/     # vLLM, quantization, KV cache, GPU scheduling
├── training/         # Active learning, prompt evolution
├── ui/               # Web dashboard (Jinja2 + CSS + JS)
└── verification/     # Consistency, adversarial checks, repair
```

---

## Configuration

Edit `config/base.yaml` to configure model paths, inference parameters, storage connections, and annotation settings. All fields are overridable via environment variables.

```bash
# Override with env vars
DATABASE_HOST=prod-db.example.com REDIS_HOST=prod-redis.example.com uvicorn src.main:app
```

---

## Testing

```bash
# All tests (83 unit + 5 integration)
PYTHONPATH=src pytest tests/ -v

# Lint
ruff check src/

# Coverage
PYTHONPATH=src pytest tests/ --cov=src
```

---

## Documentation

See [`docs/GUIDE.md`](docs/GUIDE.md) for the complete guide covering:
- Architecture deep dive
- Module-by-module reference
- API usage examples
- Docker deployment
- CI/CD pipeline
- All code snippets

---

## License

MIT
