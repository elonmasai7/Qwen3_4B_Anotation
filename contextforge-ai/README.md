# ContextForge AI

Autonomous long-context annotation intelligence for human-level dataset generation.

## Project overview

ContextForge AI is a production-oriented annotation platform for ultra-long-context labeling with Qwen3-4B and In-Context Learning (ICL). It covers ingestion, schema normalization, token-aware chunking, memory construction, adaptive retrieval, prompt rendering, multi-agent validation/repair, confidence calibration, evaluation, and submission packaging.

## Problem statement

High-quality annotation for long documents is expensive, slow, and inconsistent when done manually. Standard ICL pipelines often degrade on long context because relevant evidence is fragmented, retrieval becomes noisy, and weak output checks allow invalid labels.

## Solution architecture

Core pipeline:

1. Dataset ingestion and schema detection.
2. Token-aware long-context chunking with overlap.
3. Hierarchical memory generation (summary, evidence, contradiction cues).
4. Hybrid retrieval for adaptive examples:
   - persisted FAISS index lifecycle (build/load/search)
   - optional pgvector backend for ANN in PostgreSQL
5. Prompt construction via Jinja2 templates.
6. Qwen3-4B inference (Transformers, optional vLLM; deterministic local fallback).
7. Multi-agent annotation, verification, repair, and confidence calibration.
8. Self-consistency voting across multiple branches.
9. Evaluation and submission export.

Queue mode for higher throughput:

- enqueue jobs through API (`POST /annotation/enqueue`)
- run async worker (`scripts/run_worker.py`)
- fetch status/results (`GET /annotation/job/{job_id}`)

## Architecture flow (pipeline walkthrough)

1. `DatasetLoader` reads JSON/JSONL/CSV/TSV/Parquet and normalizes rows into the internal sample schema.
2. `SchemaDetector` maps source columns to `id`, `input_text`, optional task text, and optional label options.
3. `Chunker` splits each sample into overlapping chunks and assigns importance scores.
4. `MemoryEngine` extracts high-priority summaries/evidence signals from chunks.
5. `RetrievalEngine` selects few-shot examples using BM25 + TF-IDF + vector similarity (FAISS or pgvector).
6. `PromptBuilder` renders the annotation prompt from template + memory + examples + evidence chunks.
7. `AnnotationAgent` calls `QwenInference` for model output; `VerifierAgent` checks schema/quality; `RepairAgent` patches invalid outputs.
8. `ConfidenceEngine` calibrates confidence; pipeline performs confidence-weighted voting across three branches.
9. `Exporter` writes predictions, and `Evaluator` generates report artifacts.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional containerized run:

```bash
docker compose up --build
```

## Dataset format

Each normalized sample:

```json
{
  "id": "sample_id",
  "input_text": "long document text",
  "task_instruction": "annotation task",
  "label_options": ["label_a", "label_b"],
  "metadata": {}
}
```

Supported source formats: JSON, JSONL, CSV, TSV, Parquet.

## Running annotation

Synchronous CLI mode:

```bash
python scripts/run_annotation.py --input data/raw/test.jsonl --output data/predictions/preds.jsonl
```

Queue-based async mode:

```bash
# terminal 1: worker loop
python scripts/run_worker.py

# terminal 2: enqueue and wait for completion
python scripts/run_annotation_async.py --input data/raw/test.jsonl --output data/predictions/preds.jsonl
```

## Running evaluation

```bash
python scripts/run_evaluation.py --pred data/predictions/preds.jsonl --gold data/raw/gold.jsonl
```

Reports are saved to:

- `data/reports/evaluation_report.json`
- `data/reports/evaluation_report.md`

## Dashboard usage

```bash
streamlit run dashboard/streamlit_app.py
```

Pages:

- Overview
- Dataset Explorer
- Prompt Lab
- Evaluation
- Submission

## API mode

Start API server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Run batch annotation through API:

```bash
curl -X POST http://127.0.0.1:8000/annotation/run \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "data/raw/test.jsonl",
    "output_path": "data/predictions/preds.jsonl",
    "max_samples": 10
  }'
```

Async queue through API:

```bash
curl -X POST http://127.0.0.1:8000/annotation/enqueue \
  -H "Content-Type: application/json" \
  -d '{
    "sample": {
      "id": "sample_001",
      "input_text": "Document text",
      "task_instruction": "Classify",
      "label_options": ["positive", "negative"],
      "metadata": {}
    },
    "example_pool": []
  }'

curl http://127.0.0.1:8000/annotation/job/<job_id>
```

## Experiment configuration

Use the ablation config and runner:

```bash
python scripts/run_ablation.py --config experiments/configs/ablation.yaml
```

## Retrieval backends

- `RETRIEVAL_BACKEND=faiss` (default): uses `FaissIndexStore` persisted under `FAISS_INDEX_DIR` (default `data/processed/faiss`).
- `RETRIEVAL_BACKEND=pgvector`: uses PostgreSQL + pgvector via `PGVectorStore` table `PGVECTOR_TABLE`.

FAISS persistence notes:

- Index artifacts are persisted as `index.faiss` (when `faiss` is installed), `metadata.json`, `vectors.npy`, and `vectorizer.pkl`.
- Retrieval engine auto-loads existing artifacts on startup and refreshes them when example pool size changes.

pgvector prerequisites and config:

1. Run PostgreSQL with pgvector extension available.
2. Set `DATABASE_URL` to a PostgreSQL DSN.
3. Set `RETRIEVAL_BACKEND=pgvector` and optionally `PGVECTOR_TABLE`.
4. First upsert auto-bootstraps extension/table/index from application code.

Example pgvector `DATABASE_URL`:

```bash
DATABASE_URL=postgresql+psycopg://contextforge:contextforge@localhost:5432/contextforge
```

## One-command workflow (Makefile)

```bash
make setup          # create venv + install deps
make test           # run pytest tests -q
make annotate       # run scripts/run_annotation.py
make worker         # run queue worker loop
make annotate-async # run scripts/run_annotation_async.py
make evaluate       # run scripts/run_evaluation.py
make ablation       # run scripts/run_ablation.py
make submit         # run scripts/build_submission.py
make api            # run uvicorn app.main:app
make dashboard      # run streamlit dashboard
make all            # test + annotate + evaluate + submit
```

## Submission generation

```bash
python scripts/build_submission.py
```

Output bundle:

```text
submission/
|- predictions.jsonl
|- config.json
|- technical_report.md
`- source_code.zip
```

## Codebase guide

### Application entry and API routers

- `app/main.py`: FastAPI app initialization, router registration, startup DB initialization, and `/health` endpoint.
- `app/api/routes_annotation.py`: sync batch annotation, single-sample annotation, async queue enqueue/status endpoints.
- `app/api/routes_dataset.py`: dataset preview/load endpoint.
- `app/api/routes_evaluation.py`: evaluation endpoint that writes report artifacts.
- `app/api/routes_experiments.py`: experiment log/list endpoints backed by JSONL file.

### Core pipeline modules (`app/core/*.py`)

- `dataset_loader.py`: reads heterogeneous file formats and normalizes records.
- `schema_detector.py`: field mapping heuristics for id/text/task/labels.
- `chunker.py`: overlap-based segmentation with lightweight importance scoring.
- `memory_engine.py`: extracts top summaries, evidence tokens, contradiction snippets.
- `retrieval_engine.py`: hybrid retrieval and backend handoff logic (FAISS/pgvector).
- `prompt_builder.py`: Jinja2 prompt rendering from pipeline context.
- `qwen_inference.py`: model backend wrapper (vLLM/Transformers/mock fallback).
- `annotation_agent.py`: primary model invocation agent.
- `verifier_agent.py`: schema and quality checks.
- `repair_agent.py`: deterministic patching for invalid/low-quality outputs.
- `confidence_engine.py`: confidence calibration using evidence bonus and issue penalties.
- `annotation_pipeline.py`: orchestrates end-to-end inference, 3-branch voting, final output.
- `evaluator.py`: classification metrics and report serialization (JSON + Markdown).
- `exporter.py`: prediction JSONL export and submission bundle creation.
- `vector_store.py`: persisted local vector index implementation.
- `pgvector_store.py`: PostgreSQL vector storage and ANN search.
- `job_queue.py`: DB-backed async job lifecycle and worker loop.

### Data and persistence layer

- `app/db/database.py`: SQLAlchemy engine/session setup and `init_db()`.
- `app/db/models.py`: ORM models for datasets, annotations, experiments, and async jobs.

### Scripts and operations

- `scripts/run_annotation.py`: synchronous annotation CLI runner.
- `scripts/run_annotation_async.py`: queue-submit + wait CLI runner.
- `scripts/run_worker.py`: async worker process.
- `scripts/run_evaluation.py`: metrics computation and report generation.
- `scripts/run_ablation.py`: variant sweep runner from YAML config.
- `scripts/build_submission.py`: packaging utility for leaderboard submission.

### Dashboard and tests

- `dashboard/streamlit_app.py`: operator UI for predictions, evaluation, and bundle checks.
- `tests/*`: unit and integration tests for API, pipeline, retrieval, queue, exporter, evaluator, DB/utils, and edge cases.

## Practical developer workflows

1. Sync mode (single process): run `python scripts/run_annotation.py` then `python scripts/run_evaluation.py`.
2. Async queue mode (two terminals): run `python scripts/run_worker.py`, then `python scripts/run_annotation_async.py`.
3. API mode: run `uvicorn app.main:app --reload` and call `/annotation/run` or `/annotation/enqueue`.
4. Dashboard mode: run `streamlit run dashboard/streamlit_app.py` after generating predictions/report files.
5. Makefile mode: use `make all` for quick validation or individual targets (`make annotate`, `make evaluate`, etc.).

## Troubleshooting

- `ModuleNotFoundError: app...`: run commands from repository root (`contextforge-ai`) or use `make` targets that set `PYTHONPATH=.`.
- No predictions generated: verify input path exists and contains non-empty `input_text` after normalization.
- Async run hangs: ensure `scripts/run_worker.py` is running and DB file/URL is the same for worker and client.
- Retrieval quality is poor on first run: build example pool with enough samples; FAISS ranking improves as pool grows.
- `pgvector` backend not active: check `DATABASE_URL` starts with `postgresql`, extension installation permissions, and `RETRIEVAL_BACKEND=pgvector`.
- Evaluation report missing: run `python scripts/run_evaluation.py` and verify `data/reports/` write permissions.

## Contribution guide

How to add a new module:

1. Add implementation under the correct package (`app/core`, `app/api`, `app/utils`, etc.).
2. Wire dependencies in the orchestrator (`app/core/annotation_pipeline.py`) or router (`app/main.py`) as needed.
3. Add/update config knobs in `app/config.py` if runtime behavior must be tunable.
4. Update this README Codebase Guide when module responsibilities change.

How to add tests:

1. Add a `tests/test_<feature>.py` file.
2. Prefer focused unit tests for new logic and API tests for new endpoints.
3. Reuse existing patterns in `tests/test_pipeline.py`, `tests/test_api.py`, and `tests/test_retrieval.py`.
4. Run `make test` before submitting changes.

## Technical innovation

1. Hierarchical Cognitive Memory
2. Adaptive ICL Demonstration Selection
3. Multi-Agent Annotation Reasoning
4. Self-Consistency Voting
5. Confidence Calibration
6. Leaderboard Optimization Loop

## Security notes

- Validate file paths and file types before loading datasets.
- Keep secrets in environment variables only.
- Do not log raw sensitive content.
- Separate system instructions from user dataset text.

## Limitations

- Local fallback inference uses deterministic mock output when model backends are unavailable.
- Semantic memory and contradiction detection are lightweight heuristics in this baseline.
- For large-scale deployment, PostgreSQL + pgvector is recommended with managed index tuning.

## Future work

- Integrate stronger reranking and contradiction-aware retrieval.
- Add online confidence calibration with isotonic regression.
- Add distributed worker orchestration for high-throughput GPU clusters.
