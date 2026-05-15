# ContextForge AI

Autonomous long-context annotation intelligence for human-level dataset generation.

## Project overview

ContextForge AI is a production-oriented annotation platform for ultra-long-context labeling with Qwen3-4B and In-Context Learning (ICL). It automates ingestion, chunking, memory construction, adaptive few-shot retrieval, multi-agent reasoning, verification, repair, confidence calibration, evaluation, and submission export.

## Problem statement

High-quality annotation for long documents is expensive, slow, and inconsistent when done manually. Standard ICL pipelines degrade on long context due to noisy examples, attention dilution, and weak output verification.

## Solution architecture

Core pipeline:

1. Dataset ingestion and schema detection
2. Token-aware long-context chunking
3. Hierarchical memory generation
4. Hybrid retrieval for adaptive examples
   - persistent FAISS index lifecycle (build/load/search)
   - optional pgvector backend for large-corpus ANN in PostgreSQL
5. Prompt construction via Jinja2 templates
6. Qwen3-4B inference (Transformers, optional vLLM)
7. Multi-agent annotation, verification, repair, and confidence
8. Self-consistency voting across branches
9. Evaluation and leaderboard export

Queue mode for higher throughput:

- enqueue jobs through API (`/annotation/enqueue`)
- run async worker (`scripts/run_worker.py`)
- fetch status/results (`/annotation/job/{job_id}`)

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

```bash
python scripts/run_annotation.py --input data/raw/test.jsonl --output data/predictions/preds.jsonl
```

Queue-based async mode:

```bash
# terminal 1
python scripts/run_worker.py

# terminal 2
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

## Experiment configuration

Use the ablation config and runner:

```bash
python scripts/run_ablation.py --config experiments/configs/ablation.yaml
```

## Retrieval backends

- `RETRIEVAL_BACKEND=faiss` (default) uses persisted FAISS index under `FAISS_INDEX_DIR`
- `RETRIEVAL_BACKEND=pgvector` uses PostgreSQL + pgvector table `PGVECTOR_TABLE`

Example pgvector `DATABASE_URL`:

```bash
DATABASE_URL=postgresql+psycopg://contextforge:contextforge@localhost:5432/contextforge
```

## One-command workflow (Makefile)

```bash
make setup      # create venv + install deps
make test       # run pytest
make annotate   # generate predictions
make evaluate   # evaluate predictions
make submit     # build submission bundle
make all        # test + annotate + evaluate + submit
```

## Submission generation

```bash
python scripts/build_submission.py
```

Output bundle:

```
submission/
├── predictions.jsonl
├── config.json
├── technical_report.md
└── source_code.zip
```

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

- Local fallback inference uses a deterministic mock output when model backends are unavailable.
- Semantic memory and contradiction detection are lightweight heuristics in this baseline.
- For large-scale deployment, PostgreSQL + FAISS service mode is recommended.

## Future work

- Integrate stronger reranking and contradiction-aware retrieval.
- Add online confidence calibration with isotonic regression.
- Add distributed worker orchestration for high-throughput GPU clusters.
