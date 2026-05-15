# ContextForge AI Technical Report

## Abstract

ContextForge AI is an autonomous annotation system for ultra-long-context documents built around Qwen3-4B and adaptive ICL. The platform combines hierarchical memory, hybrid retrieval, multi-agent reasoning, self-consistency voting, and confidence calibration to produce leaderboard-ready annotations with reproducible outputs.

## Problem analysis

Long-context annotation fails when examples are noisy, evidence is diluted across long spans, and generated labels are not verified. Human-only annotation does not scale for large benchmark datasets and introduces inconsistency.

## System architecture

The system uses modular components:

- Ingestion: schema detection, normalization, deduplication
- Chunking: token-aware segmentation with overlap
- Memory: global, section, evidence, contradiction memory
- Retrieval: BM25 + vector hybrid example selection
  - FAISS persistence for local reproducible vector retrieval
  - optional pgvector backend for scalable PostgreSQL ANN
- Prompting: strict JSON-output templates
- Inference: Qwen3-4B wrapper (Transformers or vLLM)
- Agents: annotation, verifier, repair, confidence, final judge
- Evaluation: metric computation and report export

## Long-context ICL strategy

ContextForge compresses long documents into layered memory and retrieves high-value evidence chunks plus dynamic demonstrations per sample. This reduces irrelevant context and stabilizes reasoning.

## Example selection method

Hybrid score = alpha * normalized BM25 + beta * normalized vector similarity. Selection applies simple diversity constraints to avoid label collapse and near-duplicate demonstrations.

## Memory design

- Global Memory: top-chunk summary
- Section Memory: section-level condensed spans
- Evidence Memory: salient terms and support cues
- Contradiction Memory: potentially conflicting spans

## Multi-agent reasoning pipeline

Input -> Chunking -> Memory -> Retrieval -> Prompt -> Annotation Agent -> Verifier -> Repair -> Confidence -> Final Vote -> Output

Async mode:

Input -> API enqueue -> DB queue -> Worker process -> Annotation pipeline -> Stored job result

## Verification strategy

The verifier checks JSON validity, schema conformity, missing evidence, and low confidence. Failed predictions are repaired and re-scored.

## Evaluation methodology

Implemented metrics:

- Accuracy
- Precision
- Recall
- Macro F1
- Micro F1
- Exact Match
- Confusion Matrix
- Average Confidence
- Latency per sample
- Throughput

Reports are generated in JSON and Markdown.

## Ablation study

`scripts/run_ablation.py` runs config variants for chunk size, overlap, retrieval method, example count, and temperature. Results are stored in `experiments/results/ablation_results.json`.

## Performance results

This baseline is designed for reproducibility-first local execution. Real leaderboard performance depends on model checkpoint, hardware, retrieval corpus quality, and prompt tuning.

## Reproducibility guide

1. Install dependencies from `requirements.txt`
2. Use fixed configs in `experiments/configs/ablation.yaml`
3. Run annotation and evaluation scripts with explicit file paths
4. Store reports and submission artifacts under versioned outputs

## Winning innovation section

1. Hierarchical Cognitive Memory
2. Adaptive ICL Demonstration Selection
3. Multi-Agent Annotation Reasoning
4. Self-Consistency Voting
5. Confidence Calibration
6. Leaderboard Optimization Loop

## Conclusion

ContextForge AI provides a complete end-to-end blueprint and working implementation for long-context autonomous annotation using Qwen3-4B. The architecture is modular, reproducible, locally runnable, and production-deployable with clear paths for scale and leaderboard optimization.
