"""End-to-end multi-agent annotation pipeline."""

from __future__ import annotations

from time import perf_counter

from app.config import settings
from app.core.annotation_agent import AnnotationAgent
from app.core.chunker import Chunker
from app.core.confidence_engine import ConfidenceEngine
from app.core.memory_engine import MemoryEngine
from app.core.prompt_builder import PromptBuilder
from app.core.repair_agent import RepairAgent
from app.core.retrieval_engine import RetrievalEngine
from app.core.verifier_agent import VerifierAgent


class AnnotationPipeline:
    def __init__(self) -> None:
        self.chunker = Chunker(settings.max_chunk_tokens, settings.chunk_overlap_tokens)
        self.memory = MemoryEngine()
        self.retrieval = RetrievalEngine()
        self.prompt_builder = PromptBuilder()
        self.annotation_agent = AnnotationAgent()
        self.verifier = VerifierAgent()
        self.repair = RepairAgent()
        self.confidence = ConfidenceEngine()

    def _vote(self, candidates: list[dict]) -> dict:
        label_votes: dict[str, float] = {}
        for cand in candidates:
            label = str(cand.get("label", "unknown"))
            label_votes[label] = label_votes.get(label, 0.0) + float(cand.get("confidence", 0.0) or 0.0)
        winner = max(label_votes, key=label_votes.get)
        best = max(candidates, key=lambda c: (c.get("label") == winner, c.get("confidence", 0.0)))
        best["votes"] = label_votes
        best["final_label"] = winner
        best["winning_reason"] = "confidence-weighted voting with evidence checks"
        return best

    def annotate_sample(self, sample: dict, example_pool: list[dict]) -> tuple[dict, float]:
        start = perf_counter()
        chunks = self.chunker.chunk(sample["id"], sample["input_text"])
        memory = self.memory.build(sample["input_text"], chunks)
        retrieval = self.retrieval.select_examples(sample["input_text"], example_pool, top_k=settings.retrieval_top_k)
        evidence = sorted(chunks, key=lambda c: c["importance_score"], reverse=True)[:4]

        branches: list[dict] = []
        for _ in range(3):
            prompt = self.prompt_builder.build_annotation_prompt(
                task_instruction=sample.get("task_instruction", "Annotate the sample"),
                label_schema=sample.get("label_options", []),
                examples=retrieval["selected_examples"],
                memory=memory,
                evidence_chunks=evidence,
            )
            first = self.annotation_agent.annotate(prompt)
            check = self.verifier.verify(first, sample.get("label_options", []))
            fixed = self.repair.repair(first, check["issues"], sample.get("label_options", [])) if not check["valid"] else first
            calibrated = self.confidence.calibrate(fixed, check)
            fixed["confidence"] = round(calibrated, 4)
            branches.append(fixed)

        winner = self._vote(branches)
        output = {
            "id": sample["id"],
            "label": winner.get("final_label", winner.get("label", "unknown")),
            "confidence": round(float(winner.get("confidence", 0.0) or 0.0), 4),
            "evidence": winner.get("evidence", []),
            "rationale": winner.get("rationale", ""),
            "uncertainty_reason": winner.get("uncertainty_reason", ""),
            "votes": winner.get("votes", {}),
            "winning_reason": winner.get("winning_reason", ""),
        }
        return output, perf_counter() - start
