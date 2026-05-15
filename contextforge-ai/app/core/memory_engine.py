"""Hierarchical memory construction for long documents."""

from __future__ import annotations

from collections import Counter


class MemoryEngine:
    def build(self, text: str, chunks: list[dict]) -> dict:
        sorted_chunks = sorted(chunks, key=lambda c: c.get("importance_score", 0.0), reverse=True)
        top_chunks = sorted_chunks[: min(5, len(sorted_chunks))]

        global_summary = " ".join(c["text"][:220] for c in top_chunks[:2]).strip()
        section_summaries = [c["text"][:180].strip() for c in sorted_chunks[: min(8, len(sorted_chunks))]]

        token_counter: Counter[str] = Counter()
        for c in sorted_chunks[:10]:
            for token in c["text"].split():
                t = token.strip(".,:;!?()[]{}\"'").lower()
                if len(t) > 4 and t.isalpha():
                    token_counter[t] += 1

        key_evidence = [tok for tok, _ in token_counter.most_common(12)]
        contradictions = [
            c["text"][:160]
            for c in sorted_chunks
            if "however" in c["text"].lower() or "but" in c["text"].lower()
        ][:5]

        return {
            "global_summary": global_summary,
            "section_summaries": section_summaries,
            "key_evidence": key_evidence,
            "contradictions": contradictions,
            "high_priority_chunks": [c["chunk_id"] for c in top_chunks],
        }
