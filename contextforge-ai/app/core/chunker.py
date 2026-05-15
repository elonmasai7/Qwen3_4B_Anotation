"""Long-context chunking engine with overlap and simple importance scoring."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    start_char: int
    end_char: int
    importance_score: float


class Chunker:
    def __init__(self, chunk_size: int = 2048, overlap: int = 256) -> None:
        self.chunk_size = max(chunk_size, 128)
        self.overlap = max(min(overlap, self.chunk_size // 2), 0)

    @staticmethod
    def _token_count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def _importance(text: str) -> float:
        text_l = text.lower()
        bonus = 0.0
        for kw in ("must", "critical", "important", "evidence", "because", "date", "entity"):
            if kw in text_l:
                bonus += 0.08
        heading_bonus = 0.08 if re.search(r"^#{1,3}\s|^[A-Z][A-Za-z0-9 ]{2,40}:", text.strip()) else 0.0
        density = min(len(text.split()) / 400.0, 1.0)
        return round(min(1.0, 0.35 + density * 0.45 + bonus + heading_bonus), 3)

    def chunk(self, document_id: str, text: str) -> list[dict]:
        words = text.split()
        if not words:
            return []

        chunks: list[Chunk] = []
        start_token = 0
        chunk_index = 0
        char_cursor = 0
        while start_token < len(words):
            end_token = min(start_token + self.chunk_size, len(words))
            chunk_words = words[start_token:end_token]
            chunk_text = " ".join(chunk_words)

            start_char = text.find(chunk_words[0], char_cursor) if chunk_words else char_cursor
            if start_char == -1:
                start_char = char_cursor
            end_char = start_char + len(chunk_text)
            char_cursor = max(char_cursor, end_char)

            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index:03d}",
                    document_id=document_id,
                    text=chunk_text,
                    start_char=start_char,
                    end_char=end_char,
                    importance_score=self._importance(chunk_text),
                )
            )

            chunk_index += 1
            if end_token >= len(words):
                break
            start_token = max(0, end_token - self.overlap)

        return [asdict(c) for c in chunks]
