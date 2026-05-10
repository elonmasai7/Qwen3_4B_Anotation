from typing import AsyncIterator
import re
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from common.types import Chunk, ChunkStrategy, DataRow
from common.config import get_settings
from common.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class ChunkingConfig:
    max_tokens: int = 512
    overlap_tokens: int = 50
    strategy: ChunkStrategy = ChunkStrategy.SEMANTIC
    min_chunk_size: int = 50


class ContextProcessor:
    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()
        self._embedding_model = None

    @property
    def embedding_model(self):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise RuntimeError("sentence-transformers not installed")
        if self._embedding_model is None:
            logger.info("loading_embedding_model", model=settings.retrieval.embedding_model)
            self._embedding_model = SentenceTransformer(settings.retrieval.embedding_model)
        return self._embedding_model

    async def process(self, data: DataRow) -> list[Chunk]:
        text = data.content
        strategy = self.config.strategy

        if strategy == ChunkStrategy.SEMANTIC:
            return await self._semantic_chunk(text)
        elif strategy == ChunkStrategy.HIERARCHICAL:
            return await self._hierarchical_chunk(text)
        elif strategy == ChunkStrategy.RECURSIVE:
            return await self._recursive_chunk(text)
        elif strategy == ChunkStrategy.SLIDING:
            return await self._sliding_window_chunk(text)
        else:
            return await self._semantic_chunk(text)

    async def _semantic_chunk(self, text: str) -> list[Chunk]:
        sentences = self._split_by_sentences(text)
        chunks: list[Chunk] = []
        current_chunk_text = ""
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self._estimate_tokens(sent)
            if current_tokens + sent_tokens > self.config.max_tokens and current_chunk_text:
                chunks.append(self._create_chunk(current_chunk_text, len(chunks)))
                overlap_text = self._get_overlap_text(current_chunk_text)
                current_chunk_text = overlap_text + sent
                current_tokens = self._estimate_tokens(current_chunk_text)
            else:
                current_chunk_text += " " + sent if current_chunk_text else sent
                current_tokens += sent_tokens

        if current_chunk_text:
            chunks.append(self._create_chunk(current_chunk_text, len(chunks)))

        await self._compute_embeddings(chunks)
        await self._compute_importance(chunks)
        return chunks

    async def _hierarchical_chunk(self, text: str) -> list[Chunk]:
        paragraphs = text.split("\n\n")
        chunks: list[Chunk] = []
        current_section = ""

        for para in paragraphs:
            if self._estimate_tokens(current_section + para) > self.config.max_tokens:
                if current_section:
                    sub_chunks = await self._split_into_smaller_chunks(current_section)
                    chunks.extend(sub_chunks)
                current_section = para
            else:
                current_section += "\n\n" + para if current_section else para

        if current_section:
            sub_chunks = await self._split_into_smaller_chunks(current_section)
            chunks.extend(sub_chunks)

        await self._compute_embeddings(chunks)
        await self._compute_importance(chunks)
        return chunks

    async def _recursive_chunk(self, text: str) -> list[Chunk]:
        return await self._recursive_split(text, self.config.max_tokens)

    async def _recursive_split(self, text: str, max_tokens: int) -> list[Chunk]:
        if self._estimate_tokens(text) <= max_tokens:
            return [self._create_chunk(text, 0)]

        parts = self._split_middle(text)
        if len(parts) == 1:
            return [self._create_chunk(text[: len(text) // 2], 0)]

        chunks: list[Chunk] = []
        for i, part in enumerate(parts):
            sub_chunks = await self._recursive_split(part, max_tokens)
            for sc in sub_chunks:
                sc.chunk_index = len(chunks)
                chunks.append(sc)
        return chunks

    async def _sliding_window_chunk(self, text: str) -> list[Chunk]:
        tokens = text.split()
        window_size = self.config.max_tokens
        step = window_size - self.config.overlap_tokens
        chunks: list[Chunk] = []

        for i in range(0, len(tokens), step):
            window = " ".join(tokens[i : i + window_size])
            if len(window) >= self.config.min_chunk_size:
                chunks.append(self._create_chunk(window, len(chunks)))

        await self._compute_embeddings(chunks)
        await self._compute_importance(chunks)
        return chunks

    async def _split_into_smaller_chunks(self, text: str) -> list[Chunk]:
        sentences = self._split_by_sentences(text)
        chunks: list[Chunk] = []
        current = ""
        idx = 0

        for sent in sentences:
            if self._estimate_tokens(current + sent) > self.config.max_tokens:
                if current:
                    chunks.append(self._create_chunk(current, idx))
                    idx += 1
                current = sent
            else:
                current += " " + sent if current else sent

        if current:
            chunks.append(self._create_chunk(current, idx))

        return chunks

    def _split_by_sentences(self, text: str) -> list[str]:
        sentence_endings = re.compile(r"(?<=[.!?])\s+")
        return [s.strip() for s in sentence_endings.split(text) if s.strip()]

    def _split_middle(self, text: str) -> list[str]:
        midpoint = len(text) // 2
        approx_mid = text.find(" ", midpoint)
        if approx_mid == -1:
            approx_mid = midpoint
        return [text[:approx_mid], text[approx_mid:]]

    def _estimate_tokens(self, text: str) -> int:
        return len(text.split()) * 4 // 3

    def _get_overlap_text(self, text: str) -> str:
        tokens = text.split()
        overlap_tokens = min(self.config.overlap_tokens, len(tokens) // 2)
        return " ".join(tokens[-overlap_tokens:]) if overlap_tokens > 0 else ""

    def _create_chunk(self, text: str, index: int) -> Chunk:
        return Chunk(
            text=text,
            start_idx=0,
            end_idx=len(text),
            chunk_index=index,
        )

    async def _compute_embeddings(self, chunks: list[Chunk]) -> None:
        if not HAS_SENTENCE_TRANSFORMERS:
            return
        try:
            texts = [c.text for c in chunks]
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb.tolist()
        except Exception as e:
            logger.warning("embedding_computation_failed", error=str(e))

    async def _compute_importance(self, chunks: list[Chunk]) -> None:
        if not chunks or not HAS_NUMPY:
            return
        try:
            embeddings = np.array([c.embedding for c in chunks if c.embedding])
            if len(embeddings) == 0:
                return
            mean_emb = np.mean(embeddings, axis=0)
            similarities = np.dot(embeddings, mean_emb) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(mean_emb) + 1e-8
            )
            for chunk, sim in zip(chunks, similarities):
                chunk.importance_score = float(sim)
        except Exception as e:
            logger.warning("importance_computation_failed", error=str(e))


class MemoryCompressor:
    def __init__(self, compression_ratio: float = 0.3):
        self.compression_ratio = compression_ratio

    def compress(self, text: str) -> str:
        lines = text.split("\n")
        if len(lines) <= 3:
            return text

        important_lines = []
        for line in lines:
            if self._is_important(line):
                important_lines.append(line)

        if not important_lines:
            return "\n".join(lines[:3])

        return "\n".join(important_lines[: int(len(lines) * self.compression_ratio)])

    def _is_important(self, line: str) -> bool:
        important_patterns = [
            r"^\s*[-*\d]+\.",
            r"^(header|title|section|chapter):",
            r"(conclusion|summary|result|important|critical)",
        ]
        return any(re.search(p, line.lower()) for p in important_patterns)


class RecursiveSummarizer:
    def __init__(self, max_summary_length: int = 512):
        self.max_summary_length = max_summary_length

    def summarize(self, text: str) -> str:
        if self._estimate_tokens(text) <= self.max_summary_length:
            return text

        paragraphs = text.split("\n\n")
        summaries = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)
            if current_tokens + para_tokens <= self.max_summary_length:
                summaries.append(para[:200])
                current_tokens += para_tokens
            else:
                break

        return "\n\n".join(summaries) if summaries else text[: self.max_summary_length]

    def _estimate_tokens(self, text: str) -> int:
        return len(text.split()) * 4 // 3