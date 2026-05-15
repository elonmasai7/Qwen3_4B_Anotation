"""Text cleaning helpers for ingestion and prompting."""

from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def safe_excerpt(text: str, start: int, end: int) -> str:
    if not text:
        return ""
    s = max(start, 0)
    e = min(end, len(text))
    return text[s:e]
