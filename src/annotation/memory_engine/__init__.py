from common.types import Chunk
from common.logging import get_logger

logger = get_logger(__name__)


class MemoryEngine:
    async def process(self, context: str) -> list[Chunk]:
        return []

    async def compress(self, context: str, ratio: float = 0.3) -> str:
        return context
