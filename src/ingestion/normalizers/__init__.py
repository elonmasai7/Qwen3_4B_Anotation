from common.types import DataRow
from common.logging import get_logger

logger = get_logger(__name__)


class DataNormalizer:
    async def normalize(self, row: DataRow) -> DataRow:
        return row

    async def normalize_batch(self, rows: list[DataRow]) -> list[DataRow]:
        return rows
