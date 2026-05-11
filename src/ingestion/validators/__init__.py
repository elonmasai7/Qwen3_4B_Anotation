from common.types import DataRow
from common.logging import get_logger

logger = get_logger(__name__)


class DataValidator:
    async def validate(self, row: DataRow) -> list[str]:
        issues: list[str] = []
        if not row.content:
            issues.append("Empty content")
        if not row.id:
            issues.append("Missing id")
        return issues

    async def validate_batch(self, rows: list[DataRow]) -> list[list[str]]:
        return [await self.validate(r) for r in rows]
