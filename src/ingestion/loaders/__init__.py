from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncIterator, Callable
import json
import csv
import xml.etree.ElementTree as ET
import pyarrow.parquet as pq
import yaml
from datetime import datetime
import uuid
from common.types import DataRow, DataFormat, Metadata
from common.logging import get_logger

logger = get_logger(__name__)


class DataLoader(ABC):
    def __init__(self, file_path: Path):
        self.file_path = file_path

    @abstractmethod
    async def load(self) -> AsyncIterator[DataRow]:
        pass

    @abstractmethod
    async def validate(self) -> bool:
        pass

    async def stream_load(
        self,
        batch_size: int = 100,
        transform_fn: Callable[[dict], dict[str, Any]] | None = None,
    ) -> AsyncIterator[list[DataRow]]:
        batch: list[DataRow] = []
        async for row in self.load():
            if transform_fn:
                row.raw_data = transform_fn(row.raw_data)
            batch.append(row)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


class JSONLoader(DataLoader):
    async def load(self) -> AsyncIterator[DataRow]:
        with open(self.file_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                yield self._parse_item(item)
        elif isinstance(data, dict):
            yield self._parse_item(data)

    def _parse_item(self, item: dict[str, Any]) -> DataRow:
        return DataRow(
            id=item.get("id", str(uuid.uuid4())),
            content=item.get("content", item.get("text", item.get("data", str(item)))),
            metadata=Metadata(
                source=str(self.file_path),
                created_at=datetime.utcnow(),
            ),
            raw_data=item,
        )

    async def validate(self) -> bool:
        try:
            with open(self.file_path) as f:
                json.load(f)
            return True
        except Exception as e:
            logger.error("json_validation_failed", error=str(e))
            return False


class JSONLLoader(DataLoader):
    async def load(self) -> AsyncIterator[DataRow]:
        with open(self.file_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    yield self._parse_item(item)
                except json.JSONDecodeError as e:
                    logger.warning("jsonl_parse_error", error=str(e), line=line[:100])

    def _parse_item(self, item: dict[str, Any]) -> DataRow:
        return DataRow(
            id=item.get("id", str(uuid.uuid4())),
            content=item.get("content", item.get("text", str(item))),
            metadata=Metadata(
                source=str(self.file_path),
                created_at=datetime.utcnow(),
            ),
            raw_data=item,
        )

    async def validate(self) -> bool:
        try:
            with open(self.file_path) as f:
                for line in f:
                    json.loads(line.strip())
            return True
        except Exception as e:
            logger.error("jsonl_validation_failed", error=str(e))
            return False


class CSVLoader(DataLoader):
    def __init__(self, file_path: Path, text_column: str = "text", id_column: str = "id"):
        super().__init__(file_path)
        self.text_column = text_column
        self.id_column = id_column

    async def load(self) -> AsyncIterator[DataRow]:
        with open(self.file_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield DataRow(
                    id=row.get(self.id_column, str(uuid.uuid4())),
                    content=row.get(self.text_column, ""),
                    metadata=Metadata(
                        source=str(self.file_path),
                        created_at=datetime.utcnow(),
                    ),
                    raw_data=row,
                )

    async def validate(self) -> bool:
        try:
            with open(self.file_path, newline="") as f:
                csv.DictReader(f)
            return True
        except Exception as e:
            logger.error("csv_validation_failed", error=str(e))
            return False


class TSVLoader(CSVLoader):
    async def load(self) -> AsyncIterator[DataRow]:
        with open(self.file_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                yield DataRow(
                    id=row.get(self.id_column, str(uuid.uuid4())),
                    content=row.get(self.text_column, ""),
                    metadata=Metadata(
                        source=str(self.file_path),
                        created_at=datetime.utcnow(),
                    ),
                    raw_data=row,
                )


class ParquetLoader(DataLoader):
    async def load(self) -> AsyncIterator[DataRow]:
        table = pq.read_table(self.file_path)
        df = table.to_pandas()
        for _, row in df.iterrows():
            yield DataRow(
                id=str(row.get("id", str(uuid.uuid4()))),
                content=str(row.get("content", row.get("text", str(row)))),
                metadata=Metadata(
                    source=str(self.file_path),
                    created_at=datetime.utcnow(),
                ),
                raw_data=row.to_dict(),
            )

    async def validate(self) -> bool:
        try:
            pq.read_table(self.file_path)
            return True
        except Exception as e:
            logger.error("parquet_validation_failed", error=str(e))
            return False


class XMLLoader(DataLoader):
    async def load(self) -> AsyncIterator[DataRow]:
        tree = ET.parse(self.file_path)
        root = tree.getroot()
        for elem in root.findall(".//record"):
            content = "".join(elem.itertext())
            yield DataRow(
                id=elem.get("id", str(uuid.uuid4())),
                content=content,
                metadata=Metadata(
                    source=str(self.file_path),
                    created_at=datetime.utcnow(),
                ),
                raw_data=self._elem_to_dict(elem),
            )

    def _elem_to_dict(self, elem: ET.Element) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if elem.attrib:
            result.update(elem.attrib)
        for child in elem:
            child_data = self._elem_to_dict(child)
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
        return result

    async def validate(self) -> bool:
        try:
            ET.parse(self.file_path)
            return True
        except Exception as e:
            logger.error("xml_validation_failed", error=str(e))
            return False


class YAMLLoader(DataLoader):
    async def load(self) -> AsyncIterator[DataRow]:
        with open(self.file_path) as f:
            data = yaml.safe_load(f)
        if isinstance(data, list):
            for item in data:
                yield self._parse_item(item)
        elif isinstance(data, dict):
            yield self._parse_item(data)

    def _parse_item(self, item: dict[str, Any]) -> DataRow:
        return DataRow(
            id=item.get("id", str(uuid.uuid4())),
            content=item.get("content", item.get("text", str(item))),
            metadata=Metadata(
                source=str(self.file_path),
                created_at=datetime.utcnow(),
            ),
            raw_data=item,
        )

    async def validate(self) -> bool:
        try:
            with open(self.file_path) as f:
                yaml.safe_load(f)
            return True
        except Exception as e:
            logger.error("yaml_validation_failed", error=str(e))
            return False


def get_loader(file_path: Path) -> DataLoader:
    suffix = file_path.suffix.lower()
    loaders = {
        ".json": JSONLoader,
        ".jsonl": JSONLLoader,
        ".csv": CSVLoader,
        ".tsv": TSVLoader,
        ".parquet": ParquetLoader,
        ".xml": XMLLoader,
        ".yaml": YAMLLoader,
        ".yml": YAMLLoader,
    }
    loader_class = loaders.get(suffix)
    if not loader_class:
        raise ValueError(f"Unsupported file format: {suffix}")
    return loader_class(file_path)