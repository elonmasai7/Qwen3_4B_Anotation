import pytest
from pathlib import Path
from src.ingestion.loaders import CSVLoader, get_loader
from src.ingestion.normalizers import DataNormalizer
from src.ingestion.validators import DataValidator


class TestIngestionPipeline:
    @pytest.mark.asyncio
    async def test_load_real_dataset(self):
        dataset_path = Path("data/FINAL_DATASET.csv")
        if not dataset_path.exists():
            pytest.skip("FINAL_DATASET.csv not found")

        loader = CSVLoader(dataset_path, text_column="image_url", id_column="image_id")
        valid = await loader.validate()
        assert valid

        rows = [row async for row in loader.load()]
        assert len(rows) == 6557
        assert all(r.content != "" for r in rows)
        assert all(r.id for r in rows)
        sample = rows[0]
        assert sample.raw_data["label"] in ("REAL", "FAKE")

    @pytest.mark.asyncio
    async def test_loader_factory(self):
        dataset_path = Path("data/FINAL_DATASET.csv")
        if not dataset_path.exists():
            pytest.skip("FINAL_DATASET.csv not found")

        loader = get_loader(dataset_path)
        assert isinstance(loader, CSVLoader)

        valid = await loader.validate()
        assert valid

    @pytest.mark.asyncio
    async def test_stream_load_batching(self):
        dataset_path = Path("data/FINAL_DATASET.csv")
        if not dataset_path.exists():
            pytest.skip("FINAL_DATASET.csv not found")

        loader = CSVLoader(dataset_path, text_column="image_url", id_column="image_id")
        batch_count = 0
        async for batch in loader.stream_load(batch_size=1000):
            batch_count += 1
            assert len(batch) <= 1000
        assert batch_count == 7

    @pytest.mark.asyncio
    async def test_normalize_and_validate_pipeline(self):
        dataset_path = Path("data/FINAL_DATASET.csv")
        if not dataset_path.exists():
            pytest.skip("FINAL_DATASET.csv not found")

        loader = CSVLoader(dataset_path, text_column="image_url", id_column="image_id")
        normalizer = DataNormalizer()
        validator = DataValidator()

        rows = [row async for row in loader.load()]
        batch = rows[:10]

        for row in batch:
            normalized = await normalizer.normalize(row)
            assert normalized.id == row.id

            issues = await validator.validate(normalized)
            assert len(issues) == 0

    def test_data_file_exists(self):
        assert Path("data/FINAL_DATASET.csv").exists(), "Dataset file not found at data/FINAL_DATASET.csv"
