from app.db.database import init_db
from app.utils.logging import get_logger
from app.utils.metrics import RunMetrics
from app.utils.text_cleaning import normalize_text, safe_excerpt


def test_db_init_and_utils() -> None:
    init_db()

    logger = get_logger("contextforge-test")
    logger.info("log smoke test")

    m = RunMetrics()
    m.finish(10)
    assert m.latency >= 0.0
    assert m.throughput >= 0.0

    assert normalize_text("Hello   world\n\n\nTest") == "Hello world\n\nTest"
    assert safe_excerpt("abcdef", 1, 4) == "bcd"
