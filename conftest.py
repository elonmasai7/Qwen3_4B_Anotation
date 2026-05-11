import sys
from pathlib import Path

project_root = Path(__file__).parent
src_path = project_root / "src"

sys.path.insert(0, str(src_path))

import pytest
import asyncio


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()