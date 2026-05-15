"""Run async queue worker for annotation jobs."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.core.job_queue import AnnotationJobQueue
from app.db.database import init_db


async def _main(poll_interval_s: float) -> None:
    init_db()
    queue = AnnotationJobQueue()
    await queue.run_worker_loop(poll_interval_s=poll_interval_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-interval", type=float, default=settings.queue_poll_interval_s)
    args = parser.parse_args()

    asyncio.run(_main(args.poll_interval))


if __name__ == "__main__":
    main()
