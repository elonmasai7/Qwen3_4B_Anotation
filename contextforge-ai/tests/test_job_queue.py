import asyncio

from app.core.job_queue import AnnotationJobQueue
from app.db.database import init_db


def test_job_queue_enqueue_and_process() -> None:
    init_db()
    queue = AnnotationJobQueue()
    sample = {
        "id": "queue_sample_1",
        "input_text": "The evidence strongly supports a positive judgment.",
        "task_instruction": "Classify sentiment",
        "label_options": ["positive", "negative", "neutral"],
    }

    job_id = queue.enqueue(sample)
    for _ in range(20):
        asyncio.run(queue._process_once())
        job = queue.get_job(job_id)
        if job and job["status"] in {"completed", "failed"}:
            break

    job = queue.get_job(job_id)
    assert job is not None
    assert job["status"] in {"completed", "failed"}
