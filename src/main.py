from contextlib import asynccontextmanager
from fastapi import FastAPI
from .common.config import get_settings
from .common.logging import setup_logging, get_logger
from .backend.api import app as api_app
from .backend.cache import cache_manager
from .backend.messaging import publisher

settings = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings.info("starting_annotation_platform")

    try:
        await cache_manager.connect()
    except Exception as e:
        settings.warning("cache_connection_failed", error=str(e))

    try:
        await publisher.connect()
    except Exception as e:
        settings.warning("publisher_connection_failed", error=str(e))

    yield

    try:
        await cache_manager.disconnect()
    except Exception:
        pass

    try:
        await publisher.disconnect()
    except Exception:
        pass

    settings.info("shutdown_complete")


app = FastAPI(
    title="Annotation Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/api/v1", api_app)


@app.get("/")
async def root():
    return {
        "name": "Annotation Platform",
        "version": "1.0.0",
        "status": "running",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)