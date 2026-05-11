import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .common.logging import setup_logging, get_logger
from .backend.api import app as api_app
from .backend.cache import cache_manager
from .backend.messaging import publisher
from .ui.routes import router as ui_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("starting_annotation_platform")

    try:
        await cache_manager.connect()
    except Exception as e:
        logger.warning("cache_connection_failed", error=str(e))

    try:
        await publisher.connect()
    except Exception as e:
        logger.warning("publisher_connection_failed", error=str(e))

    yield

    try:
        await cache_manager.disconnect()
    except Exception:
        pass

    try:
        await publisher.disconnect()
    except Exception:
        pass

    logger.info("shutdown_complete")


app = FastAPI(
    title="Annotation Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/api/v1", api_app)
app.mount("/ui/static", StaticFiles(directory=Path(__file__).resolve().parent / "ui" / "static"), name="static")
app.include_router(ui_router, prefix="/ui")


@app.get("/")
async def root():
    return {
        "name": "Annotation Platform",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/docs")
async def api_docs_redirect():
    return {"message": "API docs at /api/v1/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)