import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import get_settings
from app.core.exceptions import AppError

logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if settings.ingest_samples_on_startup:
        try:
            from app.dependencies import get_ingestion_service

            results = get_ingestion_service().ingest_directory(settings.sample_documents_dir)
            logger.info("Sample documents ready: %s", [(r.file_name, r.status) for r in results])
        except Exception:  # never block startup on demo data
            logger.exception("Failed to ingest sample documents on startup")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Reliable RAG API",
        description="Grounded question answering over your documents (FastAPI + LlamaIndex + ChromaDB + Groq).",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        logger.info(
            "%s %s -> %s (%.0f ms)", request.method, request.url.path, response.status_code,
            (time.perf_counter() - start) * 1000,
        )
        return response

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.code, "message": exc.message})

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception):
        logger.exception("Unhandled error")
        return JSONResponse(
            status_code=500, content={"error": "internal_error", "message": "An unexpected error occurred."}
        )

    app.include_router(router)
    return app


app = create_app()
