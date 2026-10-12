"""Orchestrates load -> chunk -> embed -> store. Ingestion is idempotent per file name."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from app.config import Settings
from app.core.exceptions import AppError, FileTooLargeError
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import SUPPORTED_EXTENSIONS, content_hash, load_document, safe_filename
from app.retrieval.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    file_name: str
    status: str  # ingested | replaced | skipped_unchanged | failed
    chunks: int = 0
    error: str | None = None
    error_code: str | None = None


class IngestionService:
    def __init__(self, store: VectorStoreManager, settings: Settings):
        self._store = store
        self._settings = settings

    def ingest_bytes(self, filename: str, data: bytes, force: bool = False) -> IngestionResult:
        """Ingest one file. Re-uploading identical content is a no-op; changed content replaces old chunks."""
        file_name = safe_filename(filename)
        max_bytes = self._settings.max_upload_mb * 1024 * 1024
        if len(data) > max_bytes:
            raise FileTooLargeError(f"'{file_name}' exceeds the {self._settings.max_upload_mb} MB limit.")

        existing = self._store.get_document(file_name)
        if existing and not force and existing.content_hash == content_hash(data):
            logger.info("Skipping '%s' (unchanged)", file_name)
            return IngestionResult(file_name, "skipped_unchanged", existing.chunks)

        documents = load_document(file_name, data)  # raises on bad input *before* we touch the store
        nodes = chunk_documents(documents, self._settings.chunk_size, self._settings.chunk_overlap)

        if existing:
            self._store.delete_document(file_name)
        self._store.add_nodes(nodes)
        status = "replaced" if existing else "ingested"
        logger.info("%s '%s' -> %d chunks", status.capitalize(), file_name, len(nodes))
        return IngestionResult(file_name, status, len(nodes))

    def ingest_safely(self, filename: str, data: bytes, force: bool = False) -> IngestionResult:
        """Like ``ingest_bytes`` but converts expected errors into a ``failed`` result."""
        try:
            return self.ingest_bytes(filename, data, force=force)
        except AppError as exc:
            return IngestionResult(safe_filename(filename), "failed", error=exc.message, error_code=exc.code)

    def ingest_directory(self, directory: str) -> List[IngestionResult]:
        results = []
        for path in sorted(Path(directory).glob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                results.append(self.ingest_safely(path.name, path.read_bytes()))
        return results

    def delete(self, file_name: str) -> int:
        return self._store.delete_document(safe_filename(file_name))
