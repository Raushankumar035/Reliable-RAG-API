"""Index everything in data/sample_documents without starting the API.

    python -m scripts.ingest_samples
"""
from app.config import get_settings
from app.dependencies import get_ingestion_service

if __name__ == "__main__":
    for r in get_ingestion_service().ingest_directory(get_settings().sample_documents_dir):
        print(f"{r.status:<18} {r.file_name:<28} chunks={r.chunks} {r.error or ''}")
