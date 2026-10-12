"""Central configuration. Every setting can be overridden via environment variables / .env."""
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App -------------------------------------------------------------
    app_name: str = "Reliable RAG API"
    log_level: str = "INFO"

    # --- LLM (Groq) ------------------------------------------------------
    groq_api_key: Optional[str] = None
    groq_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    llm_max_tokens: int = 4096

    # --- Embeddings ------------------------------------------------------
    # "fastembed" = real local ONNX model (default). "hash" = deterministic offline
    # bag-of-words embedding, used by the test-suite and for quick offline demos.
    embedding_provider: Literal["fastembed", "hash"] = "fastembed"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Vector store (ChromaDB) ----------------------------------------
    chroma_persist_dir: str = "./storage/chroma"
    chroma_collection: str = "documents"

    # --- Chunking --------------------------------------------------------
    chunk_size: int = Field(200, ge=64)
    chunk_overlap: int = Field(30, ge=0)

    # --- Retrieval / answering ------------------------------------------
    top_k: int = Field(5, ge=1)
    max_top_k: int = Field(20, ge=1)
    # Chunks with cosine similarity below this are discarded *before* calling the LLM.
    # If nothing survives, the API answers "unavailable" without spending an LLM call.
    min_similarity: float = Field(0.25, ge=0.0, le=1.0)
    max_question_chars: int = 1000

    # --- Uploads ---------------------------------------------------------
    max_upload_mb: int = 20
    sample_documents_dir: str = "./data/sample_documents"
    ingest_samples_on_startup: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
