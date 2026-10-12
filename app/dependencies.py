"""Dependency providers (singletons). Tests replace these via ``app.dependency_overrides``."""
from functools import lru_cache

from llama_index.core.base.embeddings.base import BaseEmbedding

from app.config import Settings, get_settings
from app.generation.llm import get_llm
from app.retrieval.embeddings import get_embed_model
from app.retrieval.vector_store import VectorStoreManager
from app.services.ingestion_service import IngestionService
from app.services.qa_service import QAService


@lru_cache
def get_embeddings() -> BaseEmbedding:
    return get_embed_model(get_settings())


@lru_cache
def get_vector_store() -> VectorStoreManager:
    s = get_settings()
    return VectorStoreManager(s.chroma_persist_dir, s.chroma_collection, get_embeddings())


@lru_cache
def get_llm_client():
    return get_llm(get_settings())  # raises LLMNotConfiguredError (-> HTTP 503) when no API key


def get_ingestion_service() -> IngestionService:
    return IngestionService(get_vector_store(), get_settings())


def get_qa_service() -> QAService:
    return QAService(get_vector_store(), get_llm_client(), get_settings())


__all__ = [
    "Settings",
    "get_settings",
    "get_vector_store",
    "get_ingestion_service",
    "get_qa_service",
]
