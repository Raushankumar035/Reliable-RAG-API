"""Shared fixtures. Tests run fully offline: hash embeddings + a scripted fake LLM + temp Chroma dir."""
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Union

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.dependencies import get_ingestion_service, get_qa_service, get_vector_store
from app.main import create_app
from app.retrieval.embeddings import HashEmbedding
from app.retrieval.vector_store import VectorStoreManager
from app.services.ingestion_service import IngestionService
from app.services.qa_service import QAService

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_documents"


class FakeLLM:
    """Stands in for the Groq LLM. ``responses`` is a list of replies (or a callable(messages)->str)."""

    def __init__(self, responses: Union[List[str], Callable] = None):
        self.responses = responses if responses is not None else []
        self.calls: List[list] = []

    def chat(self, messages):
        self.calls.append(messages)
        if callable(self.responses):
            reply = self.responses(messages)
        else:
            reply = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(message=SimpleNamespace(content=reply))


def llm_json(answerable=True, answer="ok [1]", citations=None) -> str:
    return json.dumps({"answerable": answerable, "answer": answer, "citations": citations or []})


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        groq_api_key="test-key",
        embedding_provider="hash",
        chroma_persist_dir=str(tmp_path / "chroma"),
        chroma_collection="test_docs",
        min_similarity=0.15,
        top_k=5,
        max_upload_mb=1,
    )


@pytest.fixture
def store(settings) -> VectorStoreManager:
    return VectorStoreManager(settings.chroma_persist_dir, settings.chroma_collection, HashEmbedding())


@pytest.fixture
def ingestion(store, settings) -> IngestionService:
    return IngestionService(store, settings)


@pytest.fixture
def loaded_store(store, ingestion):
    """Store pre-loaded with the sample documents."""
    ingestion.ingest_directory(str(SAMPLES_DIR))
    return store


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def qa(loaded_store, fake_llm, settings) -> QAService:
    return QAService(loaded_store, fake_llm, settings)


@pytest.fixture
def client(store, ingestion, fake_llm, settings):
    app = create_app()
    qa_service = QAService(store, fake_llm, settings)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_ingestion_service] = lambda: ingestion
    app.dependency_overrides[get_qa_service] = lambda: qa_service
    with TestClient(app) as test_client:
        yield test_client
