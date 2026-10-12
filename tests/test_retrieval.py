import json

import pytest

from app.config import Settings
from app.retrieval.embeddings import HashEmbedding, get_embed_model
from app.retrieval.vector_store import VectorStoreManager
from tests.conftest import SAMPLES_DIR

QUESTIONS = json.loads((SAMPLES_DIR.parent.parent / "evaluation" / "questions.json").read_text())


def test_empty_store_returns_nothing(store):
    assert store.query("anything", 5) == []


def test_relevant_chunk_ranked_first(loaded_store):
    hits = loaded_store.query("How many days of annual leave do employees get?", 3)
    assert hits[0].file_name == "employee_handbook.md"
    assert "24 days" in hits[0].text


def test_scores_are_cosine_similarities_sorted_desc(loaded_store):
    hits = loaded_store.query("password length requirement", 5)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in scores)
    assert hits[0].file_name == "it_security_policy.txt"


def test_pdf_chunks_carry_page_number(loaded_store):
    hits = loaded_store.query("when do I collect my laptop on the first day", 3)
    top = next(h for h in hits if h.file_name == "onboarding_guide.pdf")
    assert top.page == 1


def test_top_k_is_respected_and_capped_by_collection_size(loaded_store):
    assert len(loaded_store.query("leave", 2)) == 2
    assert len(loaded_store.query("leave", 10_000)) == loaded_store.count()


def test_delete_document_removes_its_chunks(loaded_store):
    removed = loaded_store.delete_document("product_faq.md")
    assert removed > 0
    assert loaded_store.delete_document("product_faq.md") == 0
    assert all(h.file_name != "product_faq.md" for h in loaded_store.query("Growth plan price", 20))


def test_list_documents(loaded_store):
    names = [d.file_name for d in loaded_store.list_documents()]
    assert names == sorted(names) and len(names) == 4


def test_data_persists_across_instances(settings, loaded_store):
    reopened = VectorStoreManager(settings.chroma_persist_dir, settings.chroma_collection, HashEmbedding())
    assert reopened.count() == loaded_store.count() > 0


def test_unrelated_query_scores_below_relevant_query(loaded_store):
    relevant = loaded_store.query("annual leave days", 1)[0].score
    unrelated = loaded_store.query("capital city of France", 1)[0].score
    assert relevant > unrelated
    assert unrelated < 0.15  # below the gate used in tests


def test_hash_embedding_is_deterministic_and_normalised():
    emb = HashEmbedding()
    a, b = emb.get_text_embedding("Annual leave policy"), emb.get_text_embedding("Annual leave policy")
    assert a == b
    assert sum(x * x for x in a) == pytest.approx(1.0)


def test_embedding_factory_selects_provider():
    assert isinstance(get_embed_model(Settings(_env_file=None, embedding_provider="hash")), HashEmbedding)


@pytest.mark.parametrize("q", [q for q in QUESTIONS if q["answerable"]], ids=lambda q: q["id"])
def test_retrieval_recall_on_evaluation_set(loaded_store, q):
    """Every document needed for an answerable question appears in the top-5 retrieved chunks."""
    retrieved_docs = {c.file_name for c in loaded_store.query(q["question"], 5)}
    assert set(q["expected_sources"]) <= retrieved_docs, f"{q['id']}: got {retrieved_docs}"
