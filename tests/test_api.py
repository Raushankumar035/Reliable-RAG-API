from app.core.exceptions import LLMNotConfiguredError
from tests.conftest import FakeLLM, SAMPLES_DIR, llm_json
from app.dependencies import get_qa_service


def _upload(client, *names):
    files = [("files", (n, (SAMPLES_DIR / n).read_bytes())) for n in names]
    return client.post("/documents", files=files)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["documents"] == 0 and body["llm_configured"] is True


def test_upload_multiple_files_and_list(client):
    r = _upload(client, "employee_handbook.md", "it_security_policy.txt", "onboarding_guide.pdf")
    assert r.status_code == 200
    assert [x["status"] for x in r.json()["results"]] == ["ingested"] * 3
    listing = client.get("/documents").json()
    assert listing["total_documents"] == 3 and listing["total_chunks"] > 0


def test_reupload_same_file_is_skipped(client):
    _upload(client, "product_faq.md")
    assert _upload(client, "product_faq.md").json()["results"][0]["status"] == "skipped_unchanged"


def test_upload_unsupported_type_returns_400_with_reason(client):
    r = client.post("/documents", files=[("files", ("evil.exe", b"MZ"))])
    assert r.status_code == 400
    assert r.json()["results"][0]["error_code"] == "unsupported_file_type"


def test_upload_partial_failure_still_200(client):
    r = client.post(
        "/documents",
        files=[("files", ("ok.txt", b"hello world document")), ("files", ("empty.txt", b""))],
    )
    assert r.status_code == 200
    statuses = {x["file_name"]: x["status"] for x in r.json()["results"]}
    assert statuses == {"ok.txt": "ingested", "empty.txt": "failed"}


def test_upload_oversized_file_rejected(client):
    r = client.post("/documents", files=[("files", ("big.txt", b"a" * (2 * 1024 * 1024)))])
    assert r.status_code == 400 and r.json()["results"][0]["error_code"] == "file_too_large"


def test_upload_corrupt_pdf_rejected(client):
    r = client.post("/documents", files=[("files", ("bad.pdf", b"not a pdf"))])
    assert r.status_code == 400 and r.json()["results"][0]["error_code"] == "corrupt_document"


def test_upload_without_files_is_422(client):
    assert client.post("/documents").status_code == 422


def test_delete_document_and_404(client):
    _upload(client, "product_faq.md")
    assert client.delete("/documents/product_faq.md").status_code == 200
    r = client.delete("/documents/product_faq.md")
    assert r.status_code == 404 and r.json()["error"] == "document_not_found"


def test_ask_returns_answer_with_evidence(client, fake_llm):
    _upload(client, "employee_handbook.md")
    fake_llm.responses = [llm_json(answer="24 days [1]", citations=[{"source": 1, "quote": "24 days of paid annual leave"}])]
    r = client.post("/ask", json={"question": "How many annual leave days do employees get?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answerable"] is True and body["reason"] == "answered"
    assert body["sources"][0]["document"] == "employee_handbook.md"
    assert body["sources"][0]["excerpt"]


def test_ask_unanswerable_says_so(client, fake_llm):
    _upload(client, "employee_handbook.md")
    r = client.post("/ask", json={"question": "What is the capital of France?"})
    body = r.json()
    assert r.status_code == 200 and body["answerable"] is False and body["sources"] == []
    assert "do not contain" in body["answer"]


def test_ask_on_empty_knowledge_base(client):
    body = client.post("/ask", json={"question": "Anything in here?"}).json()
    assert body["answerable"] is False and body["reason"] == "no_documents"


def test_ask_validation_errors(client):
    assert client.post("/ask", json={}).status_code == 422
    assert client.post("/ask", json={"question": "  "}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "top_k": 0}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 2000}).status_code == 422


def test_ask_when_llm_fails_returns_502(client):
    _upload(client, "employee_handbook.md")
    client.app.dependency_overrides[get_qa_service] = lambda: _failing_qa(client)
    r = client.post("/ask", json={"question": "How many annual leave days are there?"})
    assert r.status_code == 502 and r.json()["error"] == "llm_unavailable"


def test_ask_when_llm_returns_garbage_returns_502(client, fake_llm):
    _upload(client, "employee_handbook.md")
    fake_llm.responses = ["garbage", "more garbage"]
    r = client.post("/ask", json={"question": "How many annual leave days are there?"})
    assert r.status_code == 502 and r.json()["error"] == "llm_invalid_response"


def test_ask_without_api_key_returns_503(client):
    def missing():
        raise LLMNotConfiguredError("GROQ_API_KEY is not set")

    client.app.dependency_overrides[get_qa_service] = missing
    r = client.post("/ask", json={"question": "Does this work?"})
    assert r.status_code == 503 and r.json()["error"] == "llm_not_configured"


def _failing_qa(client):
    from app.services.qa_service import QAService
    from app.dependencies import get_vector_store
    from app.config import get_settings

    store = client.app.dependency_overrides[get_vector_store]()
    settings = client.app.dependency_overrides[get_settings]()
    return QAService(store, FakeLLM([ConnectionError("boom")]), settings)
