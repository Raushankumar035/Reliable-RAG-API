import pytest

from app.core.exceptions import (
    CorruptDocumentError,
    EmptyDocumentError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_document, safe_filename
from tests.conftest import SAMPLES_DIR


def test_load_markdown_and_txt():
    md = load_document("a.md", b"# Title\n\nSome content here.")
    txt = load_document("b.txt", b"plain text content")
    assert md[0].metadata["file_name"] == "a.md"
    assert txt[0].text == "plain text content"


def test_load_pdf_one_document_per_page_with_page_numbers():
    docs = load_document("onboarding_guide.pdf", (SAMPLES_DIR / "onboarding_guide.pdf").read_bytes())
    assert [d.metadata["page"] for d in docs] == [1, 2]
    assert "laptop" in docs[0].text.lower()


def test_unsupported_extension_rejected():
    with pytest.raises(UnsupportedFileTypeError):
        load_document("malware.exe", b"x")


def test_empty_and_whitespace_files_rejected():
    for data in (b"", b"   \n\t "):
        with pytest.raises(EmptyDocumentError):
            load_document("empty.txt", data)


def test_corrupt_pdf_rejected():
    with pytest.raises(CorruptDocumentError):
        load_document("broken.pdf", b"this is not a pdf")


def test_path_traversal_filename_is_sanitised():
    assert safe_filename("../../etc/passwd.txt") == "passwd.txt"
    assert safe_filename("C:\\temp\\notes.md") == "notes.md"


def test_chunker_ids_overlap_and_metadata():
    text = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(200))
    docs = load_document("long.txt", text.encode())
    nodes = chunk_documents(docs, chunk_size=128, chunk_overlap=20)
    assert len(nodes) > 3
    assert [n.id_ for n in nodes[:2]] == ["long.txt::0000", "long.txt::0001"]
    assert all(n.metadata["file_name"] == "long.txt" for n in nodes)
    # metadata must not leak into the text that gets embedded
    assert "file_name" not in nodes[0].get_content(metadata_mode="embed")


def test_chunker_rejects_bad_overlap():
    docs = load_document("a.txt", b"hello world")
    with pytest.raises(ValueError):
        chunk_documents(docs, chunk_size=100, chunk_overlap=100)


def test_ingest_is_idempotent_and_replaces_changed_content(ingestion, store):
    first = ingestion.ingest_bytes("note.txt", b"Alpha document about apples.")
    assert first.status == "ingested"
    count = store.count()

    again = ingestion.ingest_bytes("note.txt", b"Alpha document about apples.")
    assert again.status == "skipped_unchanged"
    assert store.count() == count

    changed = ingestion.ingest_bytes("note.txt", b"Completely different text about oranges.")
    assert changed.status == "replaced"
    hits = store.query("oranges", 3)
    assert hits and "oranges" in hits[0].text
    assert all("apples" not in h.text for h in store.query("apples", 3))


def test_ingest_rejects_oversized_file(ingestion):
    with pytest.raises(FileTooLargeError):
        ingestion.ingest_bytes("big.txt", b"a" * (2 * 1024 * 1024))


def test_failed_ingest_does_not_modify_store(ingestion, store):
    ingestion.ingest_bytes("keep.txt", b"Important content to keep.")
    before = store.count()
    result = ingestion.ingest_safely("keep.pdf", b"garbage")
    assert result.status == "failed" and result.error_code == "corrupt_document"
    assert store.count() == before


def test_ingest_directory_loads_all_sample_types(ingestion, store):
    results = ingestion.ingest_directory(str(SAMPLES_DIR))
    assert {r.file_name for r in results} == {
        "employee_handbook.md", "it_security_policy.txt", "product_faq.md", "onboarding_guide.pdf",
    }
    assert all(r.status == "ingested" and r.chunks > 0 for r in results)
