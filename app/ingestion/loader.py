"""Turn raw uploaded bytes (PDF / TXT / Markdown) into LlamaIndex ``Document`` objects."""
import hashlib
import io
import logging
from pathlib import Path
from typing import List

from llama_index.core import Document
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app.core.exceptions import CorruptDocumentError, EmptyDocumentError, UnsupportedFileTypeError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}
META_KEYS = ["file_name", "page", "content_hash"]


def safe_filename(filename: str) -> str:
    """Strip any directory components (prevents path tricks like ../../etc/passwd)."""
    return Path(filename.replace("\\", "/")).name


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_document(text: str, file_name: str, digest: str, page: int | None = None) -> Document:
    metadata = {"file_name": file_name, "content_hash": digest}
    if page is not None:
        metadata["page"] = page
    # Metadata is stored for citations only; it must not pollute embeddings or LLM prompts
    # (and must not count against the chunk-size budget).
    return Document(
        text=text,
        metadata=metadata,
        excluded_embed_metadata_keys=META_KEYS,
        excluded_llm_metadata_keys=META_KEYS,
    )


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return data.decode("utf-8", errors="replace")


def _load_pdf(data: bytes, file_name: str, digest: str) -> List[Document]:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise CorruptDocumentError(f"'{file_name}' is password-protected and cannot be read.")
        docs = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(_make_document(text, file_name, digest, page=page_number))
        return docs
    except CorruptDocumentError:
        raise
    except (PyPdfError, ValueError, OSError, KeyError, AttributeError) as exc:
        raise CorruptDocumentError(f"'{file_name}' is not a valid PDF: {exc}") from exc


def load_document(filename: str, data: bytes) -> List[Document]:
    """Parse a file into one or more Documents (one per PDF page, one for TXT/MD)."""
    file_name = safe_filename(filename)
    extension = Path(file_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{extension or 'none'}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    digest = content_hash(data)
    if extension == ".pdf":
        docs = _load_pdf(data, file_name, digest)
    else:
        text = _decode_text(data).strip()
        docs = [_make_document(text, file_name, digest)] if text else []

    if not docs:
        raise EmptyDocumentError(
            f"'{file_name}' contains no extractable text (it may be empty or a scanned image PDF)."
        )
    logger.info("Loaded '%s' (%d section(s))", file_name, len(docs))
    return docs
