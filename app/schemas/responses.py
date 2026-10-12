from typing import List, Literal, Optional

from pydantic import BaseModel


class Source(BaseModel):
    document: str
    page: Optional[int] = None
    excerpt: str
    similarity: float
    chunk_id: str
    quote_verified: bool  # True when the excerpt is a verbatim quote found in the chunk text


AnswerReason = Literal[
    "answered",  # grounded answer with evidence
    "no_documents",  # knowledge base is empty
    "no_relevant_context",  # nothing retrieved was similar enough (LLM not called)
    "not_in_documents",  # LLM judged the context insufficient
    "ungrounded_answer",  # LLM answered but gave no valid citation -> rejected
]


class AskResponse(BaseModel):
    question: str
    answer: str
    answerable: bool
    reason: AnswerReason
    sources: List[Source]
    top_similarity: Optional[float] = None


class IngestFileResult(BaseModel):
    file_name: str
    status: Literal["ingested", "replaced", "skipped_unchanged", "failed"]
    chunks: int = 0
    error: Optional[str] = None
    error_code: Optional[str] = None


class IngestResponse(BaseModel):
    results: List[IngestFileResult]


class DocumentOut(BaseModel):
    file_name: str
    chunks: int


class DocumentList(BaseModel):
    total_documents: int
    total_chunks: int
    documents: List[DocumentOut]


class HealthResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    llm_configured: bool
    llm_model: str
    embedding_provider: str
    embedding_model: str


class ErrorResponse(BaseModel):
    error: str
    message: str
