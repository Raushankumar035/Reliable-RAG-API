"""Question answering with layered grounding checks.

Pipeline:  retrieve -> similarity gate -> LLM (strict JSON) -> validate citations -> verify quotes.
Any layer can end the request with an honest "information unavailable" answer.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core.llms import ChatMessage, MessageRole

from app.config import Settings
from app.core.exceptions import LLMResponseError, LLMUnavailableError
from app.generation.prompts import RETRY_NUDGE, SYSTEM_PROMPT, UNAVAILABLE_MESSAGE, build_user_prompt
from app.retrieval.vector_store import RetrievedChunk, VectorStoreManager
from app.schemas.responses import AnswerReason, AskResponse, Source

logger = logging.getLogger(__name__)

_EXCERPT_FALLBACK_CHARS = 400


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def parse_llm_json(raw: str) -> Dict[str, Any]:
    """Tolerantly extract the JSON object from an LLM reply (handles ``` fences / stray prose)."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "answerable" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError("no valid JSON object with an 'answerable' field found")


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip().strip("[]"))
    except (TypeError, ValueError):
        return None


class QAService:
    def __init__(self, store: VectorStoreManager, llm: Any, settings: Settings):
        self._store = store
        self._llm = llm
        self._settings = settings

    # ------------------------------------------------------------------ public
    def answer(self, question: str, top_k: Optional[int] = None) -> AskResponse:
        k = min(top_k or self._settings.top_k, self._settings.max_top_k)
        retrieved = self._store.query(question, k)

        if not retrieved:
            return self._unavailable(question, "no_documents", None)

        top_score = retrieved[0].score
        relevant = [c for c in retrieved if c.score >= self._settings.min_similarity]
        if not relevant:
            logger.info("Gate: best similarity %.3f < %.3f", top_score, self._settings.min_similarity)
            return self._unavailable(question, "no_relevant_context", top_score)

        parsed = self._ask_llm(question, relevant)

        if not parsed.get("answerable"):
            return self._unavailable(question, "not_in_documents", top_score)

        answer_text = str(parsed.get("answer", "")).strip()
        sources = self._validate_citations(parsed.get("citations"), relevant)
        if not answer_text or not sources:
            logger.warning("Rejecting ungrounded answer (no valid citations)")
            return self._unavailable(question, "ungrounded_answer", top_score)

        return AskResponse(
            question=question,
            answer=answer_text,
            answerable=True,
            reason="answered",
            sources=sources,
            top_similarity=top_score,
        )

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _unavailable(question: str, reason: AnswerReason, top_score: Optional[float]) -> AskResponse:
        return AskResponse(
            question=question,
            answer=UNAVAILABLE_MESSAGE,
            answerable=False,
            reason=reason,
            sources=[],
            top_similarity=top_score,
        )

    def _chat(self, messages: List[ChatMessage]) -> str:
        try:
            response = self._llm.chat(messages)
        except Exception as exc:  # network, auth, rate limit, timeout ...
            logger.exception("LLM call failed")
            raise LLMUnavailableError(f"The LLM provider call failed: {type(exc).__name__}: {exc}") from exc
        return str(response.message.content or "")

    def _ask_llm(self, question: str, chunks: List[RetrievedChunk]) -> Dict[str, Any]:
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=build_user_prompt(question, chunks)),
        ]
        raw = self._chat(messages)
        try:
            return parse_llm_json(raw)
        except ValueError:
            logger.warning("LLM returned invalid JSON; retrying once. Raw: %.200s", raw)

        messages += [
            ChatMessage(role=MessageRole.ASSISTANT, content=raw),
            ChatMessage(role=MessageRole.USER, content=RETRY_NUDGE),
        ]
        raw = self._chat(messages)
        try:
            return parse_llm_json(raw)
        except ValueError as exc:
            raise LLMResponseError("The LLM did not return a valid structured answer.") from exc

    @staticmethod
    def _validate_citations(raw_citations: Any, chunks: List[RetrievedChunk]) -> List[Source]:
        """Keep only citations pointing at real retrieved passages; prefer verified verbatim quotes."""
        if not isinstance(raw_citations, list):
            return []
        by_index: Dict[int, Tuple[RetrievedChunk, Optional[str]]] = {}
        for item in raw_citations:
            index = _as_int(item.get("source") if isinstance(item, dict) else item)
            if index is None or not 1 <= index <= len(chunks):
                continue  # hallucinated reference
            chunk = chunks[index - 1]
            quote = str(item.get("quote", "")).strip() if isinstance(item, dict) else ""
            verified = quote if quote and _norm(quote) in _norm(chunk.text) else None
            if index not in by_index or (verified and by_index[index][1] is None):
                by_index[index] = (chunk, verified)

        sources = []
        for chunk, verified in by_index.values():
            excerpt = verified or chunk.text.strip()[:_EXCERPT_FALLBACK_CHARS]
            sources.append(
                Source(
                    document=chunk.file_name,
                    page=chunk.page,
                    excerpt=excerpt,
                    similarity=chunk.score,
                    chunk_id=chunk.chunk_id,
                    quote_verified=verified is not None,
                )
            )
        return sources
