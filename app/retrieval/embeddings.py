"""Embedding model factory.

* ``fastembed`` – real semantic embeddings, run locally with ONNX (no API key, no GPU, no torch).
* ``hash``      – deterministic bag-of-words hashing embedding. Not semantic, but fully offline and
                  reproducible; used by the automated tests and for quick offline smoke runs.
"""
import hashlib
import math
import re
from typing import List

from llama_index.core.base.embeddings.base import BaseEmbedding

from app.config import Settings

_STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from has have how i if in is it its of on or "
    "our that the their there they this to was we what when where which who why will with you your "
    "many much should would could about into than then them these those".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    out = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOPWORDS:
            continue
        if len(tok) > 3 and tok.endswith("s"):  # crude plural folding
            tok = tok[:-1]
        out.append(tok)
    return out


class HashEmbedding(BaseEmbedding):
    """Feature-hashing bag-of-words embedding (L2-normalised, so dot product == cosine)."""

    dim: int = 512

    @classmethod
    def class_name(cls) -> str:
        return "HashEmbedding"

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed(text)


def get_embed_model(settings: Settings) -> BaseEmbedding:
    if settings.embedding_provider == "hash":
        return HashEmbedding()
    from llama_index.embeddings.fastembed import FastEmbedEmbedding  # lazy: heavy import

    return FastEmbedEmbedding(model_name=settings.embedding_model)
