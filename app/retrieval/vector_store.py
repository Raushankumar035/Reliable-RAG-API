"""ChromaDB-backed vector store.

LlamaIndex handles node insertion/embedding (``VectorStoreIndex``). Queries go straight to the
Chroma collection so we get a true cosine similarity score (the LlamaIndex Chroma adapter reports
``exp(-distance)``, which is awkward to threshold). Cosine similarity = ``1 - cosine_distance``.
"""
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float  # cosine similarity, higher = more similar
    file_name: str
    page: Optional[int] = None
    chunk_index: Optional[int] = None


@dataclass
class DocumentInfo:
    file_name: str
    chunks: int
    content_hash: Optional[str]


class VectorStoreManager:
    def __init__(self, persist_dir: str, collection_name: str, embed_model: BaseEmbedding):
        self._embed_model = embed_model
        self._client = chromadb.PersistentClient(
            path=persist_dir, settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._index = VectorStoreIndex.from_vector_store(
            ChromaVectorStore(chroma_collection=self._collection), embed_model=embed_model
        )

    # ---- writes -----------------------------------------------------------
    def add_nodes(self, nodes: List[TextNode]) -> int:
        self._index.insert_nodes(nodes)
        return len(nodes)

    def delete_document(self, file_name: str) -> int:
        existing = self._collection.get(where={"file_name": file_name}, include=[])
        ids = existing["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    # ---- reads ------------------------------------------------------------
    def count(self) -> int:
        return self._collection.count()

    def query(self, question: str, top_k: int) -> List[RetrievedChunk]:
        total = self.count()
        if total == 0:
            return []
        embedding = self._embed_model.get_query_embedding(question)
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    score=round(1.0 - float(distance), 4),
                    file_name=meta.get("file_name", "unknown"),
                    page=meta.get("page"),
                    chunk_index=meta.get("chunk_index"),
                )
            )
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks

    def list_documents(self) -> List[DocumentInfo]:
        data = self._collection.get(include=["metadatas"])
        counts: Dict[str, int] = defaultdict(int)
        hashes: Dict[str, Optional[str]] = {}
        for meta in data["metadatas"]:
            name = meta.get("file_name", "unknown")
            counts[name] += 1
            hashes[name] = meta.get("content_hash")
        return [DocumentInfo(n, counts[n], hashes[n]) for n in sorted(counts)]

    def get_document(self, file_name: str) -> Optional[DocumentInfo]:
        return next((d for d in self.list_documents() if d.file_name == file_name), None)
