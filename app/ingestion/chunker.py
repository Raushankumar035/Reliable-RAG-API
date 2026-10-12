"""Split Documents into overlapping, sentence-aware chunks (LlamaIndex nodes)."""
from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from app.core.exceptions import EmptyDocumentError


def chunk_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[TextNode]:
    """Chunk documents. Chunk ids are deterministic (``<file>::<index>``) so re-ingestion is idempotent."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = [n for n in splitter.get_nodes_from_documents(documents) if n.get_content().strip()]
    if not nodes:
        raise EmptyDocumentError("Document produced no chunks.")

    for index, node in enumerate(nodes):
        node.id_ = f"{node.metadata['file_name']}::{index:04d}"
        node.metadata["chunk_index"] = index
        for keys in (node.excluded_embed_metadata_keys, node.excluded_llm_metadata_keys):
            if "chunk_index" not in keys:
                keys.append("chunk_index")
    return nodes
