"""fsa_retrieval — permission-aware RAG over the policy corpus (ARCHITECTURE.md §10).

The one rule this package exists to enforce: **authorisation is a query predicate,
never a post-retrieval discard.** `VectorStore.search` has no signature that omits the
allowed-document list, so the wrong thing is not merely discouraged — it is unwritable.
"""

from fsa_retrieval.chunking import Chunk
from fsa_retrieval.embedding import AzureOpenAIEmbedder, Embedder, HashingEmbedder
from fsa_retrieval.ingestion import IngestionReport, PolicyRuleLike, ingest
from fsa_retrieval.retriever import (
    PermissionAwareRetriever,
    RetrievalResult,
    RetrievedChunk,
)
from fsa_retrieval.store import PGVECTOR_SEARCH_SQL, InMemoryVectorStore, VectorStore

__all__ = [
    "PGVECTOR_SEARCH_SQL",
    "AzureOpenAIEmbedder",
    "Chunk",
    "Embedder",
    "HashingEmbedder",
    "InMemoryVectorStore",
    "IngestionReport",
    "PermissionAwareRetriever",
    "PolicyRuleLike",
    "RetrievalResult",
    "RetrievedChunk",
    "VectorStore",
    "ingest",
]
