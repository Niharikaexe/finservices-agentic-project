"""Vector stores behind one interface — and the ACL is a *query argument*, not a filter
applied afterwards.

Read the signature of `search` closely. `allowed_document_ids` is a required parameter,
not an optional one, and there is no method that searches without it. That is the whole
design: it is not possible to call this store in a way that ranks over documents the
caller may not read, because there is no such method to call.

Post-retrieval filtering — fetch 20, throw away 12 — fails in three ways that a code
reviewer will not catch:

  1. **Ranking is computed over forbidden documents.** The 8 you keep were ranked
     against 12 you had no right to see, so their order was influenced by them.
  2. **Result counts leak existence.** Ask for 20, get 8 back — you have learned that
     12 documents exist that you cannot read.
  3. **Latency leaks too.** A query that matches many forbidden documents is slower.

`InMemoryVectorStore` is what the tests use. `PGVECTOR_SEARCH_SQL` is the same
predicate expressed for Postgres, with the ACL as `document_id = ANY($3)`.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

import numpy as np
import numpy.typing as npt

from fsa_retrieval.chunking import Chunk


class VectorStore(Protocol):
    def index(self, chunks: list[Chunk], vectors: npt.NDArray[np.float64]) -> None: ...

    def search(
        self,
        *,
        query_vector: npt.NDArray[np.float64],
        tenant_id: str,
        allowed_document_ids: list[str],
        as_of: date,
        k: int,
    ) -> list[tuple[Chunk, float]]: ...

    @property
    def size(self) -> int: ...


class InMemoryVectorStore:
    """Numpy-backed store. Exact search — no ANN index, no approximation.

    A policy corpus is thousands of chunks, not millions, so exact cosine over a dense
    matrix is both faster and more honest than an approximate index. Recall@k of an ANN
    index is a confound you do not want sitting underneath an authorisation test: if a
    probe returns nothing, you need to know it was the ACL and not the index.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: npt.NDArray[np.float64] | None = None

    @property
    def size(self) -> int:
        return len(self._chunks)

    def index(self, chunks: list[Chunk], vectors: npt.NDArray[np.float64]) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(f"{len(chunks)} chunks vs {vectors.shape[0]} vectors")
        self._chunks = list(chunks)
        self._vectors = vectors

    def search(
        self,
        *,
        query_vector: npt.NDArray[np.float64],
        tenant_id: str,
        allowed_document_ids: list[str],
        as_of: date,
        k: int,
    ) -> list[tuple[Chunk, float]]:
        if self._vectors is None:
            return []
        if not allowed_document_ids:
            # No readable documents means no results. Not "search anyway and filter" —
            # there is nothing this caller is entitled to have ranked.
            return []

        allowed = set(allowed_document_ids)

        # ── the predicate, applied BEFORE any scoring ───────────────────────
        # Three independent conditions, matching the three layers of defence in §10:
        # tenant (defence in depth behind RLS), ACL (object-level), and temporal.
        candidate_idx = [
            i
            for i, chunk in enumerate(self._chunks)
            if chunk.tenant_id == tenant_id
            and chunk.document_id in allowed
            and chunk.is_live(as_of)
        ]
        if not candidate_idx:
            return []

        # Only now does anything get scored. The similarity computation never sees a
        # row the caller may not read.
        subset = self._vectors[candidate_idx]
        scores = subset @ query_vector
        order = np.argsort(-scores)[:k]
        return [(self._chunks[candidate_idx[int(i)]], float(scores[int(i)])) for i in order]


#: The same query for Postgres + pgvector. Note that the ACL is `= ANY($3)` inside the
#: WHERE clause, so the index scan and the ORDER BY both happen over the permitted set
#: only. RLS on `tenant_id` is the primary tenant defence; the explicit `tenant_id = $2`
#: is defence in depth, and the ACL list is the object-level layer on top.
PGVECTOR_SEARCH_SQL = """
SELECT chunk_id, document_id, content, rule_ref,
       embedding <=> $1 AS distance
FROM policy_chunks
WHERE tenant_id = $2                       -- RLS also enforces this
  AND document_id = ANY($3)                -- ACL as predicate, not post-filter
  AND effective_from <= $4
  AND (effective_to IS NULL OR effective_to > $4)
ORDER BY embedding <=> $1
LIMIT $5
"""
