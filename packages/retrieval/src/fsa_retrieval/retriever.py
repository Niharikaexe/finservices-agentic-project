"""Permission-aware retrieval — ARCHITECTURE.md §10, implemented.

The whole milestone is four lines of ordering:

    1. ask the authorisation store which documents this principal may read
    2. if none, return nothing — do not search
    3. push that list into the query as a predicate
    4. rank only what came back

Anything that reverses steps 3 and 4 is post-retrieval filtering, and §10 forbids it.

The retriever also does *not* fall back. If the authz store raises, the exception
propagates as `FailClosedError`. There is no `except: allowed = []` — that would turn
an outage into a silent, system-wide denial that looks exactly like correct behaviour
and would never page anyone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date

from fsa_authz import AuthorizationStore, Principal
from fsa_common import get_logger
from fsa_retrieval.chunking import Chunk
from fsa_retrieval.embedding import Embedder
from fsa_retrieval.store import VectorStore

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float

    @property
    def citation(self) -> str:
        """What a policy verdict points at. Every verdict the agent reaches must carry
        at least one of these — `TriageDecision.citations` has `min_length=1`."""
        return f"{self.chunk.document_id}#{self.chunk.rule_ref}"


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    allowed_document_count: int
    candidate_count: int
    latency_ms: float

    @property
    def citations(self) -> list[str]:
        return [c.citation for c in self.chunks]


class PermissionAwareRetriever:
    """The only retrieval path. There is no unfiltered variant, deliberately."""

    def __init__(
        self,
        *,
        authz: AuthorizationStore,
        store: VectorStore,
        embedder: Embedder,
    ) -> None:
        self._authz = authz
        self._store = store
        self._embedder = embedder

    def retrieve(
        self,
        query: str,
        principal: Principal,
        *,
        as_of: date,
        k: int = 5,
    ) -> RetrievalResult:
        """Retrieve policy chunks this principal may read, in force on `as_of`.

        `as_of` is the **expense date**, not `now()`. A claim submitted in June for a
        March dinner is judged against March's policy. Defaulting this to today is the
        single easiest way to produce a confidently wrong verdict, so there is no
        default.
        """
        started = time.perf_counter()

        # 1. ListObjects, not N checks. One round trip; see store.py's docstring for
        #    why per-chunk checking is both slower and architecturally wrong.
        allowed = self._authz.list_objects(
            user=principal.fga_user, relation="reader", type="policy_document"
        )

        # 2. Nothing readable means nothing returned — and we never touch the index.
        if not allowed:
            return RetrievalResult([], 0, 0, (time.perf_counter() - started) * 1000)

        # 3 & 4. The ACL goes in as a predicate; ranking happens over the permitted set.
        vector = self._embedder.encode([query])[0]
        hits = self._store.search(
            query_vector=vector,
            tenant_id=principal.tenant_id,
            allowed_document_ids=allowed,
            as_of=as_of,
            k=k,
        )

        latency_ms = (time.perf_counter() - started) * 1000
        result = RetrievalResult(
            chunks=[RetrievedChunk(chunk, score) for chunk, score in hits],
            allowed_document_count=len(allowed),
            candidate_count=len(hits),
            latency_ms=latency_ms,
        )

        # Note what is NOT logged: no chunk text, no user id in a metric label. The
        # log line carries the role and counts; per-request forensics is Phoenix's job.
        log.debug(
            "policy retrieval",
            role=principal.role.value,
            tenant=principal.tenant_id,
            allowed_documents=len(allowed),
            returned=len(hits),
            latency_ms=round(latency_ms, 2),
        )
        return result
