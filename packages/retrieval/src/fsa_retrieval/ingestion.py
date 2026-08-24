"""Ingestion: policy documents in, indexed chunks out.

Chunking is per *rule*, not per fixed token window. That is a deliberate departure from
the default recipe, and it is the right call for this corpus: a policy rule is already
the unit a citation points at, so splitting on rule boundaries means every retrieved
chunk maps cleanly to a `rule_ref` and every verdict is groundable. Fixed-size windows
would straddle two rules and make citation ambiguous.

For a corpus of prose you would chunk differently. Knowing why you chose one over the
other is the part that matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as np

from fsa_retrieval.chunking import Chunk
from fsa_retrieval.embedding import Embedder
from fsa_retrieval.store import VectorStore


class PolicyRuleLike(Protocol):
    """What ingestion needs from a rule row.

    A structural type rather than an import: `packages/*` must not depend on the
    simulator (import-linter enforces it), and in production these rows come from
    Postgres. Anything with these attributes ingests, and mypy still checks the call
    sites — duck typing without giving up the type checker.
    """

    rule_ref: str
    document_id: str
    tenant_id: str
    scope_department_id: str | None
    text: str
    effective_from: date
    effective_to: date | None


@dataclass(frozen=True, slots=True)
class IngestionReport:
    documents: int
    chunks: int
    dimensions: int


def ingest(
    rules: list[PolicyRuleLike],
    *,
    store: VectorStore,
    embedder: Embedder,
) -> IngestionReport:
    """Index policy rules.

    See `PolicyRuleLike` for the shape a rule must have.
    """
    chunks: list[Chunk] = []
    for i, rule in enumerate(rules):
        rule_ref = rule.rule_ref
        document_id = rule.document_id
        # The embedded text includes the rule reference, so a query citing "T&E-4.2"
        # can find it lexically as well as semantically.
        content = f"[{rule_ref}] {rule.text}"
        chunks.append(
            Chunk(
                chunk_id=f"{document_id}#{rule_ref}#{i:05d}",
                document_id=document_id,
                tenant_id=rule.tenant_id,
                scope_department_id=rule.scope_department_id,
                rule_ref=rule_ref,
                content=content,
                effective_from=rule.effective_from,
                effective_to=rule.effective_to,
            )
        )

    corpus = [c.content for c in chunks]
    embedder.fit(corpus)
    vectors = embedder.encode(corpus) if corpus else np.zeros((0, embedder.dimensions))
    store.index(chunks, vectors)

    return IngestionReport(
        documents=len({c.document_id for c in chunks}),
        chunks=len(chunks),
        dimensions=embedder.dimensions,
    )
