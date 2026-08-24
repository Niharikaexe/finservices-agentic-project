"""Chunks carry their own authorisation and validity metadata.

A chunk is not just text. It carries `document_id` (what the ACL is expressed over),
`rule_ref` (what a citation points at) and `effective_from`/`effective_to` (when it was
true). Losing any of those at chunk time makes correct retrieval impossible later —
you cannot filter on metadata you did not keep.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    tenant_id: str
    scope_department_id: str | None
    rule_ref: str
    content: str
    effective_from: date
    effective_to: date | None

    def is_live(self, as_of: date) -> bool:
        """In force on `as_of`. Half-open interval: `effective_to` is exclusive, so a
        v1 ending and a v2 starting on the same day never both apply."""
        return self.effective_from <= as_of and (
            self.effective_to is None or self.effective_to > as_of
        )
