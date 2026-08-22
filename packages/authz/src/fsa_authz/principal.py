"""Who is asking.

`Principal` is passed to every retrieval and every tool invocation. It is *not* a
convenience bag of user fields — it exists so that "authorise at query time" (§10) is
the path of least resistance: a function that needs a Principal cannot accidentally be
called without one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    """Coarse role, used for OPA decisions and metric labels (low cardinality)."""

    EMPLOYEE = "employee"
    MANAGER = "manager"
    FINANCE = "finance"
    AUDITOR = "auditor"
    ADMIN = "admin"


@dataclass(frozen=True, slots=True)
class Principal:
    user_id: str
    tenant_id: str
    role: Role
    department_id: str | None = None
    scopes: frozenset[str] = frozenset()

    @property
    def fga_user(self) -> str:
        """OpenFGA subject string, e.g. `user:u-1234`."""
        return f"user:{self.user_id}"
