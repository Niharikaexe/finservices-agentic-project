"""fsa_authz — the trust plane client library.

Two systems, deliberately (ARCHITECTURE.md §2):
  OPA     coarse, request-level : "may this role call this endpoint / this tool?"
  OpenFGA fine, object-level    : "may user U read expense E?"

Implementations land in M1. What is defined now is the `Principal` — the caller
identity that every retrieval and every tool call threads through — because the
simulator and the ML layer both need to talk about who did what.
"""

from fsa_authz.principal import Principal, Role

__all__ = ["Principal", "Role"]
