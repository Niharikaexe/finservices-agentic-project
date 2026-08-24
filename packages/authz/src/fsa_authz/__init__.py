"""fsa_authz — the trust plane client library.

Two systems, deliberately (ARCHITECTURE.md §2):
  OPA     coarse, request-level : "may this role call this endpoint / this tool?"
  OpenFGA fine, object-level    : "may user U read expense E / policy document D?"

Everything here fails closed. If the store cannot answer, the answer is no, and the
caller gets a `FailClosedError` rather than a `False` that would be indistinguishable
from a legitimate denial.
"""

from fsa_authz.local import LocalAuthorizationStore
from fsa_authz.openfga import OpenFGAStore
from fsa_authz.principal import Principal, Role
from fsa_authz.store import AuthorizationStore
from fsa_authz.tuples import RelationTuple, build_tuples, expense_tuples, obj

__all__ = [
    "AuthorizationStore",
    "LocalAuthorizationStore",
    "OpenFGAStore",
    "Principal",
    "RelationTuple",
    "Role",
    "build_tuples",
    "expense_tuples",
    "obj",
]
