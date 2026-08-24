"""In-process evaluation of the authorisation model.

This exists so the ACL isolation suite runs in CI on every commit without a server.
That matters more than it sounds: an authz test suite that only runs when docker is
healthy is a suite that gets skipped, and a skipped authz suite is worse than none
because the badge still says green.

The evaluator implements exactly the rules in `infra/openfga/model.fga`. Where the
model says `viewer from parent`, this walks the parent chain. Where it says
`but not owner`, this subtracts. If you change the model file, change this, and the
parity test will tell you if you got it wrong.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from fsa_authz.tuples import RelationTuple
from fsa_common import FailClosedError


class LocalAuthorizationStore:
    """Evaluates the Argus model over an in-memory tuple set."""

    #: Relations a caller is allowed to ask about. An unknown relation is a bug in the
    #: caller, and answering `False` would hide it — so we raise instead.
    KNOWN_RELATIONS = frozenset(
        {
            "member",
            "manager",
            "viewer",
            "owner",
            "approver",
            "can_approve",
            "reader",
            "employee",
            "auditor",
            "finance",
            "admin",
        }
    )

    def __init__(self, tuples: Iterable[RelationTuple], *, available: bool = True) -> None:
        self._available = available
        self._direct: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._parent: dict[str, str] = {}
        self._dept_of_expense: dict[str, str] = {}
        self._tenant_of: dict[str, str] = {}
        self._scope_of_doc: dict[str, str] = {}
        self._tenant_scope_of_doc: dict[str, str] = {}

        for t in tuples:
            if t.relation == "parent":
                self._parent[t.object] = t.user
            elif t.relation == "department":
                self._dept_of_expense[t.object] = t.user
            elif t.relation == "tenant":
                self._tenant_of[t.object] = t.user
            elif t.relation == "scope":
                self._scope_of_doc[t.object] = t.user
            elif t.relation == "tenant_scope":
                self._tenant_scope_of_doc[t.object] = t.user
            else:
                self._direct[(t.relation, t.object)].add(t.user)

        self._objects_by_type: dict[str, set[str]] = defaultdict(set)
        for key in (*self._tenant_of, *self._scope_of_doc, *self._tenant_scope_of_doc):
            self._objects_by_type[key.split(":", 1)[0]].add(key)

    # ── availability ────────────────────────────────────────────────────────
    def set_available(self, available: bool) -> None:
        """Used by the chaos test to prove the system denies when authz is down."""
        self._available = available

    def _require_available(self) -> None:
        if not self._available:
            raise FailClosedError("authorisation store unavailable; denying", store="local")

    # ── primitives ──────────────────────────────────────────────────────────
    def _has_direct(self, relation: str, obj: str, user: str) -> bool:
        return user in self._direct.get((relation, obj), ())

    def _manager_chain(self, user: str, department: str, _depth: int = 0) -> bool:
        """`manager or manager_chain from parent` — recursive over MANAGERS ONLY.

        The recursion carries managership up the tree, not viewership down it. Make it
        `viewer from parent` and an ordinary member of a parent department can read a
        child department's addendum, which is over-permissioning that reads as correct
        until a probe finds it. See ADR 0006.

        The depth guard is not defensive padding — a malformed tuple set can contain a
        parent cycle, and an unbounded walk would hang the request rather than deny it.
        Failing closed means bounding the work too.
        """
        if _depth > 32:
            raise FailClosedError(
                "department parent chain too deep or cyclic", department=department
            )
        if self._has_direct("manager", department, user):
            return True
        parent = self._parent.get(department)
        if parent is None:
            return False
        return self._manager_chain(user, parent, _depth + 1)

    def _department_viewer(self, user: str, department: str) -> bool:
        """`member or manager_chain`."""
        return self._has_direct("member", department, user) or self._manager_chain(user, department)

    def _tenant_relation(self, user: str, obj: str, relation: str) -> bool:
        tenant = self._tenant_of.get(obj)
        return tenant is not None and self._has_direct(relation, tenant, user)

    # ── the interface ───────────────────────────────────────────────────────
    def check(self, *, user: str, relation: str, object: str) -> bool:
        self._require_available()
        if relation not in self.KNOWN_RELATIONS:
            raise FailClosedError("unknown relation", relation=relation)

        type_ = object.split(":", 1)[0]

        if type_ == "department":
            if relation == "viewer":
                return self._department_viewer(user, object)
            if relation == "manager_chain":
                return self._manager_chain(user, object)
            return self._has_direct(relation, object, user)

        if type_ == "expense":
            department = self._dept_of_expense.get(object)
            is_owner = self._has_direct("owner", object, user)
            is_approver = department is not None and self._has_direct("manager", department, user)
            if relation == "owner":
                return is_owner
            if relation == "approver":
                return is_approver
            if relation == "can_approve":
                # `approver but not owner` — self-approval prevention lives here, in
                # the model, not in an application `if` that a new endpoint forgets.
                return is_approver and not is_owner
            if relation == "viewer":
                return (
                    is_owner
                    or is_approver
                    or self._tenant_relation(user, object, "auditor")
                    or self._tenant_relation(user, object, "finance")
                )
            return False

        if type_ == "policy_document":
            if relation != "reader":
                return False
            scope = self._scope_of_doc.get(object)
            if scope and self._department_viewer(user, scope):
                return True
            tenant_scope = self._tenant_scope_of_doc.get(object)
            if tenant_scope is not None and self._has_direct("employee", tenant_scope, user):
                return True
            return self._tenant_relation(user, object, "auditor") or self._tenant_relation(
                user, object, "finance"
            )

        if type_ == "tenant":
            return self._has_direct(relation, object, user)

        return False

    def list_objects(self, *, user: str, relation: str, type: str) -> list[str]:
        self._require_available()
        candidates = sorted(self._objects_by_type.get(type, ()))
        return [
            obj.split(":", 1)[1]
            for obj in candidates
            if self.check(user=user, relation=relation, object=obj)
        ]

    def revoke(self, tuple_: RelationTuple) -> None:
        """Remove a tuple. Takes effect on the very next query — there is no cache to
        go stale, which is the property `test_revocation_is_immediate` asserts."""
        self._direct.get((tuple_.relation, tuple_.object), set()).discard(tuple_.user)

    def grant(self, tuple_: RelationTuple) -> None:
        self._direct[(tuple_.relation, tuple_.object)].add(tuple_.user)
