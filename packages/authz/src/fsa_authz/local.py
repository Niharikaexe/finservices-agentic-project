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
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import ClassVar

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
        # Every one of these is a SET, matching OpenFGA. They were single-valued
        # dicts, so a second `scope` tuple silently discarded the first — a document
        # co-scoped to two departments (an entirely reasonable thing to want) behaved
        # differently here than in production, and which department won depended on
        # tuple insertion order.
        self._direct: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._parent: dict[str, set[str]] = defaultdict(set)
        self._dept_of_expense: dict[str, set[str]] = defaultdict(set)
        self._tenant_of: dict[str, set[str]] = defaultdict(set)
        self._scope_of_doc: dict[str, set[str]] = defaultdict(set)
        self._tenant_scope_of_doc: dict[str, set[str]] = defaultdict(set)

        for t in tuples:
            structural = self._STRUCTURAL.get(t.relation)
            if structural is not None:
                mapping: dict[str, set[str]] = getattr(self, structural)
                mapping[t.object].add(t.user)
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
        # OR across every parent — a department can have more than one in the model.
        return any(
            self._manager_chain(user, parent, _depth + 1)
            for parent in self._parent.get(department, ())
        )

    def _department_viewer(self, user: str, department: str) -> bool:
        """`member or manager_chain`."""
        return self._has_direct("member", department, user) or self._manager_chain(user, department)

    def _tenant_relation(self, user: str, obj: str, relation: str) -> bool:
        return any(
            self._has_direct(relation, tenant, user) for tenant in self._tenant_of.get(obj, ())
        )

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
            departments = self._dept_of_expense.get(object, set())
            is_owner = self._has_direct("owner", object, user)
            is_approver = any(self._has_direct("manager", d, user) for d in departments)
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
            if any(
                self._department_viewer(user, scope) for scope in self._scope_of_doc.get(object, ())
            ):
                return True
            if any(
                self._has_direct("employee", tenant_scope, user)
                for tenant_scope in self._tenant_scope_of_doc.get(object, ())
            ):
                return True
            return self._tenant_relation(user, object, "auditor") or self._tenant_relation(
                user, object, "finance"
            )

        if type_ == "tenant":
            return self._has_direct(relation, object, user)

        return False

    #: Object types this store knows about. An unknown type used to yield an empty
    #: candidate set and therefore an empty result — indistinguishable from a correct
    #: denial, which is the exact anti-pattern `store.py` argues against.
    KNOWN_TYPES = frozenset({"department", "expense", "policy_document", "tenant"})

    def list_objects(self, *, user: str, relation: str, type: str) -> list[str]:
        self._require_available()
        if relation not in self.KNOWN_RELATIONS:
            raise FailClosedError("unknown relation", relation=relation)
        if type not in self.KNOWN_TYPES:
            raise FailClosedError("unknown object type", type=type)
        candidates = sorted(self._objects_by_type.get(type, ()))
        return [
            obj.split(":", 1)[1]
            for obj in candidates
            if self.check(user=user, relation=relation, object=obj)
        ]

    # ── mutation ────────────────────────────────────────────────────────────
    #: Relations held in the single-valued structural maps rather than in `_direct`.
    _STRUCTURAL: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "parent": "_parent",
            "department": "_dept_of_expense",
            "tenant": "_tenant_of",
            "scope": "_scope_of_doc",
            "tenant_scope": "_tenant_scope_of_doc",
        }
    )

    def _apply(self, tuple_: RelationTuple, *, add: bool) -> None:
        """Add or remove one tuple, routing by relation exactly as `__init__` does.

        An earlier version only touched `_direct`, so revoking a `scope` or
        `tenant_scope` tuple was a **silent no-op that returned success** — the
        operational equivalent of "un-publish this leaked policy document" reporting
        done while the document stayed readable. Grants of the same relations were
        the mirror no-op. Both now route correctly, and an unknown relation raises
        rather than falling through.
        """
        attribute = self._STRUCTURAL.get(tuple_.relation)
        if attribute is not None:
            mapping: dict[str, set[str]] = getattr(self, attribute)
            if add:
                mapping[tuple_.object].add(tuple_.user)
                self._objects_by_type[tuple_.object.split(":", 1)[0]].add(tuple_.object)
            else:
                mapping.get(tuple_.object, set()).discard(tuple_.user)
            return

        if tuple_.relation not in self.KNOWN_RELATIONS:
            raise FailClosedError(
                "refusing to mutate an unknown relation", relation=tuple_.relation
            )
        if add:
            self._direct[(tuple_.relation, tuple_.object)].add(tuple_.user)
        else:
            self._direct.get((tuple_.relation, tuple_.object), set()).discard(tuple_.user)

    def revoke(self, tuple_: RelationTuple) -> None:
        """Remove a tuple. Takes effect on the very next query — this store holds no
        cache, so there is no staleness window. Note that this is a property of *this*
        implementation; see `OpenFGAStore` for the consistency setting that makes the
        same claim true against a real server."""
        self._apply(tuple_, add=False)

    def grant(self, tuple_: RelationTuple) -> None:
        self._apply(tuple_, add=True)
