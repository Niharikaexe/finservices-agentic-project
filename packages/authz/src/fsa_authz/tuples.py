"""Relationship tuples — the data behind the authorisation model.

OpenFGA stores facts, not rules. `user:u-14 is member of department:d-03` is a tuple;
"a manager of a parent department can view the child" is a rule in the model. Keeping
those apart is the whole point: adding a sub-department writes three tuples and
visibility is correct, with no application code changed and no migration.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RelationTuple:
    """`user#relation@object`, in OpenFGA's vocabulary.

    `user` is a subject string — `user:u-14`, or a userset like
    `department:d-03#member`. `object` is `type:id`.
    """

    user: str
    relation: str
    object: str

    def __str__(self) -> str:
        return f"{self.user} {self.relation} {self.object}"


def obj(type_: str, id_: str) -> str:
    return f"{type_}:{id_}"


def build_tuples(
    *,
    departments: Iterable[tuple[str, str, str | None]],
    users: Iterable[tuple[str, str, str, str]],
    policy_documents: Iterable[tuple[str, str, str | None]],
) -> list[RelationTuple]:
    """Materialise every tuple for one world.

    Arguments are deliberately plain tuples rather than the simulator's dataclasses:
    `fsa_authz` sits below `fsa_sim` in the layering and must not import it. The
    seeding script does the translation.

    departments       (department_id, tenant_id, parent_department_id | None)
    users             (user_id, tenant_id, department_id, role)
    policy_documents  (document_id, tenant_id, scope_department_id | None)
                      — a None scope means tenant-wide
    """
    out: list[RelationTuple] = []

    for department_id, tenant_id, parent_id in departments:
        out.append(
            RelationTuple(obj("tenant", tenant_id), "tenant", obj("department", department_id))
        )
        if parent_id:
            out.append(
                RelationTuple(
                    obj("department", parent_id), "parent", obj("department", department_id)
                )
            )

    for user_id, tenant_id, department_id, role in users:
        subject = obj("user", user_id)
        # Every user is an employee of their tenant — this is what makes a tenant-wide
        # policy document readable without writing a tuple per department.
        out.append(RelationTuple(subject, "employee", obj("tenant", tenant_id)))

        if role == "manager":
            out.append(RelationTuple(subject, "manager", obj("department", department_id)))
        elif role in {"auditor", "finance", "admin"}:
            # Deliberately NOT a department member. The auditor reads through the
            # tenant relation; the admin reads nothing. Making them members would
            # quietly grant department-scoped access and is the classic mistake.
            out.append(RelationTuple(subject, role, obj("tenant", tenant_id)))
        else:
            out.append(RelationTuple(subject, "member", obj("department", department_id)))

    for document_id, tenant_id, scope_department_id in policy_documents:
        target = obj("policy_document", document_id)
        out.append(RelationTuple(obj("tenant", tenant_id), "tenant", target))
        if scope_department_id:
            out.append(RelationTuple(obj("department", scope_department_id), "scope", target))
        else:
            out.append(RelationTuple(obj("tenant", tenant_id), "tenant_scope", target))

    return out


def expense_tuples(
    expense_id: str, tenant_id: str, owner_user_id: str, department_id: str
) -> Iterator[RelationTuple]:
    """Tuples written when an expense is created. Three, every time, in the same
    transaction as the row — an expense whose tuples failed to write is invisible to
    its own owner, which is the correct failure direction."""
    target = obj("expense", expense_id)
    yield RelationTuple(obj("tenant", tenant_id), "tenant", target)
    yield RelationTuple(obj("user", owner_user_id), "owner", target)
    yield RelationTuple(obj("department", department_id), "department", target)
