"""Org tree generation — worked example.

Read this one closely: it is the reference for the style the rest of the simulator
should follow.

Three properties every generator here must have:

  1. **Deterministic given an RNG.** Every function takes `rng: numpy.random.Generator`
     and calls no global random. `np.random.seed()` is banned — global seeding makes
     two generators that run in a different order produce different worlds, and then
     "reproducible dataset" quietly stops being true.
  2. **Pure.** It returns new objects; it mutates nothing it was handed. Makes it
     testable without fixtures and safe to run per-tenant in parallel later.
  3. **Structurally valid by construction**, not by post-hoc repair. The manager chain
     is built as a tree, so it cannot contain a cycle — rather than being built loosely
     and then checked for cycles.

Why the org tree matters beyond realism: manager visibility in OpenFGA is
`manager from parent` (ARCHITECTURE.md §2), so a deep tree with real chains is what
makes the M1 ACL test suite meaningful. A flat org would let a broken authz rule pass.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from fsa_sim.world.config import TenantShape
from fsa_sim.world.entities import Department, SpendPersona, Tenant, User

DEPARTMENT_NAMES: tuple[str, ...] = (
    "Engineering",
    "Sales",
    "Marketing",
    "Finance",
    "People",
    "Operations",
    "Customer Success",
    "Legal",
    "Product",
    "Data",
    "IT",
    "Facilities",
)

# Which persona a department's people tend to have. Weights, not hard assignments —
# every department has some desk-bound people and some travellers.
DEPARTMENT_PERSONA_BIAS: dict[str, dict[SpendPersona, float]] = {
    "Sales": {
        SpendPersona.CLIENT_FACING: 6.0,
        SpendPersona.FREQUENT_TRAVELLER: 3.0,
        SpendPersona.DESK_BOUND: 1.0,
    },
    "Marketing": {
        SpendPersona.CLIENT_FACING: 3.0,
        SpendPersona.FREQUENT_TRAVELLER: 2.0,
        SpendPersona.DESK_BOUND: 3.0,
    },
    "Engineering": {SpendPersona.DESK_BOUND: 8.0, SpendPersona.FREQUENT_TRAVELLER: 1.0},
    "Operations": {SpendPersona.FREQUENT_TRAVELLER: 4.0, SpendPersona.DESK_BOUND: 4.0},
}
_DEFAULT_PERSONA_BIAS: dict[SpendPersona, float] = {
    SpendPersona.DESK_BOUND: 6.0,
    SpendPersona.CLIENT_FACING: 2.0,
    SpendPersona.FREQUENT_TRAVELLER: 2.0,
}


def _choose(rng: np.random.Generator, weights: dict[SpendPersona, float]) -> SpendPersona:
    keys = list(weights)
    probs = np.array([weights[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=probs / probs.sum()))]


def build_departments(
    tenant: Tenant, shape: TenantShape, rng: np.random.Generator
) -> list[Department]:
    """Build a department forest of at most `shape.max_depth` levels.

    The first `ceil(n/3)` departments are roots; the rest attach to a random department
    that is not already at max depth. Attaching to a *random* eligible parent (rather
    than filling level by level) gives ragged, realistic trees — some deep branches,
    some shallow — which is exactly the shape that breaks naive authz rules.
    """
    names = list(DEPARTMENT_NAMES)
    rng.shuffle(names)
    n = min(shape.n_departments, len(names))

    departments: list[Department] = []
    depth_of: dict[str, int] = {}
    n_roots = max(1, -(-n // 3))  # ceil division

    for i in range(n):
        dept_id = f"{tenant.tenant_id}-dept-{i:02d}"
        if i < n_roots:
            parent_id = None
            depth = 1
        else:
            eligible = [d for d in departments if depth_of[d.department_id] < shape.max_depth]
            parent = (
                departments[int(rng.integers(len(departments)))]
                if not eligible
                else eligible[int(rng.integers(len(eligible)))]
            )
            parent_id = parent.department_id
            depth = depth_of[parent_id] + 1
        depth_of[dept_id] = depth
        departments.append(
            Department(
                department_id=dept_id,
                tenant_id=tenant.tenant_id,
                name=names[i],
                parent_id=parent_id,
            )
        )
    return departments


def build_users(
    tenant: Tenant,
    shape: TenantShape,
    departments: list[Department],
    rng: np.random.Generator,
    *,
    world_start: date,
) -> list[User]:
    """Populate departments with a real reporting chain.

    Per department: one head (grade 5). Members are chunked by `span_of_control`; each
    chunk above the first gets its own manager (grade 4) reporting to the head. A
    department head reports to the head of its *parent* department, so the chain spans
    department boundaries — which is what `manager from parent` in the OpenFGA model
    actually walks.
    """
    users: list[User] = []
    head_of: dict[str, str] = {}
    seq = 0

    def next_id() -> str:
        nonlocal seq
        seq += 1
        return f"{tenant.tenant_id}-u-{seq:04d}"

    def joined(rng: np.random.Generator) -> date:
        # Tenure is right-skewed: lots of recent joiners, a long tail of veterans.
        days_ago = int(rng.exponential(500))
        return world_start - timedelta(days=min(days_ago, 3650))

    # Heads first, so a child department can point at its parent's head.
    for dept in departments:
        head_id = next_id()
        head_of[dept.department_id] = head_id

    per_dept = max(1, shape.n_employees // len(departments))

    for dept in departments:
        bias = DEPARTMENT_PERSONA_BIAS.get(dept.name, _DEFAULT_PERSONA_BIAS)
        parent_head = head_of.get(dept.parent_id) if dept.parent_id else None

        users.append(
            User(
                user_id=head_of[dept.department_id],
                tenant_id=tenant.tenant_id,
                department_id=dept.department_id,
                manager_id=parent_head,
                role="manager",
                grade=5,
                persona=_choose(rng, bias),
                joined_on=joined(rng),
            )
        )

        n_members = max(0, per_dept - 1)
        chunk_manager: str | None = None
        for i in range(n_members):
            if i % shape.span_of_control == 0:
                if i == 0:
                    chunk_manager = head_of[dept.department_id]
                else:
                    chunk_manager = next_id()
                    users.append(
                        User(
                            user_id=chunk_manager,
                            tenant_id=tenant.tenant_id,
                            department_id=dept.department_id,
                            manager_id=head_of[dept.department_id],
                            role="manager",
                            grade=4,
                            persona=_choose(rng, bias),
                            joined_on=joined(rng),
                        )
                    )
            join_date = joined(rng)
            # A very recent joiner has no spend history — the cold-start case that
            # employee-historical features cannot compute. Model it explicitly.
            recent = (world_start - join_date).days < 60
            users.append(
                User(
                    user_id=next_id(),
                    tenant_id=tenant.tenant_id,
                    department_id=dept.department_id,
                    manager_id=chunk_manager,
                    role="employee",
                    grade=int(rng.integers(1, 4)),
                    persona=SpendPersona.NEW_JOINER if recent else _choose(rng, bias),
                    joined_on=join_date,
                )
            )

    # Tenant-level personas. These two are the strongest tests of the authz layer
    # (§2): the auditor reads everything and writes nothing; the admin touches config
    # and no expense data at all. A naive role check gets both wrong.
    root_dept = departments[0].department_id
    for role, grade in (("finance", 4), ("auditor", 4), ("admin", 3)):
        users.append(
            User(
                user_id=next_id(),
                tenant_id=tenant.tenant_id,
                department_id=root_dept,
                manager_id=None,
                role=role,
                grade=grade,
                persona=SpendPersona.DESK_BOUND,
                joined_on=joined(rng),
            )
        )

    return users


def build_org(
    tenant: Tenant, shape: TenantShape, rng: np.random.Generator, *, world_start: date
) -> tuple[list[Department], list[User]]:
    """Convenience wrapper: departments then users, sharing one RNG stream."""
    departments = build_departments(tenant, shape, rng)
    users = build_users(tenant, shape, departments, rng, world_start=world_start)
    return departments, users


def manager_chain(users: list[User], user_id: str) -> list[str]:
    """Walk up the reporting line. Useful in tests and in the collusion typology."""
    by_id = {u.user_id: u for u in users}
    chain: list[str] = []
    seen: set[str] = set()
    current = by_id[user_id].manager_id
    while current is not None and current not in seen:
        chain.append(current)
        seen.add(current)
        current = by_id[current].manager_id
    return chain
