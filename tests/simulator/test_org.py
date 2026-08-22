"""The org tree is the substrate the M1 ACL suite runs on. If the tree is wrong —
cycles, orphan managers, cross-tenant edges — then a green authz suite means nothing.
So it gets tested as a graph, not just as a row count."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from fsa_sim.world.config import TENANT_SHAPES, TenantShape
from fsa_sim.world.entities import Tenant
from fsa_sim.world.org import build_org, manager_chain

WORLD_START = date(2026, 1, 1)


@pytest.fixture
def tenant() -> Tenant:
    return Tenant(tenant_id="t00", name="Acme", currency="INR", shape="startup")


@pytest.mark.parametrize("shape", TENANT_SHAPES, ids=lambda s: s.name)
def test_org_is_a_forest_without_cycles(
    tenant: Tenant, shape: TenantShape, rng: np.random.Generator
) -> None:
    _, users = build_org(tenant, shape, rng, world_start=WORLD_START)
    for user in users:
        chain = manager_chain(users, user.user_id)
        assert user.user_id not in chain, f"{user.user_id} manages itself, transitively"
        assert len(chain) == len(set(chain))


def test_every_manager_and_department_reference_resolves(
    tenant: Tenant, rng: np.random.Generator
) -> None:
    departments, users = build_org(tenant, TENANT_SHAPES[1], rng, world_start=WORLD_START)
    dept_ids = {d.department_id for d in departments}
    user_ids = {u.user_id for u in users}
    for user in users:
        assert user.department_id in dept_ids
        assert user.manager_id is None or user.manager_id in user_ids
    for dept in departments:
        assert dept.parent_id is None or dept.parent_id in dept_ids


def test_departments_respect_max_depth(tenant: Tenant, rng: np.random.Generator) -> None:
    shape = TENANT_SHAPES[1]
    departments, _ = build_org(tenant, shape, rng, world_start=WORLD_START)
    by_id = {d.department_id: d for d in departments}

    def depth(dept_id: str) -> int:
        d = by_id[dept_id]
        return 1 if d.parent_id is None else 1 + depth(d.parent_id)

    assert max(depth(d.department_id) for d in departments) <= shape.max_depth


def test_tenant_level_personas_exist(tenant: Tenant, rng: np.random.Generator) -> None:
    """Auditor and admin are the two personas a naive role check gets wrong (§2), so
    every generated tenant must contain one of each for the ACL suite to probe."""
    _, users = build_org(tenant, TENANT_SHAPES[0], rng, world_start=WORLD_START)
    roles = {u.role for u in users}
    assert {"auditor", "admin", "finance", "manager", "employee"} <= roles


def test_generation_is_deterministic(tenant: Tenant) -> None:
    a = build_org(tenant, TENANT_SHAPES[0], np.random.default_rng(99), world_start=WORLD_START)
    b = build_org(tenant, TENANT_SHAPES[0], np.random.default_rng(99), world_start=WORLD_START)
    assert a == b


def test_no_cross_tenant_leakage_in_ids(tenant: Tenant, rng: np.random.Generator) -> None:
    departments, users = build_org(tenant, TENANT_SHAPES[0], rng, world_start=WORLD_START)
    assert all(d.tenant_id == tenant.tenant_id for d in departments)
    assert all(u.tenant_id == tenant.tenant_id for u in users)
    assert all(u.user_id.startswith(tenant.tenant_id) for u in users)
