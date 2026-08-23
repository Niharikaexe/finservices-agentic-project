"""World assembly: config + seed -> a complete `World`.

This is the orchestrator. It owns two things worth calling out:

**One RNG per tenant, derived from the world seed.** `np.random.default_rng(seed)`
gives a root generator; `root.spawn(n)` gives n independent child streams. Deriving a
child per tenant means adding a fourth tenant does not change the data of the first
three — which is what lets you grow the world without invalidating every experiment
you have already run. Sharing one stream across tenants would silently do the opposite.

**Budgets are derived from realised spend, not invented.** A budget set independently
of behaviour is either never breached (boring) or always breached (useless). Setting a
department's limit at a percentile of its own historical spend guarantees that budget
overruns are rare-but-real, which is what the forecaster in M5 has to predict.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from fsa_sim.adversarial import inject_fraud, inject_payloads
from fsa_sim.world.config import TenantShape, WorldConfig
from fsa_sim.world.entities import (
    Budget,
    Category,
    ExpenseRecord,
    FraudLabel,
    InjectionLabel,
    Tenant,
    World,
)
from fsa_sim.world.org import build_org
from fsa_sim.world.spend import generate_baseline_expenses
from fsa_sim.world.vendors import build_vendors


def _quarter(day: date) -> str:
    return f"{day.year}-Q{(day.month - 1) // 3 + 1}"


def derive_budgets(
    tenant: Tenant, expenses: list[ExpenseRecord], *, headroom: float = 1.15
) -> list[Budget]:
    """Per (department, category, quarter) limit = realised spend x headroom.

    With 15% headroom most quarters land comfortably inside budget and the noisy ones
    breach — a base rate of overruns in the low tens of percent, which is roughly what
    a real finance team sees and enough signal for the early-warning alert in M5.
    """
    totals: dict[tuple[str, Category, str], int] = {}
    for e in expenses:
        key = (e.department_id, e.category, _quarter(e.transaction_date))
        totals[key] = totals.get(key, 0) + e.amount_minor

    return [
        Budget(
            tenant_id=tenant.tenant_id,
            department_id=dept,
            category=category,
            period=period,
            limit_minor=int(total * headroom),
            currency=tenant.currency,
        )
        for (dept, category, period), total in sorted(
            totals.items(), key=lambda kv: (kv[0][0], kv[0][1].value, kv[0][2])
        )
    ]


def generate_tenant(
    shape: TenantShape,
    index: int,
    config: WorldConfig,
    rng: np.random.Generator,
    *,
    with_spend: bool = True,
) -> World:
    """Everything for one tenant, in its own `World` fragment."""
    tenant = Tenant(
        tenant_id=f"t{index:02d}",
        name=f"{shape.name.title()} Co {index}",
        currency=shape.currency,
        shape=shape.name,
    )
    departments, users = build_org(tenant, shape, rng, world_start=config.start_date)
    vendors = build_vendors(tenant, rng, world_start=config.start_date)

    expenses: list[ExpenseRecord] = []
    budgets: list[Budget] = []
    fraud_labels: list[FraudLabel] = []
    injection_labels: list[InjectionLabel] = []

    if with_spend:
        # Order matters and is not arbitrary:
        #   1. honest spend           — the population
        #   2. fraud transformation   — mutates a subset of it (ADR 0004)
        #   3. injection payloads     — independent of fraud, so the two ground truths
        #                               do not correlate and let one model cheat at the
        #                               other's job
        #   4. budgets                — derived from *realised* spend, fraud included,
        #                               because a finance team budgets against what was
        #                               actually claimed, not against what was honest
        expenses = generate_baseline_expenses(
            users, vendors, config, rng, tenant_currency=tenant.currency
        )
        expenses, ghost_vendors, fraud_labels = inject_fraud(
            expenses, users, config.fraud, rng, world_start=config.start_date
        )
        vendors = [*vendors, *ghost_vendors]
        expenses, injection_labels = inject_payloads(expenses, config.injection_rate, rng)
        budgets = derive_budgets(tenant, expenses)

    return World(
        tenants=[tenant],
        departments=departments,
        users=users,
        vendors=vendors,
        budgets=budgets,
        expenses=expenses,
        fraud_labels=fraud_labels,
        injection_labels=injection_labels,
    )


def merge(fragments: list[World]) -> World:
    """Concatenate per-tenant fragments into one world."""
    merged = World()
    for f in fragments:
        merged.tenants.extend(f.tenants)
        merged.departments.extend(f.departments)
        merged.users.extend(f.users)
        merged.vendors.extend(f.vendors)
        merged.budgets.extend(f.budgets)
        merged.expenses.extend(f.expenses)
        merged.fraud_labels.extend(f.fraud_labels)
        merged.injection_labels.extend(f.injection_labels)
    return merged


def generate_world(config: WorldConfig, *, with_spend: bool = True) -> World:
    """The entry point. `(config, config.seed)` fully determines the result."""
    root = np.random.default_rng(config.seed)
    streams = root.spawn(len(config.shapes))
    fragments = [
        generate_tenant(shape, i, config, streams[i], with_spend=with_spend)
        for i, shape in enumerate(config.shapes)
    ]
    return merge(fragments)
