"""The contract for `fsa_sim.world.spend` — your Step 1.

These tests are **skipped while the functions still raise NotImplementedError** and
start running the moment you implement them. Nothing to uncomment: write the code,
run `make test`, and the skips turn into passes (or, more usefully at first, failures).

Read the assertions before you write the implementation. They are the spec.
"""

from __future__ import annotations

from collections import Counter
from datetime import date

import numpy as np
import pytest

from fsa_sim.world.config import TENANT_SHAPES, WorldConfig
from fsa_sim.world.entities import Category, ExpenseRecord, SpendPersona, Tenant
from fsa_sim.world.org import build_org
from fsa_sim.world.spend import (
    generate_baseline_expenses,
    months_between,
    sample_amount_minor,
)
from fsa_sim.world.vendors import build_vendors

TENANT = Tenant(tenant_id="t00", name="Acme", currency="INR", shape="startup")
CONFIG = WorldConfig(start_date=date(2026, 1, 1), end_date=date(2026, 6, 30), seed=7)


def _raises_not_implemented(fn, *args, **kwargs) -> bool:  # type: ignore[no-untyped-def]
    try:
        fn(*args, **kwargs)
    except NotImplementedError:
        return True
    except Exception:  # implemented, but blew up — let the real tests report it
        return False
    return False


def _world_inputs():  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(5)
    _, users = build_org(TENANT, TENANT_SHAPES[0], rng, world_start=CONFIG.start_date)
    vendors = build_vendors(TENANT, rng, world_start=CONFIG.start_date)
    return users, vendors


_AMOUNT_TODO = _raises_not_implemented(
    sample_amount_minor, Category.MEALS, 3, np.random.default_rng(0)
)
_USERS, _VENDORS = _world_inputs()
_GENERATE_TODO = _raises_not_implemented(
    generate_baseline_expenses,
    _USERS,
    _VENDORS,
    CONFIG,
    np.random.default_rng(0),
    tenant_currency="INR",
)

_SKIP_A = pytest.mark.skipif(_AMOUNT_TODO, reason="Step 1a: implement sample_amount_minor")
_SKIP_B = pytest.mark.skipif(_GENERATE_TODO, reason="Step 1b: implement generate_baseline_expenses")


@_SKIP_A
class TestSampleAmount:
    def test_returns_a_positive_int(self, rng: np.random.Generator) -> None:
        for category in Category:
            value = sample_amount_minor(category, 3, rng)
            assert isinstance(value, int) and not isinstance(value, bool)
            assert value > 0

    def test_is_deterministic_given_the_rng(self) -> None:
        a = sample_amount_minor(Category.MEALS, 3, np.random.default_rng(11))
        b = sample_amount_minor(Category.MEALS, 3, np.random.default_rng(11))
        assert a == b

    def test_seniority_raises_the_typical_claim(self) -> None:
        """Not every draw — a lognormal overlaps heavily. The *median of many* draws
        must move, which is the statistically honest version of the assertion."""
        rng = np.random.default_rng(3)
        junior = np.median([sample_amount_minor(Category.MEALS, 1, rng) for _ in range(400)])
        senior = np.median([sample_amount_minor(Category.MEALS, 6, rng) for _ in range(400)])
        assert senior > junior * 2

    def test_distribution_is_right_skewed(self) -> None:
        """Expense amounts are lognormal-ish: mean well above median, long right tail.
        A symmetric distribution here would make the 'unusually large claim' feature
        meaningless."""
        rng = np.random.default_rng(4)
        draws = np.array([sample_amount_minor(Category.LODGING, 3, rng) for _ in range(2000)])
        assert draws.mean() > np.median(draws) * 1.1
        assert draws.max() > np.median(draws) * 4

    def test_some_amounts_are_suspiciously_round(self) -> None:
        """The honest world must contain round numbers too, or 'round amount' becomes
        a perfect fraud giveaway and the model learns the generator."""
        rng = np.random.default_rng(6)
        draws = [sample_amount_minor(Category.SOFTWARE, 3, rng) for _ in range(2000)]
        round_share = sum(1 for d in draws if d % 10_000 == 0) / len(draws)
        assert 0.01 < round_share < 0.60


@_SKIP_B
class TestGenerateBaseline:
    @pytest.fixture(scope="class")
    def expenses(self) -> list[ExpenseRecord]:
        return generate_baseline_expenses(
            _USERS, _VENDORS, CONFIG, np.random.default_rng(21), tenant_currency="INR"
        )

    def test_produces_a_plausible_volume(self, expenses: list[ExpenseRecord]) -> None:
        assert len(expenses) > len(_USERS)  # 6 months, everyone claims something

    def test_expense_ids_are_unique(self, expenses: list[ExpenseRecord]) -> None:
        ids = [e.expense_id for e in expenses]
        assert len(ids) == len(set(ids))

    def test_dates_are_inside_the_world_window(self, expenses: list[ExpenseRecord]) -> None:
        assert all(CONFIG.start_date <= e.transaction_date <= CONFIG.end_date for e in expenses)

    def test_nobody_claims_before_they_joined(self, expenses: list[ExpenseRecord]) -> None:
        joined = {u.user_id: u.joined_on for u in _USERS}
        assert all(e.transaction_date >= joined[e.user_id] for e in expenses)

    def test_submission_never_precedes_the_transaction(self, expenses: list[ExpenseRecord]) -> None:
        """Time only runs forwards. `days_since_transaction` is a feature, and a
        negative value there would be a leak dressed as a bug."""
        assert all(e.submitted_at.date() >= e.transaction_date for e in expenses)

    def test_amounts_and_currency_are_well_formed(self, expenses: list[ExpenseRecord]) -> None:
        assert all(isinstance(e.amount_minor, int) and e.amount_minor > 0 for e in expenses)
        assert {e.currency for e in expenses} == {"INR"}

    def test_vendor_matches_the_claim_category(self, expenses: list[ExpenseRecord]) -> None:
        vendor_category = {v.vendor_id: v.category for v in _VENDORS}
        assert all(vendor_category[e.vendor_id] == e.category for e in expenses)

    def test_personas_produce_different_claim_rates(self, expenses: list[ExpenseRecord]) -> None:
        persona = {u.user_id: u.persona for u in _USERS}
        counts: Counter[SpendPersona] = Counter(persona[e.user_id] for e in expenses)
        people: Counter[SpendPersona] = Counter(u.persona for u in _USERS)
        per_traveller = counts[SpendPersona.FREQUENT_TRAVELLER] / max(
            people[SpendPersona.FREQUENT_TRAVELLER], 1
        )
        per_desk = counts[SpendPersona.DESK_BOUND] / max(people[SpendPersona.DESK_BOUND], 1)
        assert per_traveller > per_desk * 2

    def test_weekends_are_a_minority(self, expenses: list[ExpenseRecord]) -> None:
        weekend = sum(1 for e in expenses if e.transaction_date.weekday() >= 5)
        assert weekend / len(expenses) < 0.30

    def test_spend_is_seasonal(self, expenses: list[ExpenseRecord]) -> None:
        """March (conference season, 1.30x) should out-spend May (0.90x)."""
        by_month: Counter[int] = Counter(e.transaction_date.month for e in expenses)
        assert by_month[3] > by_month[5]

    def test_regeneration_is_reproducible(self) -> None:
        a = generate_baseline_expenses(
            _USERS, _VENDORS, CONFIG, np.random.default_rng(21), tenant_currency="INR"
        )
        b = generate_baseline_expenses(
            _USERS, _VENDORS, CONFIG, np.random.default_rng(21), tenant_currency="INR"
        )
        assert [e.expense_id for e in a] == [e.expense_id for e in b]
        assert [e.amount_minor for e in a] == [e.amount_minor for e in b]


def test_months_between_covers_the_range() -> None:
    months = months_between(date(2026, 1, 15), date(2026, 4, 2))
    assert months == [date(2026, 1, 1), date(2026, 2, 1), date(2026, 3, 1), date(2026, 4, 1)]
