"""Money is the one place a rounding bug becomes an audit finding, so it gets
property-based tests, not just examples."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from fsa_common import CurrencyMismatchError, Money, UnknownCurrencyError


def test_construction_from_major_units() -> None:
    assert Money.from_major("1234.50", "INR") == Money(123_450, "INR")
    assert Money.from_major(Decimal("1000"), "JPY") == Money(1000, "JPY")


def test_floats_are_refused_everywhere() -> None:
    with pytest.raises(TypeError):
        Money(100.5, "INR")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Money.from_major(12.5, "INR")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Money(100, "INR") * 1.5  # type: ignore[operator]


def test_unknown_currency_is_rejected_not_assumed() -> None:
    with pytest.raises(UnknownCurrencyError):
        Money(100, "XYZ")
    with pytest.raises(UnknownCurrencyError):
        Money(100, "inr")


def test_cross_currency_arithmetic_is_an_error() -> None:
    with pytest.raises(CurrencyMismatchError):
        Money(100, "INR") + Money(100, "USD")
    with pytest.raises(CurrencyMismatchError):
        _ = Money(100, "INR") < Money(100, "USD")


def test_conversion_requires_a_rate_and_a_date() -> None:
    inr = Money(1_000_00, "INR")  # ₹1,000.00
    usd = inr.convert(to="USD", rate=Decimal("0.012"), as_of=date(2026, 3, 1))
    assert usd == Money(1200, "USD")  # $12.00
    with pytest.raises(TypeError):
        inr.convert(to="USD", rate=0.012, as_of=date(2026, 3, 1))  # type: ignore[arg-type]


@given(
    total=st.integers(min_value=0, max_value=10**12),
    weights=st.lists(st.integers(min_value=1, max_value=50), min_size=1, max_size=8),
)
def test_allocate_never_loses_or_invents_a_paisa(total: int, weights: list[int]) -> None:
    parts = Money(total, "INR").allocate(weights)
    assert sum(p.minor_units for p in parts) == total
    assert len(parts) == len(weights)


@given(a=st.integers(-(10**9), 10**9), b=st.integers(-(10**9), 10**9))
def test_addition_is_exact(a: int, b: int) -> None:
    assert (Money(a, "INR") + Money(b, "INR")).minor_units == a + b


def test_str_respects_the_currency_exponent() -> None:
    assert str(Money(123_450, "INR")) == "1234.50 INR"
    assert str(Money(1000, "JPY")) == "1000 JPY"
