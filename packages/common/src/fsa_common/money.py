"""Money as a value object.

Hard rule (CLAUDE.md): **money is never a float**. A claim is an integer number of
minor units (paise, cents) plus an ISO-4217 currency code. Two amounts in different
currencies are not comparable and not addable without an explicit rate *and* the date
that rate was observed — because an expense dated March must be converted at March's
rate, not today's.

Why a value object rather than a bare int:
  * it makes "add 500 to a Money" a type error instead of a silent currency bug;
  * it puts rounding in exactly one place, so the same rule applies everywhere;
  * it gives the ORM a single conversion point (BIGINT column + CHAR(3) column).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Final, Self

_CURRENCY_RE: Final = re.compile(r"^[A-Z]{3}$")

# ISO-4217 exponents. Most currencies use 2; a few (JPY, KRW) use 0. Extend as needed
# — an unknown code is rejected rather than assumed, because assuming 2 for JPY turns
# ¥1,000 into ¥10.00.
_EXPONENTS: Final[dict[str, int]] = {
    "INR": 2,
    "USD": 2,
    "EUR": 2,
    "GBP": 2,
    "AED": 2,
    "SGD": 2,
    "AUD": 2,
    "CAD": 2,
    "JPY": 0,
    "KRW": 0,
}


class CurrencyMismatchError(ValueError):
    """Raised when two Money values of different currencies are combined."""


class UnknownCurrencyError(ValueError):
    """Raised for a code that is not ISO-4217-shaped or not in the exponent table."""


def exponent_for(currency: str) -> int:
    """Minor units per major unit, as a power of ten. INR -> 2 (100 paise = ₹1)."""
    if not _CURRENCY_RE.match(currency):
        raise UnknownCurrencyError(f"not an ISO-4217 code: {currency!r}")
    try:
        return _EXPONENTS[currency]
    except KeyError as exc:
        raise UnknownCurrencyError(f"no exponent registered for {currency!r}") from exc


@dataclass(frozen=True, slots=True, order=False)
class Money:
    """An exact monetary amount. Immutable, hashable, safe to use as a dict key."""

    minor_units: int
    currency: str

    def __post_init__(self) -> None:
        if not isinstance(self.minor_units, int) or isinstance(self.minor_units, bool):
            raise TypeError("minor_units must be an int (no floats, no bools)")
        exponent_for(self.currency)  # validates, raises on unknown

    # ── construction ────────────────────────────────────────────────────────
    @classmethod
    def zero(cls, currency: str) -> Self:
        return cls(0, currency)

    @classmethod
    def from_major(cls, amount: Decimal | int | str, currency: str) -> Self:
        """Build from a human-facing amount: Money.from_major("1234.50", "INR")."""
        if not isinstance(amount, Decimal | int | str) or isinstance(amount, bool):
            raise TypeError("refusing to build Money from a float; pass Decimal, int or str")
        scaled = Decimal(amount) * (Decimal(10) ** exponent_for(currency))
        return cls(int(scaled.quantize(Decimal(1), rounding=ROUND_HALF_EVEN)), currency)

    # ── arithmetic ──────────────────────────────────────────────────────────
    def _check(self, other: Money) -> None:
        if self.currency != other.currency:
            raise CurrencyMismatchError(f"{self.currency} vs {other.currency}")

    def __add__(self, other: Money) -> Money:
        self._check(other)
        return Money(self.minor_units + other.minor_units, self.currency)

    def __sub__(self, other: Money) -> Money:
        self._check(other)
        return Money(self.minor_units - other.minor_units, self.currency)

    def __neg__(self) -> Money:
        return Money(-self.minor_units, self.currency)

    def __mul__(self, factor: int | Decimal) -> Money:
        """Scale by a count or a rate. Banker's rounding, applied once, at the end."""
        if not isinstance(factor, Decimal | int) or isinstance(factor, bool):
            raise TypeError("refusing to scale Money by a float; pass int or Decimal")
        scaled = Decimal(self.minor_units) * Decimal(factor)
        return Money(int(scaled.quantize(Decimal(1), rounding=ROUND_HALF_EVEN)), self.currency)

    __rmul__ = __mul__

    def __lt__(self, other: Money) -> bool:
        self._check(other)
        return self.minor_units < other.minor_units

    def __le__(self, other: Money) -> bool:
        self._check(other)
        return self.minor_units <= other.minor_units

    def __gt__(self, other: Money) -> bool:
        self._check(other)
        return self.minor_units > other.minor_units

    def __ge__(self, other: Money) -> bool:
        self._check(other)
        return self.minor_units >= other.minor_units

    # ── domain helpers ──────────────────────────────────────────────────────
    def allocate(self, weights: list[int]) -> list[Money]:
        """Split without losing a paisa. Remainder goes to the earliest buckets.

        Splitting a ₹100 dinner three ways as 33.33 + 33.33 + 33.33 loses a paisa, and
        losing paise is how ledgers stop balancing.
        """
        if not weights or any(w < 0 for w in weights) or sum(weights) == 0:
            raise ValueError("weights must be non-empty, non-negative and not all zero")
        total_weight = sum(weights)
        shares = [self.minor_units * w // total_weight for w in weights]
        remainder = self.minor_units - sum(shares)
        for i in range(abs(remainder)):
            shares[i % len(shares)] += 1 if remainder > 0 else -1
        return [Money(s, self.currency) for s in shares]

    def convert(self, *, to: str, rate: Decimal, as_of: date) -> Money:
        """Convert at an explicit rate observed on an explicit date.

        `as_of` is required and unused in the arithmetic on purpose: it forces the
        caller to have thought about *which day's* rate they are applying, and it is
        what the audit log records. See ARCHITECTURE.md §3.
        """
        if not isinstance(rate, Decimal):
            raise TypeError("refusing to convert at a float rate; pass Decimal")
        if rate <= 0:
            raise ValueError("rate must be positive")
        del as_of  # captured by callers for the audit trail; not part of the maths
        major = Decimal(self.minor_units) / (Decimal(10) ** exponent_for(self.currency))
        return Money.from_major(major * rate, to)

    # ── presentation ────────────────────────────────────────────────────────
    def as_decimal(self) -> Decimal:
        return Decimal(self.minor_units) / (Decimal(10) ** exponent_for(self.currency))

    def __str__(self) -> str:
        exp = exponent_for(self.currency)
        return f"{self.as_decimal():.{exp}f} {self.currency}"

    def __repr__(self) -> str:
        return f"Money({self.minor_units}, {self.currency!r})"
