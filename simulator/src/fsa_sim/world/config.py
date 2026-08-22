"""World configuration — every knob in one place, no magic numbers in the generators.

Two reasons this is a module of its own rather than defaults scattered through the
generator functions:

  1. **Reproducibility.** A world is fully described by (config, seed). Persist those
     two and anyone can regenerate the exact dataset — which is what makes a drift
     experiment repeatable and a model comparison honest.
  2. **Drift injection.** M5 needs "the fraud mix changed on 2026-04-01". That is a
     config change scheduled at a date, not a code change. Keeping the parameters as
     data is what makes a known-onset regime shift possible at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from fsa_sim.world.entities import Category, FraudTypology, SpendPersona

# ── Category behaviour ──────────────────────────────────────────────────────
# median_minor: the median claim in minor units for a grade-3 employee.
# sigma: lognormal shape — higher means a fatter tail of large claims.
# A lognormal is the right family here: expense amounts are positive, right-skewed,
# and multiplicative (a senior person's dinner is ~2x a junior's, not +₹500).


@dataclass(frozen=True, slots=True)
class CategoryProfile:
    median_minor: int
    sigma: float
    round_number_bias: float = 0.05  # P(amount lands on a suspiciously round figure)
    needs_attendees: bool = False


CATEGORY_PROFILES: dict[Category, CategoryProfile] = {
    Category.MEALS: CategoryProfile(80_000, 0.55, 0.08, needs_attendees=True),
    Category.CLIENT_ENTERTAINMENT: CategoryProfile(450_000, 0.70, 0.12, needs_attendees=True),
    Category.TRAVEL_AIR: CategoryProfile(1_200_000, 0.60),
    Category.TRAVEL_GROUND: CategoryProfile(45_000, 0.65),
    Category.LODGING: CategoryProfile(650_000, 0.50),
    Category.MILEAGE: CategoryProfile(120_000, 0.45, 0.20),
    Category.SOFTWARE: CategoryProfile(200_000, 0.80, 0.30),
    Category.OFFICE_SUPPLIES: CategoryProfile(35_000, 0.70),
    Category.TRAINING: CategoryProfile(2_500_000, 0.55, 0.25),
    Category.TELECOM: CategoryProfile(90_000, 0.30, 0.35),
}

# ── Persona behaviour ───────────────────────────────────────────────────────
# claims_per_month: Poisson rate. category_mix: unnormalised weights.


@dataclass(frozen=True, slots=True)
class PersonaProfile:
    claims_per_month: float
    category_mix: dict[Category, float]


PERSONA_PROFILES: dict[SpendPersona, PersonaProfile] = {
    SpendPersona.FREQUENT_TRAVELLER: PersonaProfile(
        claims_per_month=14.0,
        category_mix={
            Category.TRAVEL_AIR: 3.0,
            Category.LODGING: 3.0,
            Category.TRAVEL_GROUND: 4.0,
            Category.MEALS: 5.0,
            Category.MILEAGE: 1.0,
            Category.TELECOM: 0.5,
        },
    ),
    SpendPersona.DESK_BOUND: PersonaProfile(
        claims_per_month=2.5,
        category_mix={
            Category.SOFTWARE: 3.0,
            Category.OFFICE_SUPPLIES: 3.0,
            Category.MEALS: 2.0,
            Category.TELECOM: 1.5,
            Category.TRAINING: 0.5,
        },
    ),
    SpendPersona.CLIENT_FACING: PersonaProfile(
        claims_per_month=9.0,
        category_mix={
            Category.CLIENT_ENTERTAINMENT: 4.0,
            Category.MEALS: 3.0,
            Category.TRAVEL_GROUND: 3.0,
            Category.TRAVEL_AIR: 1.5,
            Category.LODGING: 1.5,
            Category.MILEAGE: 2.0,
        },
    ),
    SpendPersona.NEW_JOINER: PersonaProfile(
        claims_per_month=1.5,
        category_mix={
            Category.OFFICE_SUPPLIES: 3.0,
            Category.SOFTWARE: 2.0,
            Category.MEALS: 2.0,
            Category.TRAINING: 2.0,
        },
    ),
}

# Grade multiplier: a grade-6 exec's median claim vs a grade-1's. Index 0 unused.
GRADE_MULTIPLIER: tuple[float, ...] = (0.0, 0.55, 0.75, 1.0, 1.35, 1.9, 2.8)

# Seasonality multiplier by calendar month (1-12): conference season in Feb/Mar,
# a Q3 lull, and the Q4/year-end budget flush that every finance team recognises.
MONTH_SEASONALITY: tuple[float, ...] = (
    0.0,
    0.85,
    1.15,
    1.30,
    0.95,
    0.90,
    1.00,
    0.85,
    0.90,
    1.05,
    1.20,
    1.35,
    1.45,
)


@dataclass(frozen=True, slots=True)
class TenantShape:
    """A company archetype. Different shapes stress different parts of the system."""

    name: str
    n_employees: int
    n_departments: int
    max_depth: int  # org tree depth; deeper => harder manager-visibility authz
    span_of_control: int  # direct reports per manager
    currency: str = "INR"


TENANT_SHAPES: tuple[TenantShape, ...] = (
    TenantShape("startup", n_employees=50, n_departments=4, max_depth=2, span_of_control=8),
    TenantShape("enterprise", n_employees=600, n_departments=12, max_depth=4, span_of_control=6),
    TenantShape(
        "multi_entity",
        n_employees=250,
        n_departments=9,
        max_depth=3,
        span_of_control=5,
        currency="USD",
    ),
)


@dataclass(frozen=True, slots=True)
class FraudConfig:
    """How much fraud, and of what kind.

    `base_rate` is ~1-3% on purpose: this is a severely imbalanced problem, which is
    why §11 insists on PR-AUC and recall@k rather than accuracy. A 99% accurate model
    here is a model that predicts "not fraud" every time.
    """

    base_rate: float = 0.02
    typology_mix: dict[FraudTypology, float] = field(
        default_factory=lambda: {
            FraudTypology.DUPLICATE: 3.0,
            FraudTypology.SPLIT: 2.0,
            FraudTypology.INFLATION: 3.0,
            FraudTypology.GHOST_VENDOR: 1.5,
            FraudTypology.COLLUSION: 0.5,
            FraudTypology.PERSONAL: 2.0,
        }
    )
    # Audit lag: how long until a fraudulent claim is *confirmed*. This is what makes
    # ground truth delayed. Lognormal-ish; some claims are never audited.
    audit_lag_days_median: int = 21
    audit_lag_days_sigma: float = 0.6
    audit_coverage: float = 0.85  # fraction of true fraud that an audit ever confirms
    # Approval threshold that a "splitter" tries to stay under, in minor units.
    approval_threshold_minor: int = 3_000_000  # ₹30,000


@dataclass(frozen=True, slots=True)
class WorldConfig:
    """The full description of a world. (config, seed) => a reproducible dataset."""

    start_date: date = date(2026, 1, 1)
    end_date: date = date(2026, 12, 31)
    shapes: tuple[TenantShape, ...] = TENANT_SHAPES
    fraud: FraudConfig = field(default_factory=FraudConfig)
    injection_rate: float = 0.01  # fraction of receipts carrying a prompt-injection
    seed: int = 42

    @property
    def days(self) -> int:
        return (self.end_date - self.start_date).days + 1
