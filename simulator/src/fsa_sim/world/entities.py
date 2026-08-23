"""The synthetic world's entities.

These mirror the domain model in ARCHITECTURE.md §3 but are *plain dataclasses*, not
ORM models. That separation is deliberate: the simulator must be able to generate a
world without a database, so you can iterate on spend behaviour in a notebook in
milliseconds instead of round-tripping Postgres. `scripts/seed_db.py` maps these onto
the real tables in M1.

Money note: every amount here is `amount_minor: int` plus `currency: str`, matching
`fsa_common.Money`. We store the pair rather than a `Money` object because these
records get written straight to Parquet, and Parquet wants columns, not objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum


class Category(StrEnum):
    """Expense categories. Kept small and stable — they are metric labels (§13)."""

    MEALS = "meals"
    CLIENT_ENTERTAINMENT = "client_entertainment"
    TRAVEL_AIR = "travel_air"
    TRAVEL_GROUND = "travel_ground"
    LODGING = "lodging"
    MILEAGE = "mileage"
    SOFTWARE = "software"
    OFFICE_SUPPLIES = "office_supplies"
    TRAINING = "training"
    TELECOM = "telecom"


class SpendPersona(StrEnum):
    """How an employee spends. Drives claim frequency, category mix and amounts."""

    FREQUENT_TRAVELLER = "frequent_traveller"
    DESK_BOUND = "desk_bound"
    CLIENT_FACING = "client_facing"
    NEW_JOINER = "new_joiner"  # cold-start: no history for the model to lean on


class FraudTypology(StrEnum):
    """Real typologies (ARCHITECTURE.md §15). Each maps to detection features."""

    DUPLICATE = "duplicate"  # same receipt resubmitted, lightly altered
    SPLIT = "split"  # one large spend broken under an approval limit
    INFLATION = "inflation"  # mileage / per-diem padded 15-20%
    GHOST_VENDOR = "ghost_vendor"  # no-history vendor, suspiciously round amount
    COLLUSION = "collusion"  # manager reciprocally approving inflated claims
    PERSONAL = "personal"  # personal spend misclassified as business


@dataclass(frozen=True, slots=True)
class Tenant:
    tenant_id: str
    name: str
    currency: str
    shape: str  # "startup" | "enterprise" | "multi_entity"


@dataclass(frozen=True, slots=True)
class Department:
    department_id: str
    tenant_id: str
    name: str
    parent_id: str | None = None


@dataclass(frozen=True, slots=True)
class User:
    user_id: str
    tenant_id: str
    department_id: str
    manager_id: str | None
    role: str  # employee | manager | finance | auditor | admin
    grade: int  # 1 (junior) .. 6 (executive); correlates with claim size
    persona: SpendPersona
    joined_on: date


@dataclass(frozen=True, slots=True)
class Vendor:
    vendor_id: str
    tenant_id: str
    name: str
    category: Category
    first_seen_on: date


@dataclass(frozen=True, slots=True)
class Budget:
    tenant_id: str
    department_id: str
    category: Category
    period: str  # "2026-Q1"
    limit_minor: int
    currency: str


@dataclass(frozen=True, slots=True)
class ExpenseRecord:
    """One submitted claim, as the ledger would see it.

    NOTE: this struct carries **no label**. Ground truth lives in `FraudLabel`, in a
    separate file, with its own timestamp — see the docstring there. Keeping them apart
    is what makes point-in-time correctness enforceable instead of aspirational.
    """

    expense_id: str
    tenant_id: str
    user_id: str
    department_id: str
    category: Category
    vendor_id: str
    amount_minor: int
    currency: str
    transaction_date: date
    submitted_at: datetime
    description: str
    receipt_ocr_text: str
    receipt_phash: str  # perceptual hash of the receipt image (duplicate signal)
    line_item_count: int = 1
    attendee_count: int = 1


@dataclass(frozen=True, slots=True)
class FraudLabel:
    """Ground truth, and *when we learned it*.

    `confirmed_at` is the whole point. Fraud is confirmed by an audit days or weeks
    after submission, so a model trained or evaluated as of time T may only use labels
    where `confirmed_at <= T`. Every leakage bug in fraud modelling is some version of
    forgetting this. See ARCHITECTURE.md §11, "delayed ground truth".
    """

    expense_id: str
    is_fraud: bool
    typology: FraudTypology | None
    confirmed_at: datetime | None  # None => still un-audited at world end
    linked_expense_ids: tuple[str, ...] = ()  # e.g. the other legs of a split


@dataclass(frozen=True, slots=True)
class InjectionLabel:
    """Ground truth for the adversarial side: which receipts carry a payload."""

    expense_id: str
    payload_family: str  # "instruction_override" | "exfiltration" | "role_play" | ...
    payload_text: str


@dataclass(slots=True)
class World:
    """Everything one `generate()` run produced. The unit we persist and reload."""

    tenants: list[Tenant] = field(default_factory=list)
    departments: list[Department] = field(default_factory=list)
    users: list[User] = field(default_factory=list)
    vendors: list[Vendor] = field(default_factory=list)
    budgets: list[Budget] = field(default_factory=list)
    expenses: list[ExpenseRecord] = field(default_factory=list)
    fraud_labels: list[FraudLabel] = field(default_factory=list)
    injection_labels: list[InjectionLabel] = field(default_factory=list)

    def users_of(self, tenant_id: str) -> list[User]:
        return [u for u in self.users if u.tenant_id == tenant_id]

    def summary(self) -> dict[str, int]:
        return {
            "tenants": len(self.tenants),
            "departments": len(self.departments),
            "users": len(self.users),
            "vendors": len(self.vendors),
            "budgets": len(self.budgets),
            "expenses": len(self.expenses),
            "fraud_labels": len(self.fraud_labels),
            # Two different numbers, and the gap between them is the whole delayed-
            # ground-truth story: `fraud_positives` is what is actually true;
            # `confirmed_by_audit` is what a model is allowed to have learned from.
            "fraud_positives": sum(1 for f in self.fraud_labels if f.is_fraud),
            "confirmed_by_audit": sum(
                1 for f in self.fraud_labels if f.is_fraud and f.confirmed_at is not None
            ),
            "injections": len(self.injection_labels),
        }
