"""Baseline (honest) spend generation.

What "baseline" means here: this module generates **only legitimate spend**. Fraud is
layered on afterwards by `fsa_sim.adversarial`, which *transforms* baseline claims into
fraudulent ones. Keeping them separate is not cosmetic:

  * it lets you regenerate the same honest world with a different fraud rate, which is
    how you build the drift experiment in M5 (same population, new typology at T+30d);
  * it stops ground truth leaking into the honest data — if fraud were generated inline
    it is far too easy to give fraudulent claims a subtly different amount distribution
    *for reasons unrelated to the fraud*, and then your model learns the generator
    rather than the crime. That is the single most common way a synthetic fraud dataset
    turns into a lie.

Design constraints to respect while you implement:

  * **Determinism.** Take `rng` and use only it.
  * **Amounts are ints in minor units.** No floats leak into `amount_minor`.
  * **`submitted_at >= transaction_date`.** People expense things after they buy them,
    usually with a lag of a few days, sometimes weeks. That lag is itself a feature
    ("days since transaction") and a fraud signal, so it must be realistic.
  * **Nobody claims before they join.** `transaction_date >= user.joined_on`.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, time, timedelta

import numpy as np

from fsa_sim.world.config import (
    CATEGORY_PROFILES,
    GRADE_MULTIPLIER,
    MONTH_SEASONALITY,
    PERSONA_PROFILES,
    WorldConfig,
)
from fsa_sim.world.entities import Category, ExpenseRecord, User, Vendor

# ── Provided helpers — read them, use them ──────────────────────────────────


def seasonality_multiplier(day: date) -> float:
    """Month-of-year multiplier, plus a weekday effect.

    Two effects, both real: departments flush budget in Q4/March, and business spend
    happens on weekdays. The weekend dip is why "weekend flag" is a fraud feature at
    all — a legitimate weekend claim is uncommon, so it carries information.
    """
    month = MONTH_SEASONALITY[day.month]
    weekday = 0.35 if day.weekday() >= 5 else 1.0
    return month * weekday


def pick_category(user: User, rng: np.random.Generator) -> Category:
    """Sample a category from the user's persona mix."""
    mix = PERSONA_PROFILES[user.persona].category_mix
    categories = list(mix)
    probs = np.array([mix[c] for c in categories], dtype=float)
    return categories[int(rng.choice(len(categories), p=probs / probs.sum()))]


def submission_lag_days(rng: np.random.Generator) -> int:
    """Days between spending and claiming. Right-skewed: most within a week, a tail of
    people who submit a month later (and a few who submit on the last allowed day)."""
    return int(min(rng.exponential(5.0), 45))


def submission_timestamp(day: date, rng: np.random.Generator) -> datetime:
    """Time-of-day for the submission. Bimodal — a morning peak and a late-evening
    peak — because "submission-hour entropy" is an employee-historical fraud feature
    and a flat uniform hour would make it meaningless."""
    hour = int(rng.normal(10, 1.5)) if rng.random() < 0.6 else int(rng.normal(21, 2.0))
    hour = max(0, min(23, hour))
    return datetime.combine(day, time(hour, int(rng.integers(0, 60))))


def synthesise_receipt_text(vendor: Vendor, amount_minor: int, day: date, currency: str) -> str:
    """A plausible OCR'd receipt.

    Note what this is *not*: it is not trusted. Downstream it is wrapped in
    `fsa_guardrails.UntrustedText` before it can reach a prompt, because M3 will start
    planting injection payloads in exactly this string.
    """
    major = amount_minor / 100
    return (
        f"{vendor.name.upper()}\n"
        f"GSTIN 29ABCDE1234F1Z5\n"
        f"DATE {day.isoformat()}   TIME 19:42\n"
        f"------------------------------\n"
        f"SUBTOTAL   {major * 0.847:.2f}\n"
        f"GST 18%    {major * 0.153:.2f}\n"
        f"TOTAL      {major:.2f} {currency}\n"
        f"THANK YOU / VISIT AGAIN"
    )


def receipt_phash(vendor_id: str, amount_minor: int, day: date, salt: str = "") -> str:
    """Stand-in for a perceptual hash of the receipt image.

    Real pHash is near-identical for near-identical images. We fake that property by
    hashing (vendor, amount, date): a duplicate submitter who resubmits the same
    receipt gets the same digest, and one who lightly alters it gets a digest that
    differs in a controlled number of characters. That is enough for the duplicate
    feature family to be exercisable without generating actual images.
    """
    payload = f"{vendor_id}|{amount_minor}|{day.isoformat()}|{salt}".encode()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


# ── YOUR TASK ───────────────────────────────────────────────────────────────


def sample_amount_minor(category: Category, grade: int, rng: np.random.Generator) -> int:
    """Sample one claim amount, in minor units.

    Lognormal, because expense amounts are positive, right-skewed and *multiplicative*
    — a senior person's dinner is roughly 2x a junior's, not "+ ₹500". numpy's
    `lognormal` is parameterised by the mean and sigma of the **underlying normal**,
    so for a target median m the underlying mean is `log(m)`: the median of a
    lognormal is `exp(mu)`, while its mean is the larger `exp(mu + sigma^2/2)`.

    The round-number snap deserves a note. Round amounts are a genuine fraud signal
    (inflated and ghost-vendor claims cluster on them), so the *honest* world has to
    produce some too. If only fraudulent claims were round, `ci_is_round_100` would be
    a perfect giveaway and the model would learn the generator instead of the crime.
    """
    profile = CATEGORY_PROFILES[category]
    median = profile.median_minor * GRADE_MULTIPLIER[grade]
    value = rng.lognormal(mean=float(np.log(median)), sigma=profile.sigma)

    if rng.random() < profile.round_number_bias:
        # Nearest ₹100, expressed in paise. Floored at one unit of the rounding grid
        # so a small claim never collapses to zero.
        value = max(10_000.0, round(value / 10_000) * 10_000)

    return max(1, int(value))


def generate_baseline_expenses(
    users: list[User],
    vendors: list[Vendor],
    config: WorldConfig,
    rng: np.random.Generator,
    *,
    tenant_currency: str,
) -> list[ExpenseRecord]:
    """Every legitimate claim for one tenant over the world's date range.

    Two design decisions, both arguable:

    **1. Poisson per month, then scatter over days — not Poisson per day.**
    A per-day draw is the obvious choice and it is wrong here for a specific reason:
    it makes claims independent across days, so the velocity feature family (count and
    sum in a trailing 1d/7d/30d window) sees pure Poisson noise with no structure to
    learn. Real expense behaviour is bursty — you travel for three days and file six
    claims. So we draw a monthly count, pick a *small set of active days*, and land
    every claim of that month on one of them. Clustering falls out for free, and the
    velocity features get a real signal that a splitter has to hide inside.

    **2. The monthly rate uses `MONTH_SEASONALITY` directly, not
    `seasonality_multiplier(month_start)`.**
    That helper folds in a weekday factor, which is meaningless applied to the first
    of the month — 1 March 2026 is a Sunday, so using it would cut March's *entire*
    volume by 65% and invert the conference-season effect we are trying to model. The
    weekday effect belongs at day granularity, where it is applied as the weight when
    choosing which days are active. Same helper, right altitude.
    """
    by_category: dict[Category, list[Vendor]] = {}
    for vendor in vendors:
        by_category.setdefault(vendor.category, []).append(vendor)

    expenses: list[ExpenseRecord] = []
    seq = 0

    for user in users:
        rate = PERSONA_PROFILES[user.persona].claims_per_month
        available = set(PERSONA_PROFILES[user.persona].category_mix) & set(by_category)
        if not available:
            continue

        for month_start in months_between(config.start_date, config.end_date):
            days = [d for d in days_in_month(month_start, config) if d >= user.joined_on]
            if not days:
                continue

            n_claims = int(rng.poisson(rate * MONTH_SEASONALITY[month_start.month]))
            if n_claims == 0:
                continue

            # Roughly half as many active days as claims => an average of two claims
            # per active day, with a tail of busier ones. This is the burstiness knob.
            weights = np.array([seasonality_multiplier(d) for d in days], dtype=float)
            n_active = max(1, min(len(days), -(-n_claims // 2)))
            active_idx = rng.choice(
                len(days), size=n_active, replace=False, p=weights / weights.sum()
            )
            active_days = [days[int(i)] for i in active_idx]

            for _ in range(n_claims):
                txn_date = active_days[int(rng.integers(len(active_days)))]

                category = pick_category(user, rng)
                if category not in by_category:
                    continue
                pool = by_category[category]
                vendor = pool[int(rng.integers(len(pool)))]

                amount = sample_amount_minor(category, user.grade, rng)
                profile = CATEGORY_PROFILES[category]
                attendees = int(rng.integers(1, 7)) if profile.needs_attendees else 1

                submitted_day = txn_date + timedelta(days=submission_lag_days(rng))
                seq += 1
                expenses.append(
                    ExpenseRecord(
                        expense_id=f"{user.tenant_id}-exp-{seq:08d}",
                        tenant_id=user.tenant_id,
                        user_id=user.user_id,
                        department_id=user.department_id,
                        category=category,
                        vendor_id=vendor.vendor_id,
                        amount_minor=amount,
                        currency=tenant_currency,
                        transaction_date=txn_date,
                        submitted_at=submission_timestamp(submitted_day, rng),
                        description=f"{category.value.replace('_', ' ')} - {vendor.name}",
                        receipt_ocr_text=synthesise_receipt_text(
                            vendor, amount, txn_date, tenant_currency
                        ),
                        receipt_phash=receipt_phash(vendor.vendor_id, amount, txn_date),
                        line_item_count=int(rng.integers(1, 5)),
                        attendee_count=attendees,
                    )
                )

    return expenses


# ── A helper you will want for the loop above ───────────────────────────────


def months_between(start: date, end: date) -> list[date]:
    """First-of-month dates covering [start, end] inclusive."""
    out: list[date] = []
    cursor = start.replace(day=1)
    while cursor <= end:
        out.append(cursor)
        cursor = (cursor + timedelta(days=32)).replace(day=1)
    return out


def days_in_month(month_start: date, config: WorldConfig) -> list[date]:
    """Every day of `month_start`'s month that falls inside the world's range."""
    nxt = (month_start + timedelta(days=32)).replace(day=1)
    out: list[date] = []
    cursor = month_start
    while cursor < nxt:
        if config.start_date <= cursor <= config.end_date:
            out.append(cursor)
        cursor += timedelta(days=1)
    return out
