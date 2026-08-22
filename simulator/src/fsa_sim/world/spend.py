"""Baseline (honest) spend generation.

    ┌──────────────────────────────────────────────────────────────────────┐
    │  YOUR TASK — Step 1.                                                 │
    │  Two functions below raise NotImplementedError:                      │
    │      sample_amount_minor()                                           │
    │      generate_baseline_expenses()                                    │
    │  The contract for both is in tests/simulator/test_spend.py, which is  │
    │  skipped until you implement them and then runs automatically.        │
    │  Everything else in this file is done — use it.                       │
    └──────────────────────────────────────────────────────────────────────┘

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
    CATEGORY_PROFILES,  # noqa: F401  — pre-imported for your implementation below
    GRADE_MULTIPLIER,  # noqa: F401  — ditto; delete these noqas once you use them
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

    Contract (tests/simulator/test_spend.py::TestSampleAmount):
      * strictly positive `int`
      * lognormal around `CATEGORY_PROFILES[category].median_minor`, scaled by
        `GRADE_MULTIPLIER[grade]`, with the category's `sigma`
      * with probability `profile.round_number_bias`, snap to a "suspiciously round"
        figure — a multiple of 100 major units — because round numbers are a real
        fraud signal and the honest world must contain some too, or the feature
        becomes a perfect giveaway
      * determinism: same rng state + same inputs => same value

    Hints — the syntax you need:

        profile = CATEGORY_PROFILES[category]
        median  = profile.median_minor * GRADE_MULTIPLIER[grade]

        # numpy's lognormal is parameterised by the mean/sigma of the *underlying
        # normal*. For a target median m, the underlying mean is log(m) — because
        # the median of a lognormal is exp(mu).
        value = rng.lognormal(mean=np.log(median), sigma=profile.sigma)

        if rng.random() < profile.round_number_bias:
            value = round(value / 10_000) * 10_000     # nearest ₹100 in paise

        return max(1, int(value))
    """
    raise NotImplementedError("Step 1a — see the contract above and test_spend.py")


def generate_baseline_expenses(
    users: list[User],
    vendors: list[Vendor],
    config: WorldConfig,
    rng: np.random.Generator,
    *,
    tenant_currency: str,
) -> list[ExpenseRecord]:
    """Generate every legitimate claim for one tenant over the world's date range.

    Contract (tests/simulator/test_spend.py::TestGenerateBaseline):
      * every `expense_id` is unique
      * `transaction_date` within [config.start_date, config.end_date]
      * `transaction_date >= user.joined_on` for every claim
      * `submitted_at.date() >= transaction_date`
      * `amount_minor > 0` and `currency == tenant_currency` everywhere
      * the vendor on a claim has the same `category` as the claim
      * a FREQUENT_TRAVELLER produces materially more claims than a DESK_BOUND peer
      * weekend claims are a minority of the total

    Suggested shape — a per-user, per-month loop:

        expenses: list[ExpenseRecord] = []
        by_category: dict[Category, list[Vendor]] = {}
        for v in vendors:
            by_category.setdefault(v.category, []).append(v)

        seq = 0
        for user in users:
            rate = PERSONA_PROFILES[user.persona].claims_per_month
            for month_start in _months(config.start_date, config.end_date):
                lam = rate * seasonality_multiplier(month_start)
                n_claims = int(rng.poisson(lam))
                for _ in range(n_claims):
                    ...
                    seq += 1
                    expenses.append(ExpenseRecord(expense_id=f"exp-{seq:08d}", ...))
        return expenses

    Two decisions worth thinking about before you write it (there is no single right
    answer — pick one, and put your reasoning in a code comment):
      1. Poisson per month with a seasonal rate, or Poisson per day? Per-month is
         cheaper and gives clean seasonality; per-day gives you burstiness and
         Monday-morning spikes for free. Which does the fraud model need?
      2. `seasonality_multiplier` folds in a weekday effect, but a per-month draw has
         no weekdays. If you go per-month, where does the weekend dip come from?
    """
    raise NotImplementedError("Step 1b — see the contract above and test_spend.py")


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
