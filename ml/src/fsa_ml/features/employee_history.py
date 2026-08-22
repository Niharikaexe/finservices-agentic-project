"""Employee-historical features.

    ┌──────────────────────────────────────────────────────────────────────┐
    │  YOUR TASK — Step 3 (after the simulator produces data).             │
    │  This is the family where leakage actually happens, which is why it   │
    │  is the one you write by hand.                                        │
    └──────────────────────────────────────────────────────────────────────┘

Target features (ARCHITECTURE.md §11, "Employee-historical"):

    eh_amount_zscore_90d      claim amount vs the employee's own 90-day mean/std
                              for that category
    eh_claims_30d             count of the employee's claims in the trailing 30 days
    eh_days_since_last_claim  recency
    eh_submission_hour_entropy  Shannon entropy of submission hours over 90 days
    eh_category_share_90d     share of this employee's 90-day claims in this category
    eh_is_cold_start          1 when the employee has < 5 prior claims

The hard part, and the whole point:

    For claim C submitted at time T, every one of these must be computed over that
    employee's claims with `submitted_at < T` — **T of that claim**, not a single
    global as_of, and not the whole dataset.

That means a naive `groupby(user_id).transform("mean")` is wrong, and it will look
right, and it will hand you an AUC you will want to believe. The correct shapes:

    # Option A — sort by time, then expanding/rolling within each user. Exact.
    frame = claims.sort_values("submitted_at")
    g = frame.groupby("user_id")["amount_minor"]
    prior_mean = g.transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    #                                    ^^^^^^^^^ shift(1) is the whole ballgame:
    #                                    without it, the claim is in its own baseline.

    # Option B — time-windowed, closed on the left so the current row is excluded:
    frame = frame.set_index("submitted_at").sort_index()
    prior = (frame.groupby("user_id")["amount_minor"]
                  .rolling("90D", closed="left").mean())

    # Cold start: min_periods gives NaN, and NaN is *information* here. LightGBM
    # handles NaN natively — do NOT fillna(0), which asserts "this employee's mean
    # claim is zero" and is simply false.

Verification, before you trust any of it — write this as a test:

    A claim's features must be unchanged when every row after it is deleted.
    That single property catches almost every leak. It is `test_features.py::
    test_features_are_causal`, and it is worth more than the features themselves.
"""

from __future__ import annotations

import pandas as pd

from fsa_ml.features.pointintime import AsOf


def build(claims: pd.DataFrame, as_of: AsOf) -> pd.DataFrame:
    """See module docstring. Returns a frame indexed by `expense_id`."""
    raise NotImplementedError("Step 3 — employee-historical features")
