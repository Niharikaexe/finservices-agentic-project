"""Employee-historical features — "is this claim unusual *for this person*".

This is the family that catches INFLATION, and it is the family where leakage lives.
An employee's mean claim computed over the whole dataset already contains the
fraudulent claim you are trying to detect; the z-score then comes out near zero for
precisely the rows that should score highest.

Every aggregate here goes through `prior_rolling`, which windows on strictly earlier
rows. See `tests/ml/test_features.py::test_features_are_causal` for the property that
proves it: a claim's features must not change when every row after it is deleted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fsa_ml.features.constants import COLD_START_CLAIMS
from fsa_ml.features.pointintime import AsOf, prior_count, prior_rolling


def build(claims: pd.DataFrame, as_of: AsOf) -> pd.DataFrame:
    del as_of  # windows are per-row; the family signature stays uniform

    frame = claims[["expense_id", "user_id", "category", "submitted_at", "amount_minor"]].copy()
    frame["submitted_at"] = pd.to_datetime(frame["submitted_at"])
    frame["submitted_hour"] = frame["submitted_at"].dt.hour.astype(float)
    frame["user_cat"] = frame["user_id"].astype(str) + "|" + frame["category"].astype(str)

    out = pd.DataFrame(index=claims["expense_id"])

    # ── vs the employee's own history in this category ──────────────────────
    cat_mean = prior_rolling(frame, by=["user_cat"], value="amount_minor", window="90D", agg="mean")
    cat_std = prior_rolling(frame, by=["user_cat"], value="amount_minor", window="90D", agg="std")
    cat_n = prior_count(frame, by=["user_cat"], window="90D")

    amount = claims.set_index("expense_id")["amount_minor"].astype(float)
    # A std of 0 (or 1 prior claim) would divide by zero. Guard with NaN rather than
    # a fudge constant: "undefined" is the truthful value and LightGBM can use it.
    safe_std = cat_std.reindex(out.index).replace(0.0, np.nan)
    out["eh_amount_zscore_cat_90d"] = (amount - cat_mean.reindex(out.index)) / safe_std
    out["eh_amount_ratio_to_own_mean"] = amount / cat_mean.reindex(out.index)
    out["eh_prior_claims_cat_90d"] = cat_n.reindex(out.index)

    # ── vs the employee's overall behaviour ─────────────────────────────────
    out["eh_prior_claims_30d"] = prior_count(frame, by=["user_id"], window="30D").reindex(out.index)
    out["eh_prior_claims_90d"] = prior_count(frame, by=["user_id"], window="90D").reindex(out.index)
    out["eh_prior_amount_mean_90d"] = prior_rolling(
        frame, by=["user_id"], value="amount_minor", window="90D", agg="mean"
    ).reindex(out.index)
    out["eh_prior_amount_max_90d"] = prior_rolling(
        frame, by=["user_id"], value="amount_minor", window="90D", agg="max"
    ).reindex(out.index)

    # ── submission timing ───────────────────────────────────────────────────
    # Regularity of submission hour. A person who always files at 10am and suddenly
    # files at 02:40 has changed behaviour; that is worth a feature. Std rather than
    # Shannon entropy: entropy over a continuous hour needs binning, and a rolling
    # `apply` for it costs ~40x the runtime for no measurable lift here.
    hour_mean = prior_rolling(
        frame, by=["user_id"], value="submitted_hour", window="90D", agg="mean"
    )
    out["eh_submission_hour_std_90d"] = prior_rolling(
        frame, by=["user_id"], value="submitted_hour", window="90D", agg="std"
    ).reindex(out.index)
    out["eh_submission_hour_dev"] = (
        claims.set_index("expense_id")["submitted_at"].pipe(pd.to_datetime).dt.hour
        - hour_mean.reindex(out.index)
    ).abs()

    # ── category concentration ──────────────────────────────────────────────
    out["eh_category_share_90d"] = out["eh_prior_claims_cat_90d"] / out[
        "eh_prior_claims_90d"
    ].replace(0.0, np.nan)

    # ── cold start ──────────────────────────────────────────────────────────
    # Not a nuisance to be imputed away: a claim from someone with no history is a
    # genuinely different decision problem, and the model should be told so.
    out["eh_is_cold_start"] = (out["eh_prior_claims_90d"] < COLD_START_CLAIMS).astype("int8")

    return out
