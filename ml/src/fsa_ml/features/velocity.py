"""Velocity features — "how much has this person claimed lately".

This is the family that catches SPLIT, and it is the only one that can. A split leg,
viewed in isolation, is an ordinary claim from an ordinary vendor for an unremarkable
amount. What gives it away is three of them inside four days, each landing just under
the approval limit. No claim-intrinsic feature can see that; a trailing-window count
can.

`vel_near_threshold_7d` is the explicit split detector. It is a deliberately
hand-crafted feature rather than something the model is expected to discover, because
the approval threshold is a *business constant*, not a pattern in the data — a tree
would have to find the boundary by splitting on amount repeatedly, and it would find
a slightly wrong one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fsa_ml.features.constants import APPROVAL_THRESHOLD_MINOR, NEAR_THRESHOLD_FLOOR
from fsa_ml.features.pointintime import AsOf, prior_count, prior_rolling


def build(claims: pd.DataFrame, as_of: AsOf) -> pd.DataFrame:
    del as_of

    frame = claims[["expense_id", "user_id", "vendor_id", "submitted_at", "amount_minor"]].copy()
    frame["submitted_at"] = pd.to_datetime(frame["submitted_at"])
    frame["user_vendor"] = frame["user_id"].astype(str) + "|" + frame["vendor_id"].astype(str)
    frame["near_threshold"] = (
        (frame["amount_minor"] >= APPROVAL_THRESHOLD_MINOR * NEAR_THRESHOLD_FLOOR)
        & (frame["amount_minor"] < APPROVAL_THRESHOLD_MINOR)
    ).astype(float)

    out = pd.DataFrame(index=claims["expense_id"])

    for window in ("1D", "7D", "30D"):
        out[f"vel_claims_{window.lower()}"] = prior_count(
            frame, by=["user_id"], window=window
        ).reindex(out.index)
        out[f"vel_amount_sum_{window.lower()}"] = (
            prior_rolling(frame, by=["user_id"], value="amount_minor", window=window, agg="sum")
            .reindex(out.index)
            .fillna(0.0)
        )

    # The split detector. Counts the submitter's *prior* just-under-limit claims in a
    # week; the current claim's own proximity to the limit is a claim-intrinsic
    # feature, added below so the pair can interact.
    out["vel_near_threshold_7d"] = (
        prior_rolling(frame, by=["user_id"], value="near_threshold", window="7D", agg="sum")
        .reindex(out.index)
        .fillna(0.0)
    )
    out["vel_is_near_threshold"] = frame.set_index("expense_id")["near_threshold"].reindex(
        out.index
    )
    out["vel_threshold_headroom"] = (
        (APPROVAL_THRESHOLD_MINOR - claims.set_index("expense_id")["amount_minor"])
        / APPROVAL_THRESHOLD_MINOR
    ).reindex(out.index)

    # Same-vendor burst: three claims at one restaurant in four days is either a
    # conference or a split.
    out["vel_same_vendor_7d"] = prior_count(frame, by=["user_vendor"], window="7D").reindex(
        out.index
    )

    # Ratio of this claim to the week's running total — a large claim on a quiet week
    # is different from the same claim on a busy one.
    out["vel_amount_vs_7d_sum"] = claims.set_index("expense_id")["amount_minor"] / (
        out["vel_amount_sum_7d"].replace(0.0, np.nan)
    )
    return out
