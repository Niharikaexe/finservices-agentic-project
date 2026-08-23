"""Duplicate-detection features — "have we paid for this before".

This is the family that catches DUPLICATE. The simulator's resubmitted receipt keeps
the original's perceptual hash (the image is the same) but shifts the amount by a
percent or two and the date by a month or three — so an exact `(vendor, amount, date)`
equality check misses it entirely, which is the point.

Two independent signals, because in production either can be absent: a receipt may
fail to OCR, and a pHash may be missing for a typed claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fsa_ml.features.constants import DUPLICATE_AMOUNT_TOLERANCE
from fsa_ml.features.pointintime import AsOf, prior_count


def build(claims: pd.DataFrame, as_of: AsOf) -> pd.DataFrame:
    del as_of

    frame = claims[
        [
            "expense_id",
            "tenant_id",
            "user_id",
            "vendor_id",
            "submitted_at",
            "amount_minor",
            "receipt_phash",
        ]
    ].copy()
    frame["submitted_at"] = pd.to_datetime(frame["submitted_at"])
    frame["tenant_phash"] = frame["tenant_id"].astype(str) + "|" + frame["receipt_phash"]
    # Amount bucketed to the tolerance, so "same vendor, near-same amount" becomes a
    # group key and the whole thing stays a fast groupby instead of an O(n^2) scan.
    bucket = (frame["amount_minor"] / (frame["amount_minor"] * DUPLICATE_AMOUNT_TOLERANCE)).round()
    frame["vendor_amount"] = (
        frame["vendor_id"].astype(str) + "|" + bucket.fillna(-1).astype("int64").astype(str)
    )

    out = pd.DataFrame(index=claims["expense_id"])

    # Has this exact receipt image been submitted before, anywhere in the tenant?
    # Tenant-wide rather than per-user on purpose: two employees submitting the same
    # receipt is collusion, and scoping to the user would hide it.
    out["dup_phash_prior_365d"] = prior_count(frame, by=["tenant_phash"], window="365D").reindex(
        out.index
    )

    # Days since that receipt was last seen. The simulator resubmits 30-90 days later,
    # which is far enough apart that a human reviewer would not remember it.
    ordered = frame.sort_values(["tenant_phash", "submitted_at"])
    previous = ordered.groupby("tenant_phash")["submitted_at"].shift(1)
    gap = (ordered["submitted_at"] - previous).dt.total_seconds() / 86_400
    out["dup_days_since_same_phash"] = pd.Series(
        gap.to_numpy(), index=ordered["expense_id"].to_numpy()
    ).reindex(out.index)

    # The independent signal: same vendor, amount within tolerance, recently.
    out["dup_vendor_amount_prior_90d"] = prior_count(
        frame, by=["vendor_amount"], window="90D"
    ).reindex(out.index)

    # A hash seen before but never by this user is a weaker signal than one this user
    # has already claimed against.
    frame["user_phash"] = frame["user_id"].astype(str) + "|" + frame["receipt_phash"]
    out["dup_user_phash_prior_365d"] = prior_count(frame, by=["user_phash"], window="365D").reindex(
        out.index
    )

    out["dup_any_prior"] = (
        (out["dup_phash_prior_365d"] > 0) | (out["dup_vendor_amount_prior_90d"] > 0)
    ).astype("int8")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out
