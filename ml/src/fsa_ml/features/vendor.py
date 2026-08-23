"""Vendor features — "do we know who we are paying".

This is the family that catches GHOST_VENDOR. The signal is not the amount (which is
now scaled relative to the claim it replaced, per ADR 0005) — it is that the vendor
appeared in the ledger nine days ago and has one invoice against it.

`ven_days_since_first_seen` uses the vendor table's `first_seen_on`, which is a
property of the vendor rather than of prior claims, so it does not need windowing.
Everything else does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fsa_ml.features.pointintime import AsOf, prior_count


def build(claims: pd.DataFrame, as_of: AsOf) -> pd.DataFrame:
    del as_of

    frame = claims[["expense_id", "tenant_id", "user_id", "vendor_id", "submitted_at"]].copy()
    frame["submitted_at"] = pd.to_datetime(frame["submitted_at"])
    frame["tenant_vendor"] = frame["tenant_id"].astype(str) + "|" + frame["vendor_id"].astype(str)
    frame["user_vendor"] = frame["user_id"].astype(str) + "|" + frame["vendor_id"].astype(str)

    out = pd.DataFrame(index=claims["expense_id"])

    # How established is this vendor in the tenant, and with this employee?
    out["ven_tenant_prior_365d"] = prior_count(frame, by=["tenant_vendor"], window="365D").reindex(
        out.index
    )
    out["ven_user_prior_365d"] = prior_count(frame, by=["user_vendor"], window="365D").reindex(
        out.index
    )

    # Vendor-employee affinity: a vendor nobody else in the tenant uses, but this one
    # employee uses repeatedly, is the shape of a ghost vendor or a kickback.
    out["ven_user_share"] = out["ven_user_prior_365d"] / out["ven_tenant_prior_365d"].replace(
        0.0, np.nan
    )
    out["ven_is_first_ever"] = (out["ven_tenant_prior_365d"] == 0).astype("int8")

    if "first_seen_on" in claims.columns:
        first_seen = pd.to_datetime(claims.set_index("expense_id")["first_seen_on"])
        submitted = pd.to_datetime(claims.set_index("expense_id")["submitted_at"])
        out["ven_days_since_first_seen"] = (submitted - first_seen).dt.days

    out = out.replace([np.inf, -np.inf], np.nan)
    return out
