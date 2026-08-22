"""Claim-intrinsic features — worked example.

"Intrinsic" means computable from the claim row alone, with no history and no joins.
They are the easy family, and they are here in full so you have a reference for the
shape every other family should follow:

    def <family>(claims: pd.DataFrame, ..., as_of: AsOf) -> pd.DataFrame:
        returns a frame indexed by expense_id, one column per feature, prefixed
        with the family name.

Prefixing (`ci_`, `eh_`, `vel_`, ...) is not decoration: when SHAP hands you the top
contributors for a flagged claim, `eh_amount_zscore_90d` tells the investigator which
family fired, and the agent's `fraud_reasons` list becomes readable to a human.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fsa_ml.features.pointintime import AsOf


def build(claims: pd.DataFrame, as_of: AsOf) -> pd.DataFrame:
    """Intrinsic features. `as_of` is accepted but unused — deliberately.

    Keeping the parameter on every family makes the registry uniform and stops anyone
    "just this once" writing a family with a different signature, which is how the
    as_of discipline erodes.
    """
    del as_of

    txn = pd.to_datetime(claims["transaction_date"])
    sub = pd.to_datetime(claims["submitted_at"])
    amount = claims["amount_minor"].astype("int64")

    out = pd.DataFrame(index=claims["expense_id"])
    out["ci_amount_minor"] = amount.to_numpy()
    out["ci_amount_log"] = np.log1p(amount.to_numpy())
    out["ci_days_since_transaction"] = (sub - txn).dt.days.to_numpy()
    out["ci_is_weekend_txn"] = (txn.dt.weekday >= 5).astype("int8").to_numpy()
    out["ci_submitted_hour"] = sub.dt.hour.to_numpy()
    out["ci_txn_month"] = txn.dt.month.to_numpy()
    # Round-number flag: a real signal for inflated and ghost-vendor claims, and one
    # the honest world also produces (see CategoryProfile.round_number_bias) so the
    # model has to weigh it rather than memorise it.
    out["ci_is_round_100"] = (amount.to_numpy() % 10_000 == 0).astype("int8")
    out["ci_is_round_1000"] = (amount.to_numpy() % 100_000 == 0).astype("int8")
    out["ci_line_item_count"] = claims["line_item_count"].to_numpy()
    out["ci_attendee_count"] = claims["attendee_count"].to_numpy()
    out["ci_amount_per_attendee"] = amount.to_numpy() / np.maximum(
        claims["attendee_count"].to_numpy(), 1
    )
    return out
