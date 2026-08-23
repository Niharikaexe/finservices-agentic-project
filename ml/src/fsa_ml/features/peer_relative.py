"""Peer-relative features — "is this claim unusual for someone in this role".

Employee-historical features are blind to a fraudster who has *always* inflated: their
own baseline is already padded, so their z-score is unremarkable. Peer comparison is
the defence. It is also the family that carries the cold-start case, since a new joiner
has no personal history but does have peers.

Comparison groups are (department, category) and (grade, category). Both are needed:
a grade-6 executive's ₹40,000 dinner is normal for their grade and abnormal for their
department's median.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fsa_ml.features.pointintime import AsOf, prior_rolling


def build(claims: pd.DataFrame, as_of: AsOf) -> pd.DataFrame:
    del as_of

    required = ["expense_id", "department_id", "category", "grade", "submitted_at", "amount_minor"]
    missing = [column for column in required if column not in claims.columns]
    if missing:
        raise KeyError(
            f"peer_relative needs {missing}; join the user table onto claims first "
            "(see fsa_ml.features.build.assemble)"
        )

    frame = claims[required].copy()
    frame["submitted_at"] = pd.to_datetime(frame["submitted_at"])
    frame["dept_cat"] = frame["department_id"].astype(str) + "|" + frame["category"].astype(str)
    frame["grade_cat"] = frame["grade"].astype(str) + "|" + frame["category"].astype(str)

    out = pd.DataFrame(index=claims["expense_id"])
    amount = claims.set_index("expense_id")["amount_minor"].astype(float)

    for key, prefix in (("dept_cat", "peer_dept"), ("grade_cat", "peer_grade")):
        mean = prior_rolling(
            frame, by=[key], value="amount_minor", window="180D", agg="mean"
        ).reindex(out.index)
        std = prior_rolling(
            frame, by=[key], value="amount_minor", window="180D", agg="std"
        ).reindex(out.index)
        out[f"{prefix}_ratio"] = amount / mean.replace(0.0, np.nan)
        out[f"{prefix}_zscore"] = (amount - mean) / std.replace(0.0, np.nan)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out
