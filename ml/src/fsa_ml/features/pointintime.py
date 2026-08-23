"""Point-in-time correctness — the contract, not a convention.

The single most expensive bug in applied ML is training on information that did not
exist yet. In this domain it has two distinct forms, and they need two distinct
defences:

  **Feature leakage.** A feature computed over the employee's *whole* history, then
  used to predict a claim from the middle of that history. "Employee's mean claim
  amount" computed over the full dataset already contains the fraudulent claim you are
  trying to detect.
  Defence: every aggregate takes `as_of` and may only read rows with
  `submitted_at < as_of`. `AsOf` below makes that argument impossible to forget.

  **Label leakage.** Fraud is confirmed by an audit weeks later (§11). A model trained
  "as of 1 March" may only use labels with `confirmed_at <= 1 March`. Training on
  labels that had not arrived yet inflates your offline metric and produces a model
  that cannot be reproduced in production — the classic "great in backtest, useless
  live" failure.
  Defence: `label_as_of()` below, and never joining the raw `is_fraud` column directly.

Deliberate-leak exercise (ARCHITECTURE.md §11): once the honest pipeline works, add a
feature that violates this on purpose, watch PR-AUC jump to ~0.99, and write the ADR.
The point is to *feel* how good a leaked model looks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass(frozen=True, slots=True)
class AsOf:
    """The instant a feature is being computed for.

    Wrapping a timestamp in a type sounds like ceremony until you notice that
    `compute(df, ts)` and `compute(df, other_ts)` are indistinguishable at a call site,
    while `compute(df, AsOf(ts))` cannot be passed a stray `datetime.now()` by accident
    — and `datetime.now()` is exactly the value that turns a backtest into fiction.
    """

    at: datetime

    def mask(self, frame: pd.DataFrame, column: str = "submitted_at") -> pd.Series:
        """Boolean mask of rows visible at this instant. Strictly `<`, not `<=`: a
        claim may not be a feature of itself."""
        return frame[column] < self.at


def visible(frame: pd.DataFrame, as_of: AsOf, column: str = "submitted_at") -> pd.DataFrame:
    """The only sanctioned way to restrict history. Grep for it in review."""
    return frame.loc[as_of.mask(frame, column)]


def label_as_of(labels: pd.DataFrame, as_of: AsOf) -> pd.DataFrame:
    """Labels as the business knew them at `as_of`.

    Two filters, and both matter:

      * `confirmed_at <= as_of` — drop verdicts that had not been recorded yet.
      * return `observed_is_fraud`, never `is_fraud`.

    The second is the one that is easy to get wrong, because both columns sit in the
    same table and one of them makes the model look better. `is_fraud` is what actually
    happened and only the simulator knows it; `observed_is_fraud` is what an audit had
    concluded. Training on `is_fraud` means training on fraud nobody had caught yet —
    a model that cannot be rebuilt in production, and an offline metric that overstates
    real performance by the audit miss rate.

    Rows with a null `confirmed_at` are dropped rather than coerced to negative. In
    this world every claim eventually resolves, so nulls are rare, but the rule holds:
    "not yet decided" is not "clean".
    """
    if "observed_is_fraud" not in labels.columns:
        raise KeyError(
            "labels must carry `observed_is_fraud`; `is_fraud` is ground truth and is "
            "not a training target (see fsa_sim.world.entities.FraudLabel)"
        )
    confirmed = labels["confirmed_at"].notna() & (labels["confirmed_at"] <= as_of.at)
    out = labels.loc[confirmed].copy()
    out["is_fraud"] = out["observed_is_fraud"]
    return out


# ── the workhorse ───────────────────────────────────────────────────────────
# Every historical feature family in `fsa_ml.features` goes through this one
# function. That is deliberate: the `closed="left"` argument below is the single
# most important character sequence in the ML layer, and concentrating it here
# means it gets reviewed once rather than re-derived (and mis-derived) six times.


def prior_rolling(
    frame: pd.DataFrame,
    *,
    by: list[str],
    value: str,
    window: str,
    agg: str,
    time_col: str = "submitted_at",
    id_col: str = "expense_id",
) -> pd.Series:
    """Aggregate `value` over each group's **strictly earlier** rows in `window`.

    Returns a Series indexed by `id_col`, so callers can assign it straight onto a
    feature frame without worrying about row order.

    The `closed="left"` is what makes this point-in-time correct. Pandas' default for
    a time-based rolling window is `closed="right"`, which **includes the current
    row** — so `rolling("90D").mean()` on claim amounts puts the claim into its own
    baseline. The resulting z-score is shrunk toward zero for exactly the claims that
    should score highest, and the bug is invisible: the feature looks sane, the model
    trains, and the offline metric is wrong in a direction you cannot see.

    Rows sharing a timestamp with the current row are also excluded, which is the
    conservative choice: two claims submitted in the same minute cannot be features
    of each other without introducing an ordering the ledger does not guarantee.

    NaN is returned where there is no prior history, and NaN is **information** —
    it means cold start. Do not `fillna(0)` downstream: "this employee's mean claim
    is zero" is a different and false statement. LightGBM handles NaN natively.
    """
    ordered = frame.sort_values([*by, time_col])
    rolled = (
        ordered.set_index(time_col)
        .groupby(by, sort=True)[value]
        .rolling(window, closed="left")
        .agg(agg)
    )
    # groupby().rolling() emits groups in sorted key order and, within a group, in
    # index order — the same order as `ordered`. So positional alignment is valid,
    # and it avoids a MultiIndex join that would be both slower and easier to get
    # subtly wrong.
    return pd.Series(rolled.to_numpy(), index=ordered[id_col].to_numpy(), name=value)


def prior_count(
    frame: pd.DataFrame,
    *,
    by: list[str],
    window: str,
    time_col: str = "submitted_at",
    id_col: str = "expense_id",
) -> pd.Series:
    """Count of strictly-earlier rows in the window. Zero, not NaN, when there are
    none — a count of prior claims genuinely is zero for a new joiner, unlike a mean,
    which is genuinely undefined."""
    counted = prior_rolling(
        frame.assign(_one=1),
        by=by,
        value="_one",
        window=window,
        agg="count",
        time_col=time_col,
        id_col=id_col,
    )
    return counted.fillna(0.0)
