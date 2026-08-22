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
    """Labels that had actually been confirmed by `as_of`.

    Rows whose `confirmed_at` is null (never audited) or later than `as_of` are dropped
    — not treated as negatives. Treating "not yet confirmed" as "not fraud" is its own
    subtle leak in the opposite direction: it teaches the model that recent claims are
    clean, so it under-scores exactly the traffic you most need scored.
    """
    confirmed = labels["confirmed_at"].notna() & (labels["confirmed_at"] <= as_of.at)
    return labels.loc[confirmed]
