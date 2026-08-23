"""Assemble the training matrix.

Two jobs, and the second one is where the money is:

  1. Join the claim, user and vendor tables into one frame, then run every registered
     feature family over it.
  2. Attach labels **as they were known at a stated moment** — not as they are known
     now.

Job 2 is the delayed-ground-truth constraint (ARCHITECTURE.md §11) made executable.
A model trained "as of 1 July" may only learn from fraud an audit had confirmed by
1 July. Roughly a fifth of the fraud in this world is confirmed later than that, and a
further slice is never confirmed at all. Training on all of it inflates the offline
metric and produces a model that cannot be reproduced in production — the classic
"great in backtest, useless live".

The API deliberately makes the honest thing easy and the dishonest thing require
typing `labels["is_fraud"]` yourself.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from fsa_ml.features import REGISTRY
from fsa_ml.features.pointintime import AsOf, label_as_of

# Columns a feature family may need. Joined once, here, rather than by each family.
_USER_COLUMNS = ["user_id", "department_id", "grade", "persona", "joined_on", "manager_id"]
_VENDOR_COLUMNS = ["vendor_id", "first_seen_on"]


@dataclass(frozen=True, slots=True)
class TrainingFrame:
    """Features, labels and the moment they were valid at.

    Carrying `as_of` on the artefact rather than leaving it in a caller's variable is
    the point: an experiment you cannot date is an experiment you cannot reproduce,
    and every MLflow run logs this field.
    """

    features: pd.DataFrame
    labels: pd.Series
    as_of: AsOf
    n_dropped_unconfirmed: int

    @property
    def positive_rate(self) -> float:
        return float(self.labels.mean())


def load_world(directory: Path) -> dict[str, pd.DataFrame]:
    """Read the simulator's Parquet output.

    `fsa_ml` reads the simulator's *files*, never imports its *code* — import-linter
    enforces the boundary. In production this function reads Postgres instead, and
    nothing downstream of it changes.
    """
    tables = {}
    for name in ("expenses", "users", "vendors", "fraud_labels"):
        path = directory / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} — run `make simulate` first")
        tables[name] = pd.read_parquet(path)
    return tables


def join_context(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Claims with the user and vendor attributes the feature families need."""
    claims = tables["expenses"].copy()
    claims["submitted_at"] = pd.to_datetime(claims["submitted_at"])

    # Drop columns the claim already carries before merging. Letting pandas resolve the
    # collision with _x/_y suffixes would silently rename `department_id`, and a feature
    # family then fails on a KeyError three modules away — or worse, quietly picks up
    # the wrong copy. `validate="many_to_one"` is the other half: it turns a duplicated
    # user row (which would silently multiply the claim count) into an exception.
    users = tables["users"][_USER_COLUMNS].drop(
        columns=[c for c in _USER_COLUMNS if c != "user_id" and c in claims.columns]
    )
    vendors = tables["vendors"][_VENDOR_COLUMNS].drop(
        columns=[c for c in _VENDOR_COLUMNS if c != "vendor_id" and c in claims.columns]
    )
    return claims.merge(users, on="user_id", how="left", validate="many_to_one").merge(
        vendors, on="vendor_id", how="left", validate="many_to_one"
    )


def build_features(
    claims: pd.DataFrame, as_of: AsOf, families: list[str] | None = None
) -> pd.DataFrame:
    """Run every registered family and concatenate on `expense_id`.

    Families are run over the **whole** claim history, not over a pre-filtered slice.
    That is intentional: each family windows per row via `prior_rolling`, so a claim
    submitted in March sees only February's data whether or not December's rows are
    present in the frame. Pre-filtering to a training window would instead give the
    first claims in that window an artificially empty history — a subtle bias that
    makes the model look worse at the start of every retrain.
    """
    selected = families or list(REGISTRY)
    parts = [REGISTRY[name](claims, as_of) for name in selected]
    return pd.concat(parts, axis=1)


def assemble(
    directory: Path,
    as_of: datetime,
    *,
    families: list[str] | None = None,
) -> TrainingFrame:
    """The whole pipeline: load, join, featurise, and attach labels known at `as_of`.

    Rows whose label was not yet confirmed at `as_of` are **dropped, not zeroed**.
    Calling them negatives would teach the model that recent claims are clean, which
    is precisely backwards: recent claims are the ones it most needs to score.
    """
    at = AsOf(as_of)
    tables = load_world(directory)
    claims = join_context(tables)

    features = build_features(claims, at, families)

    known = label_as_of(tables["fraud_labels"], at).set_index("expense_id")["is_fraud"]
    submitted = claims.set_index("expense_id")["submitted_at"]

    # A claim is usable for training when (a) it was submitted before `as_of`, and
    # (b) its label had been confirmed by `as_of`. Unconfirmed claims — including the
    # genuinely fraudulent ones an audit never reached — are excluded entirely.
    eligible = submitted.index[submitted < at.at]
    usable = known.index.intersection(eligible)

    dropped = len(eligible) - len(usable)
    return TrainingFrame(
        features=features.loc[usable],
        labels=known.loc[usable].astype("int8"),
        as_of=at,
        n_dropped_unconfirmed=dropped,
    )


def assemble_evaluation(
    directory: Path,
    train_as_of: datetime,
    eval_as_of: datetime,
    *,
    families: list[str] | None = None,
) -> tuple[TrainingFrame, TrainingFrame]:
    """A temporally honest train/test split.

    Train on claims submitted before `train_as_of`, with labels confirmed by then.
    Evaluate on claims submitted **after** it, with labels confirmed by the later
    `eval_as_of` — because in production you wait for the audits to come in before you
    can score last quarter's model.

    A random `train_test_split` here would be wrong twice over: it leaks future
    behaviour into the past through the rolling features, and it evaluates the model
    on a period it was trained on. Both flatter the model.
    """
    train = assemble(directory, train_as_of, families=families)

    full = assemble(directory, eval_as_of, families=families)
    tables = load_world(directory)
    submitted = pd.to_datetime(tables["expenses"].set_index("expense_id")["submitted_at"])
    holdout = submitted.index[submitted >= train_as_of]
    keep = full.features.index.intersection(holdout)

    test = TrainingFrame(
        features=full.features.loc[keep],
        labels=full.labels.loc[keep],
        as_of=AsOf(eval_as_of),
        n_dropped_unconfirmed=full.n_dropped_unconfirmed,
    )
    return train, test
