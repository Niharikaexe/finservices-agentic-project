"""Fraud model training.

    ┌──────────────────────────────────────────────────────────────────────┐
    │  YOUR TASK.                                                          │
    │  `train()` below raises NotImplementedError.                         │
    │  The contract is tests/ml/test_train.py, which is skipped until you   │
    │  implement it and then runs automatically.                            │
    │  Feature assembly, metrics and the baseline are done — use them.       │
    └──────────────────────────────────────────────────────────────────────┘

Needs the heavy ML group:  `make install-ml`  then  `make train-fraud`

Two things this file exists to teach, neither of which is "how to call LightGBM":

**Always ship a baseline you have to beat.** `seasonal_naive`-style thinking applies to
classification too. `amount_baseline` below scores every claim by its within-category
amount percentile — three lines, no training. If the model cannot beat it by a clear
margin, the model is not earning its serving cost, and a large fraction of production
ML projects never clear this bar. Reporting both numbers is the honest thing and it is
also the thing interviewers remember.

**A raw LightGBM score is not a probability.** Gradient boosting with a heavily
imbalanced target produces scores that rank well and calibrate badly. The agent puts
this number into an LLM prompt ("fraud score 0.7"), so miscalibration is not a rounding
issue — the downstream reasoning is wrong. Fit a calibrator on data the booster never
saw, and report Brier and ECE before and after.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from fsa_ml.features.build import TrainingFrame, assemble_evaluation
from fsa_ml.fraud.metrics import FraudMetrics, evaluate  # noqa: F401  — used by train()


@dataclass(frozen=True, slots=True)
class TrainedModel:
    """What a training run produces. Everything needed to score and to audit."""

    booster: Any  # lightgbm.Booster
    calibrator: Any  # sklearn IsotonicRegression or CalibratedClassifierCV
    feature_names: list[str]
    train_as_of: datetime
    metrics: FraudMetrics
    baseline_metrics: FraudMetrics
    metrics_uncalibrated: FraudMetrics | None = None

    def predict(self, features: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Calibrated fraud probability. Column order is taken from `feature_names`,
        not from the frame — training/serving skew usually enters as a reordered or
        silently-missing column, and this makes that a KeyError instead of a wrong
        score."""
        raw = self.booster.predict(features[self.feature_names])
        return np.asarray(self.calibrator.predict(raw), dtype=float)


# ── the baseline you have to beat ───────────────────────────────────────────


def amount_baseline(features: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Score = the claim's amount percentile within its own category.

    No training, no tuning, no serving infrastructure. This is the bar.
    """
    amount = features["ci_amount_minor"].astype(float)
    # `ci_amount_log` is monotone in amount, so ranking on either is identical; the
    # percentile is taken globally here because category is not a feature column.
    return np.asarray(amount.rank(pct=True).to_numpy(), dtype=np.float64)


# ── YOUR TASK ───────────────────────────────────────────────────────────────


def train(
    train_frame: TrainingFrame,
    test_frame: TrainingFrame,
    *,
    seed: int = 42,
    n_estimators: int = 400,
) -> TrainedModel:
    """Fit, calibrate and evaluate the fraud model.

    Contract (tests/ml/test_train.py):
      * returns a `TrainedModel` whose `predict` gives values in [0, 1]
      * `feature_names` matches the training frame's columns, in order
      * `metrics` is computed on `test_frame` only — never on training data
      * calibration improves Brier score versus the uncalibrated booster
      * PR-AUC beats `amount_baseline` on the same test set
      * training twice with the same seed gives identical predictions

    Suggested shape:

        import lightgbm as lgb
        from sklearn.isotonic import IsotonicRegression

        X, y = train_frame.features, train_frame.labels

        # 1. Hold out a calibration slice the booster never sees. Split it by TIME,
        #    not randomly — the rolling features mean a random split leaks.
        cut = int(len(X) * 0.85)
        X_fit, y_fit = X.iloc[:cut], y.iloc[:cut]
        X_cal, y_cal = X.iloc[cut:], y.iloc[cut:]

        # 2. Fit. `is_unbalance` (or scale_pos_weight) matters at a 1.7% base rate.
        booster = lgb.train(
            params={
                "objective": "binary",
                "metric": "average_precision",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_data_in_leaf": 50,       # guards against a leaf per fraudster
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "is_unbalance": True,
                "seed": seed,
                "verbosity": -1,
            },
            train_set=lgb.Dataset(X_fit, label=y_fit),
            num_boost_round=n_estimators,
        )

        # 3. Calibrate on the held-out slice. Isotonic is non-parametric and needs a
        #    few hundred positives; Platt (LogisticRegression on the logit) is the
        #    lower-variance choice when positives are scarce.
        raw_cal = booster.predict(X_cal)
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, y_cal)

        # 4. Evaluate on the temporally held-out test frame, and against the baseline.
        raw_test = booster.predict(test_frame.features[list(X.columns)])
        metrics = evaluate(test_frame.labels.to_numpy(), calibrator.predict(raw_test))
        baseline = evaluate(test_frame.labels.to_numpy(),
                            amount_baseline(test_frame.features))

    Decisions to make yourself, and to justify in a comment:
      1. `is_unbalance=True` reweights the loss. It usually improves ranking and always
         wrecks calibration — which you then fix in step 3. Is that the right order, or
         would you rather leave the weights alone and keep the raw scores meaningful?
      2. The calibration slice is 15% of training data the booster does not get to
         learn from. At ~840 positives, that is ~126 positives for calibration. Is
         isotonic stable at that size, or is Platt the safer call?
      3. `num_boost_round` is fixed at 400. Where would early stopping get its
         validation set from, without leaking into either the calibration slice or the
         test set?
    """
    raise NotImplementedError("Implement train() — see the contract above")


def train_from_disk(
    world_dir: Path,
    train_as_of: datetime,
    eval_as_of: datetime,
    *,
    seed: int = 42,
) -> TrainedModel:
    """Convenience wrapper: assemble features from a generated world, then train."""
    train_frame, test_frame = assemble_evaluation(world_dir, train_as_of, eval_as_of)
    return train(train_frame, test_frame, seed=seed)
