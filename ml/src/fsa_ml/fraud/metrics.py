"""Evaluation metrics for a severely imbalanced problem.

Accuracy is banned here and the ban is not stylistic. At a 1.7% positive rate, a model
that predicts "clean" for every claim is 98.3% accurate and detects nothing. Every
metric below is chosen because it survives imbalance.

  PR-AUC          area under precision-recall. Unlike ROC-AUC it does not get flattered
                  by the enormous negative class — a model can have ROC-AUC 0.95 and
                  PR-AUC 0.15 on a problem like this, and the second number is the true
                  one.
  recall@k        of the k claims we actually have the human capacity to investigate,
                  how much fraud did we surface? This is the number a finance director
                  cares about, because k is their headcount.
  Brier / ECE     calibration. The agent consumes the score inside an LLM prompt, so an
                  uncalibrated 0.7 is not just imprecise, it is *meaningless* to the
                  thing reading it (§11).
  cost@threshold  the business decision. Missing a ₹2,00,000 fraud is not the same
                  loss as a false positive on a ₹300 coffee, and a threshold picked
                  without saying so is a business decision made by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass(frozen=True, slots=True)
class FraudMetrics:
    pr_auc: float
    roc_auc: float
    brier: float
    ece: float
    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    n: int = 0
    n_positive: int = 0

    def render(self) -> str:
        lines = [
            f"  n={self.n:,}  positives={self.n_positive:,} "
            f"({self.n_positive / max(self.n, 1):.2%})",
            f"  PR-AUC  {self.pr_auc:.4f}   ROC-AUC {self.roc_auc:.4f}",
            f"  Brier   {self.brier:.4f}   ECE     {self.ece:.4f}",
        ]
        for k in sorted(self.recall_at_k):
            lines.append(
                f"  @k={k:<5} recall {self.recall_at_k[k]:.3f}  "
                f"precision {self.precision_at_k[k]:.3f}"
            )
        return "\n".join(lines)


def expected_calibration_error(
    y_true: npt.NDArray[np.int_], y_score: npt.NDArray[np.float64], n_bins: int = 10
) -> float:
    """Mean gap between predicted probability and observed frequency, per bin.

    Quantile bins rather than equal-width: with scores piled up near zero, equal-width
    bins put 99% of the mass in the first bucket and report a meaninglessly small
    error.
    """
    frame = pd.DataFrame({"y": y_true, "p": y_score})
    try:
        frame["bin"] = pd.qcut(frame["p"], n_bins, duplicates="drop")
    except ValueError:  # degenerate scores, e.g. a constant predictor
        return float(abs(frame["p"].mean() - frame["y"].mean()))
    grouped = frame.groupby("bin", observed=True).agg(
        predicted=("p", "mean"), actual=("y", "mean"), weight=("y", "size")
    )
    gap = (grouped["predicted"] - grouped["actual"]).abs()
    return float((gap * grouped["weight"]).sum() / grouped["weight"].sum())


def recall_at_k(
    y_true: npt.NDArray[np.int_], y_score: npt.NDArray[np.float64], k: int
) -> tuple[float, float]:
    """Recall and precision within the top-k highest-scoring claims."""
    if k <= 0 or len(y_true) == 0:
        return 0.0, 0.0
    k = min(k, len(y_true))
    top = np.argsort(-y_score)[:k]
    hits = float(y_true[top].sum())
    total = float(y_true.sum())
    return (hits / total if total else 0.0), hits / k


def evaluate(
    y_true: npt.NDArray[np.int_],
    y_score: npt.NDArray[np.float64],
    ks: tuple[int, ...] = (50, 200, 1000),
) -> FraudMetrics:
    """The full report. Call this on the temporally held-out set, never on train."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    recalls, precisions = {}, {}
    for k in ks:
        recalls[k], precisions[k] = recall_at_k(y_true, y_score, k)
    return FraudMetrics(
        pr_auc=float(average_precision_score(y_true, y_score)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        brier=float(brier_score_loss(y_true, y_score)),
        ece=expected_calibration_error(y_true, y_score),
        recall_at_k=recalls,
        precision_at_k=precisions,
        n=len(y_true),
        n_positive=int(y_true.sum()),
    )
