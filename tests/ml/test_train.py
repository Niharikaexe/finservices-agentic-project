"""The contract for `fsa_ml.fraud.train.train` — your task.

Skipped while `train()` raises NotImplementedError; runs automatically once you
implement it. Requires the ML dependency group: `make install-ml`.

Read the assertions before writing the implementation. They are the spec, and two of
them (`test_beats_the_amount_baseline`, `test_calibration_improves_brier`) are the
ones that would actually block a release.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from fsa_ml.features.build import TrainingFrame
from fsa_ml.features.pointintime import AsOf
from fsa_ml.fraud.metrics import evaluate
from fsa_ml.fraud.train import amount_baseline, train

lgb = pytest.importorskip("lightgbm", reason="needs the `ml` group: make install-ml")


def _synthetic_frames() -> tuple[TrainingFrame, TrainingFrame]:
    """A small learnable problem with the right shape: ~2% positives, informative
    features plus noise, and a temporal train/test split."""
    rng = np.random.default_rng(11)
    n = 6000
    amount = rng.lognormal(11.5, 0.9, n)
    velocity = rng.poisson(1.5, n).astype(float)
    zscore = rng.normal(0, 1, n)

    logit = -4.6 + 0.55 * (zscore > 1.5) * 3 + 0.9 * (velocity > 3) + 0.4 * np.log1p(amount) / 3
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)

    features = pd.DataFrame(
        {
            "ci_amount_minor": amount,
            "ci_amount_log": np.log1p(amount),
            "vel_claims_7d": velocity,
            "eh_amount_zscore_cat_90d": zscore,
            "noise_a": rng.normal(0, 1, n),
            "noise_b": rng.normal(0, 1, n),
        },
        index=[f"e{i:05d}" for i in range(n)],
    )
    labels = pd.Series(y, index=features.index, name="is_fraud").astype("int8")

    cut = int(n * 0.7)
    return (
        TrainingFrame(features.iloc[:cut], labels.iloc[:cut], AsOf(datetime(2026, 9, 1)), 0),
        TrainingFrame(features.iloc[cut:], labels.iloc[cut:], AsOf(datetime(2027, 6, 1)), 0),
    )


def _is_stub() -> bool:
    try:
        train(*_synthetic_frames(), n_estimators=5)
    except NotImplementedError:
        return True
    except Exception:
        return False
    return False


pytestmark = pytest.mark.skipif(_is_stub(), reason="implement fsa_ml.fraud.train.train")


@pytest.fixture(scope="module")
def frames() -> tuple[TrainingFrame, TrainingFrame]:
    return _synthetic_frames()


@pytest.fixture(scope="module")
def model(frames: tuple[TrainingFrame, TrainingFrame]):  # type: ignore[no-untyped-def]
    return train(*frames, n_estimators=200)


def test_predictions_are_probabilities(model, frames) -> None:  # type: ignore[no-untyped-def]
    scores = model.predict(frames[1].features)
    assert scores.shape == (len(frames[1].features),)
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_feature_names_are_recorded_in_order(model, frames) -> None:  # type: ignore[no-untyped-def]
    """Training/serving skew usually enters as a reordered column. Pinning the order
    on the artefact is what makes the mismatch loud instead of silent."""
    assert model.feature_names == list(frames[0].features.columns)


def test_beats_the_amount_baseline(model, frames) -> None:  # type: ignore[no-untyped-def]
    """The gate that matters. A model that cannot beat a three-line ranking rule is
    not worth the serving cost, and shipping one anyway is extremely common."""
    y = frames[1].labels.to_numpy()
    model_pr = evaluate(y, model.predict(frames[1].features)).pr_auc
    baseline_pr = evaluate(y, amount_baseline(frames[1].features)).pr_auc
    assert model_pr > baseline_pr * 1.15


def test_metrics_are_computed_on_held_out_data(model, frames) -> None:  # type: ignore[no-untyped-def]
    assert model.metrics.n == len(frames[1].labels)
    assert model.metrics.n_positive == int(frames[1].labels.sum())


def test_calibration_improves_brier(model) -> None:  # type: ignore[no-untyped-def]
    """An uncalibrated boosted score is a ranking, not a probability. The agent puts
    this number in a prompt, so the difference is not cosmetic."""
    if model.metrics_uncalibrated is None:
        pytest.skip("record metrics_uncalibrated to assert the calibration gain")
    assert model.metrics.brier <= model.metrics_uncalibrated.brier


def test_training_is_reproducible(frames) -> None:  # type: ignore[no-untyped-def]
    a = train(*frames, seed=7, n_estimators=60)
    b = train(*frames, seed=7, n_estimators=60)
    np.testing.assert_allclose(a.predict(frames[1].features), b.predict(frames[1].features))
