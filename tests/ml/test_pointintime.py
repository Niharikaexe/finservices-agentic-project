"""Point-in-time helpers. Small file, disproportionate importance: everything in §11
rests on these two functions being right."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from fsa_ml.features.pointintime import AsOf, label_as_of, visible


def _claims() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "expense_id": ["e1", "e2", "e3"],
            "submitted_at": pd.to_datetime(
                ["2026-01-01 09:00", "2026-02-01 09:00", "2026-03-01 09:00"]
            ),
        }
    )


def test_visible_excludes_the_boundary_row() -> None:
    """Strictly `<`: a claim is never part of its own history."""
    got = visible(_claims(), AsOf(datetime(2026, 2, 1, 9, 0)))
    assert list(got["expense_id"]) == ["e1"]


def test_visible_includes_everything_strictly_earlier() -> None:
    got = visible(_claims(), AsOf(datetime(2026, 3, 1, 9, 1)))
    assert list(got["expense_id"]) == ["e1", "e2", "e3"]


def test_label_as_of_drops_unconfirmed_and_future_labels() -> None:
    labels = pd.DataFrame(
        {
            "expense_id": ["e1", "e2", "e3"],
            "is_fraud": [True, True, True],
            "observed_is_fraud": [True, True, True],
            "confirmed_at": pd.to_datetime(["2026-01-20", None, "2026-04-01"]),
        }
    )
    got = label_as_of(labels, AsOf(datetime(2026, 2, 1)))
    # e2 was never resolved and e3 resolves later: neither is a usable label on 1 Feb,
    # and crucially neither becomes a *negative*.
    assert list(got["expense_id"]) == ["e1"]


def test_label_as_of_returns_the_observed_label_not_ground_truth() -> None:
    """The one that stops a model training on fraud nobody had caught yet.

    e2 is genuinely fraudulent and the audit missed it, so the business recorded it
    clean. Training must see the clean record — that is the data production would
    actually have — and the resulting offline metric is honest about the miss.
    """
    labels = pd.DataFrame(
        {
            "expense_id": ["e1", "e2"],
            "is_fraud": [True, True],
            "observed_is_fraud": [True, False],
            "confirmed_at": pd.to_datetime(["2026-01-10", "2026-01-05"]),
        }
    )
    got = label_as_of(labels, AsOf(datetime(2026, 2, 1))).set_index("expense_id")
    assert bool(got.loc["e1", "is_fraud"]) is True
    assert bool(got.loc["e2", "is_fraud"]) is False


def test_label_as_of_refuses_a_table_without_the_observed_column() -> None:
    """Fail loudly rather than silently falling back to ground truth."""
    labels = pd.DataFrame(
        {
            "expense_id": ["e1"],
            "is_fraud": [True],
            "confirmed_at": pd.to_datetime(["2026-01-10"]),
        }
    )
    with pytest.raises(KeyError, match="observed_is_fraud"):
        label_as_of(labels, AsOf(datetime(2026, 2, 1)))
