"""Point-in-time helpers. Small file, disproportionate importance: everything in §11
rests on these two functions being right."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

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
            "confirmed_at": pd.to_datetime(["2026-01-20", None, "2026-04-01"]),
        }
    )
    got = label_as_of(labels, AsOf(datetime(2026, 2, 1)))
    # e2 is never audited and e3 is confirmed later: neither is a usable label on
    # 1 Feb, and crucially neither becomes a *negative*.
    assert list(got["expense_id"]) == ["e1"]
