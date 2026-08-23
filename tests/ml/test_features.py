"""The feature layer's one non-negotiable property: **causality**.

`test_features_are_causal` is the most valuable test in this repository. It asserts
that a claim's features are identical whether or not every later claim exists in the
frame. That single property catches nearly every leak — a forgotten `shift(1)`, a
`closed="right"` window, a `groupby().transform("mean")` over the whole dataset — and
it catches them without needing to know which mistake was made.

It is worth more than the features it guards, because features get rewritten and this
property does not change.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from fsa_ml.features import REGISTRY
from fsa_ml.features.build import build_features
from fsa_ml.features.pointintime import AsOf, prior_count, prior_rolling

AS_OF = AsOf(datetime(2027, 1, 1))


def _synthetic_claims(n: int = 240, seed: int = 3) -> pd.DataFrame:
    """A small frame with the columns every family needs."""
    rng = np.random.default_rng(seed)
    start = datetime(2026, 1, 1, 9, 0)
    users = [f"u{i:02d}" for i in range(8)]
    vendors = [f"v{i:02d}" for i in range(6)]
    categories = ["meals", "travel_air", "software", "mileage"]

    rows = []
    for i in range(n):
        submitted = start + timedelta(hours=int(rng.integers(1, 60)) * (i + 1) // 4)
        rows.append(
            {
                "expense_id": f"e{i:04d}",
                "tenant_id": "t00",
                "user_id": users[int(rng.integers(len(users)))],
                "department_id": f"d{int(rng.integers(3))}",
                "grade": int(rng.integers(1, 7)),
                "category": categories[int(rng.integers(len(categories)))],
                "vendor_id": vendors[int(rng.integers(len(vendors)))],
                "amount_minor": int(rng.lognormal(11.5, 0.8)),
                "transaction_date": (submitted - timedelta(days=int(rng.integers(0, 9)))).date(),
                "submitted_at": submitted,
                "receipt_phash": f"h{int(rng.integers(40)):03d}",
                "first_seen_on": (start - timedelta(days=int(rng.integers(30, 400)))).date(),
                "line_item_count": int(rng.integers(1, 5)),
                "attendee_count": int(rng.integers(1, 6)),
            }
        )
    return pd.DataFrame(rows).sort_values("submitted_at").reset_index(drop=True)


@pytest.mark.parametrize("family", sorted(REGISTRY))
def test_features_are_causal(family: str) -> None:
    """Delete the future; the past must not move.

    Concretely: featurise the whole frame, then featurise only the first 60% of it,
    and compare the rows they have in common. Any dependence on later data — however
    it got in — makes these disagree.
    """
    claims = _synthetic_claims()
    cutoff = int(len(claims) * 0.6)
    past = claims.iloc[:cutoff]

    full = REGISTRY[family](claims, AS_OF)
    truncated = REGISTRY[family](past, AS_OF)

    shared = truncated.index
    pd.testing.assert_frame_equal(
        full.loc[shared].sort_index(),
        truncated.sort_index(),
        check_dtype=False,
        obj=f"{family} leaked future information",
    )


def test_the_whole_assembled_matrix_is_causal() -> None:
    """Same property, over every family at once — catches a leak introduced by the
    concatenation itself rather than by any single family."""
    claims = _synthetic_claims()
    cutoff = int(len(claims) * 0.6)

    full = build_features(claims, AS_OF)
    truncated = build_features(claims.iloc[:cutoff], AS_OF)

    pd.testing.assert_frame_equal(
        full.loc[truncated.index].sort_index(),
        truncated.sort_index(),
        check_dtype=False,
    )


class TestPriorRolling:
    def test_excludes_the_current_row(self) -> None:
        """The `closed="left"` assertion, stated as plainly as possible."""
        frame = pd.DataFrame(
            {
                "expense_id": ["a", "b", "c"],
                "user_id": ["u1", "u1", "u1"],
                "submitted_at": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
                "amount_minor": [100, 200, 300],
            }
        )
        got = prior_rolling(frame, by=["user_id"], value="amount_minor", window="30D", agg="mean")
        assert pd.isna(got["a"])  # nothing before it
        assert got["b"] == 100.0  # only 'a'
        assert got["c"] == 150.0  # 'a' and 'b', not itself

    def test_respects_the_window(self) -> None:
        frame = pd.DataFrame(
            {
                "expense_id": ["a", "b"],
                "user_id": ["u1", "u1"],
                "submitted_at": pd.to_datetime(["2026-01-01", "2026-03-01"]),
                "amount_minor": [100, 200],
            }
        )
        got = prior_rolling(frame, by=["user_id"], value="amount_minor", window="7D", agg="mean")
        assert pd.isna(got["b"])  # 'a' is 60 days back, outside the window

    def test_does_not_mix_groups(self) -> None:
        frame = pd.DataFrame(
            {
                "expense_id": ["a", "b"],
                "user_id": ["u1", "u2"],
                "submitted_at": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "amount_minor": [100, 200],
            }
        )
        got = prior_rolling(frame, by=["user_id"], value="amount_minor", window="30D", agg="mean")
        assert pd.isna(got["b"])  # different employee, no shared history

    def test_count_is_zero_not_nan_without_history(self) -> None:
        """A mean with no history is undefined; a count with no history is zero.
        Conflating the two is how `fillna(0)` sneaks a false claim into the data."""
        frame = pd.DataFrame(
            {
                "expense_id": ["a"],
                "user_id": ["u1"],
                "submitted_at": pd.to_datetime(["2026-01-01"]),
            }
        )
        assert prior_count(frame, by=["user_id"], window="30D")["a"] == 0.0


def test_cold_start_is_nan_not_zero() -> None:
    """An employee with no history must produce NaN on ratio features, so LightGBM
    can route them down a dedicated branch instead of treating them as 'average'."""
    claims = _synthetic_claims(n=40)
    out = REGISTRY["employee_history"](claims, AS_OF)
    first_per_user = claims.groupby("user_id").head(1)["expense_id"]
    assert out.loc[first_per_user, "eh_amount_ratio_to_own_mean"].isna().all()
    assert (out.loc[first_per_user, "eh_is_cold_start"] == 1).all()
