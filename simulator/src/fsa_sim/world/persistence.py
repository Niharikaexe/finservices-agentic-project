"""Persist a `World` to Parquet, one file per entity.

Parquet rather than CSV for three reasons that matter later: it keeps dtypes (a date
stays a date, so no `pd.to_datetime` guessing at load), it is columnar (the feature
pipeline reads three columns out of twenty without parsing the rest), and it is what
the Azure ML / MLflow path expects anyway.

One directory per world, named by seed, so multiple worlds coexist:
    data/worlds/seed-42/{tenants,departments,users,vendors,budgets,expenses}.parquet
"""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fsa_sim.world.entities import World

_TABLES = (
    "tenants",
    "departments",
    "users",
    "vendors",
    "budgets",
    "expenses",
    "fraud_labels",
    "injection_labels",
)


def _to_frame(rows: list[Any]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    first = rows[0]
    if not is_dataclass(first):
        raise TypeError(f"expected dataclasses, got {type(first)!r}")
    columns = [f.name for f in fields(first)]
    return pd.DataFrame([asdict(r) for r in rows], columns=columns)


def save_world(world: World, out_dir: Path) -> dict[str, int]:
    """Write every non-empty table. Returns row counts, for logging."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for table in _TABLES:
        frame = _to_frame(getattr(world, table))
        if frame.empty:
            continue
        frame.to_parquet(out_dir / f"{table}.parquet", index=False)
        counts[table] = len(frame)
    return counts


def load_table(out_dir: Path, table: str) -> pd.DataFrame:
    path = out_dir / f"{table}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} — generate the world first (`make simulate`)")
    return pd.read_parquet(path)
