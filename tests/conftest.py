"""Shared fixtures.

Note the seeding discipline: every test that needs randomness takes the `rng` fixture,
which is a fresh `default_rng(1234)`. No test calls `np.random.*` directly, so tests
cannot influence each other through global RNG state — a class of flake that is
miserable to debug once it appears.
"""

from __future__ import annotations

import numpy as np
import pytest

from fsa_sim.world import WorldConfig
from fsa_sim.world.config import TENANT_SHAPES


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def small_config() -> WorldConfig:
    """A deliberately small world, so the suite stays fast."""
    from datetime import date

    return WorldConfig(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 31),
        shapes=(TENANT_SHAPES[0],),
        seed=7,
    )
