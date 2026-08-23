"""Feature layer.

Families are registered here so the training and serving paths build the identical
set — the cheapest defence against training/serving skew is having exactly one list of
features in the codebase.

Each family is a `(claims, as_of) -> DataFrame indexed by expense_id` callable with a
column prefix naming the family. The prefix is not decoration: when SHAP hands an
investigator the top contributors for a flagged claim, `vel_near_threshold_7d` says
"this looks like a split" in a way `f_47` does not, and the agent's `fraud_reasons`
list becomes something a human can act on.

| Prefix | Family | Typology it exists to catch |
|---|---|---|
| `ci_`   | claim_intrinsic    | PERSONAL (weekend, attendees), round amounts |
| `eh_`   | employee_history   | INFLATION |
| `vel_`  | velocity           | SPLIT |
| `dup_`  | duplicate_signals  | DUPLICATE |
| `peer_` | peer_relative      | INFLATION by a habitual inflater, cold start |
| `ven_`  | vendor             | GHOST_VENDOR, COLLUSION |
"""

from collections.abc import Callable

import pandas as pd

from fsa_ml.features import (
    claim_intrinsic,
    duplicate_signals,
    employee_history,
    peer_relative,
    velocity,
    vendor,
)
from fsa_ml.features.pointintime import AsOf, label_as_of, prior_count, prior_rolling, visible

FeatureFamily = Callable[[pd.DataFrame, AsOf], pd.DataFrame]

REGISTRY: dict[str, FeatureFamily] = {
    "claim_intrinsic": claim_intrinsic.build,
    "employee_history": employee_history.build,
    "velocity": velocity.build,
    "duplicate_signals": duplicate_signals.build,
    "peer_relative": peer_relative.build,
    "vendor": vendor.build,
}

__all__ = [
    "REGISTRY",
    "AsOf",
    "FeatureFamily",
    "label_as_of",
    "prior_count",
    "prior_rolling",
    "visible",
]
