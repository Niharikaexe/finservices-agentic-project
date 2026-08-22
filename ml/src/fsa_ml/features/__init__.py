"""Feature layer. Families are registered here so the training and serving paths
build the identical set — the cheapest defence against training/serving skew is
having exactly one list of features in the codebase."""

from collections.abc import Callable

import pandas as pd

from fsa_ml.features import claim_intrinsic, employee_history
from fsa_ml.features.pointintime import AsOf, label_as_of, visible

FeatureFamily = Callable[[pd.DataFrame, AsOf], pd.DataFrame]

REGISTRY: dict[str, FeatureFamily] = {
    "claim_intrinsic": claim_intrinsic.build,
    "employee_history": employee_history.build,
    # peer_relative, duplicate_signals, velocity, vendor, merchant_text -> M4
}

__all__ = ["REGISTRY", "AsOf", "FeatureFamily", "label_as_of", "visible"]
