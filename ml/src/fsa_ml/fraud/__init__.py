"""fsa_ml.fraud — training, evaluation and calibration for the fraud model (M4).

Serving lives in `services/fraud-scorer`, which loads the exported ONNX artefact and
never imports this module. Training and serving share only the feature registry, which
is the one thing they *must* share.
"""

from fsa_ml.fraud.metrics import FraudMetrics, evaluate, expected_calibration_error, recall_at_k
from fsa_ml.fraud.train import TrainedModel, amount_baseline, train, train_from_disk

__all__ = [
    "FraudMetrics",
    "TrainedModel",
    "amount_baseline",
    "evaluate",
    "expected_calibration_error",
    "recall_at_k",
    "train",
    "train_from_disk",
]
