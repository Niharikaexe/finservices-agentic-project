"""fsa_telemetry — one OpenTelemetry pipeline, two backends (Phoenix + Prometheus),
plus the append-only interaction log the eval pipeline is built on.

Metric *definitions* live in `metrics.py`; interaction capture in `interactions.py`.
"""

from fsa_telemetry import metrics
from fsa_telemetry.interactions import (
    SCHEMA_VERSION,
    GuardrailRecord,
    InteractionLog,
    LlmRecord,
    Record,
    RetrievalRecord,
    ToolRecord,
    new_run_id,
)

__all__ = [
    "SCHEMA_VERSION",
    "GuardrailRecord",
    "InteractionLog",
    "LlmRecord",
    "Record",
    "RetrievalRecord",
    "ToolRecord",
    "metrics",
    "new_run_id",
]
