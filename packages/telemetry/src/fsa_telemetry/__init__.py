"""fsa_telemetry — one OpenTelemetry pipeline, two backends (Phoenix + Prometheus).

Metric *definitions* live in `metrics.py`; OTel wiring lands in M2.5.
"""

from fsa_telemetry import metrics

__all__ = ["metrics"]
