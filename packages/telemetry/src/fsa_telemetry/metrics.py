"""The single Prometheus metric registry (ARCHITECTURE.md §13).

Every metric in Argus is declared here and nowhere else. That is not tidiness for its
own sake: Prometheus cost is driven by *cardinality* (series = metric x every label
value combination), and cardinality is only reviewable if the label sets sit on one
screen. A `user_id` label on a counter is how a 200-series metric becomes 2,000,000.

Cardinality rules — enforced in review:
  NEVER label with user_id, expense_id, session_id, vendor_name, or any raw text.
  `tenant` is allowed only because the simulator creates ~5 of them.
  Per-entity forensics is Phoenix's job, not Prometheus's.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ── LLM / agent ─────────────────────────────────────────────────────────────
LLM_TOKENS = Counter(
    "llm_tokens_total", "Tokens consumed", ["model", "direction", "workflow", "tenant"]
)
LLM_COST = Counter("llm_cost_usd_total", "Estimated spend", ["model", "workflow", "tenant"])
LLM_LATENCY = Histogram(
    "llm_request_duration_seconds",
    "Provider latency",
    ["model", "workflow"],
    buckets=(0.25, 0.5, 1, 2, 4, 8, 16, 32, 60),
)
TTFT = Histogram("llm_time_to_first_token_seconds", "Time to first token", ["model"])

TOOL_CALLS = Counter(
    "agent_tool_calls_total", "Tool invocations", ["tool", "outcome"]
)  # outcome: ok|error|denied|timeout
GUARDRAIL_TRIPS = Counter(
    "guardrail_trips_total", "Rail activations", ["rail", "direction", "action"]
)
GUARDRAIL_LATENCY = Histogram("guardrail_duration_seconds", "Rail cost", ["rail"])
AUTHZ_DECISIONS = Counter(
    "authz_decisions_total",
    "OpenFGA/OPA checks",
    ["subject_role", "resource_type", "decision"],
)

# ── HITL / durable workflow ─────────────────────────────────────────────────
HITL_PENDING = Gauge("hitl_approvals_pending", "Parked on human", ["workflow"])
HITL_WAIT = Histogram(
    "hitl_wait_seconds",
    "Time awaiting human",
    ["workflow"],
    buckets=(30, 120, 300, 900, 3600, 14400, 86400),
)
GRAPH_ACTIVE = Gauge("langgraph_runs_active", "In-flight runs", ["workflow", "node"])
GRAPH_RESUMES = Counter("langgraph_resumes_total", "Checkpoint resumes", ["workflow"])

# ── Expense domain ──────────────────────────────────────────────────────────
EXPENSES = Counter("expenses_total", "Submitted", ["tenant", "category", "outcome"])
EXPENSE_AMOUNT = Histogram("expense_amount_minor", "Claim size", ["tenant", "category"])
POLICY_VIOLATIONS = Counter("policy_violations_total", "Violations", ["rule_ref", "severity"])
BUDGET_UTILISATION = Gauge("budget_utilisation_ratio", "Spend/limit", ["tenant", "department"])

# ── ML ──────────────────────────────────────────────────────────────────────
PREDICTIONS = Counter("model_predictions_total", "Scored", ["model", "version"])
SCORE_DIST = Histogram("model_score", "Prediction distribution", ["model", "version"])
DRIFT = Gauge("model_drift_score", "PSI/KS", ["model", "feature", "method"])  # pushed
PERF_DELAYED = Gauge(
    "model_performance_delayed", "Backfilled metric", ["model", "metric", "lag_days"]
)  # pushed
MODEL_INFO = Gauge(
    "model_serving_version_info", "Deployed version", ["model", "version", "stage"]
)  # always 1; gives Grafana deploy annotations

__all__ = [
    "AUTHZ_DECISIONS",
    "BUDGET_UTILISATION",
    "DRIFT",
    "EXPENSES",
    "EXPENSE_AMOUNT",
    "GRAPH_ACTIVE",
    "GRAPH_RESUMES",
    "GUARDRAIL_LATENCY",
    "GUARDRAIL_TRIPS",
    "HITL_PENDING",
    "HITL_WAIT",
    "LLM_COST",
    "LLM_LATENCY",
    "LLM_TOKENS",
    "MODEL_INFO",
    "PERF_DELAYED",
    "POLICY_VIOLATIONS",
    "PREDICTIONS",
    "SCORE_DIST",
    "TOOL_CALLS",
    "TTFT",
]
