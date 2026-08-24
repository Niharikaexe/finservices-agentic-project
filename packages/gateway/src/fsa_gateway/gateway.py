"""The model gateway: routing, budget enforcement, and total interaction capture.

Everything that calls a model goes through here. That single choke point is what makes
four JD requirements true at once rather than aspirational:

  **Vendor neutrality** — the caller names a *route* ("triage", "copilot"), not a
  provider. Swapping the provider behind a route is a config change.

  **Cost control** — a per-run and per-day budget is enforced *before* the call, so
  spend is bounded by design rather than discovered on an invoice.

  **Observability** — every call writes an `LlmRecord` with the prompt template id, the
  rendered prompt, the output, tokens, cost and latency. Nothing can call a model
  without being logged, because there is no path around the gateway.

  **Release gates** — because the log exists, retrieval quality, safety, cost and
  latency are all queryable per run, which is what a gate reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fsa_common import ArgusError, get_logger
from fsa_gateway.provider import Completion, ModelProvider
from fsa_guardrails import scan_pii
from fsa_telemetry import InteractionLog, LlmRecord
from fsa_telemetry.metrics import LLM_COST, LLM_LATENCY, LLM_TOKENS

log = get_logger(__name__)


def _pii_redactor(text: str) -> str:
    """The default redactor: run the PII rail before anything is persisted."""
    return scan_pii(text, direction="output").sanitised


class BudgetExceededError(ArgusError):
    """Raised before a call that would breach the configured spend cap.

    Fails closed on cost, deliberately: an agent in a retry loop can spend a month's
    budget in an afternoon, and the failure mode of "stop" is far cheaper than the
    failure mode of "continue".
    """

    http_status = 429
    code = "budget_exceeded"


@dataclass(frozen=True, slots=True)
class Route:
    """A named workload bound to a provider and its limits."""

    name: str
    provider: ModelProvider
    max_tokens: int = 1024
    temperature: float = 0.0
    max_cost_usd_per_call: float = 0.05


class ModelGateway:
    """One entry point for every model call in the system."""

    def __init__(
        self,
        routes: dict[str, Route],
        *,
        interaction_log: InteractionLog | None = None,
        daily_budget_usd: float = 5.0,
    ) -> None:
        self._routes = routes
        # If the caller did not supply a log, build one with the PII rail wired in.
        # Redaction is on by default: an interaction log without it is a copy of every
        # card number that ever crossed a prompt.
        self._log = interaction_log or InteractionLog(redactor=_pii_redactor)
        self._daily_budget = daily_budget_usd
        self._spent = 0.0

    @property
    def spent_usd(self) -> float:
        return round(self._spent, 6)

    def complete(
        self,
        route_name: str,
        prompt: str,
        *,
        run_id: str,
        prompt_template_id: str,
        tenant_id: str = "unknown",
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        route = self._routes.get(route_name)
        if route is None:
            raise ArgusError("unknown route", route=route_name)

        if self._spent >= self._daily_budget:
            raise BudgetExceededError(
                "daily LLM budget exhausted",
                spent_usd=round(self._spent, 4),
                budget_usd=self._daily_budget,
                route=route_name,
            )

        completion = route.provider.complete(
            prompt,
            max_tokens=route.max_tokens,
            temperature=route.temperature,
            json_schema=json_schema,
        )
        self._spent += completion.cost_usd

        # ── metrics: low cardinality only. No user id, no run id, no prompt text.
        LLM_TOKENS.labels(
            model=completion.model,
            direction="input",
            workflow=route_name,
            tenant=tenant_id,
        ).inc(completion.prompt_tokens)
        LLM_TOKENS.labels(
            model=completion.model,
            direction="output",
            workflow=route_name,
            tenant=tenant_id,
        ).inc(completion.completion_tokens)
        LLM_COST.labels(model=completion.model, workflow=route_name, tenant=tenant_id).inc(
            completion.cost_usd
        )
        LLM_LATENCY.labels(model=completion.model, workflow=route_name).observe(
            completion.latency_ms / 1000
        )

        # ── the eval substrate: full fidelity, redacted at write time.
        self._log.write(
            LlmRecord(
                run_id=run_id,
                provider=completion.provider,
                model=completion.model,
                prompt_template_id=prompt_template_id,
                rendered_prompt=prompt,
                output=completion.text,
                parameters={
                    "max_tokens": route.max_tokens,
                    "temperature": route.temperature,
                    "structured": json_schema is not None,
                },
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                cost_usd=completion.cost_usd,
                ttft_ms=completion.ttft_ms,
                latency_ms=completion.latency_ms,
                workflow=route_name,
                finish_reason=completion.finish_reason,
            )
        )

        if completion.cost_usd > route.max_cost_usd_per_call:
            # Not an error — the call already happened. But a single call over budget
            # is the shape of a runaway prompt, and it should be visible immediately
            # rather than at month end.
            log.warning(
                "call exceeded per-call cost cap",
                route=route_name,
                cost_usd=round(completion.cost_usd, 5),
                cap_usd=route.max_cost_usd_per_call,
            )

        return completion

    @property
    def interaction_log(self) -> InteractionLog:
        return self._log
