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

import threading
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from fsa_common import ArgusError, get_logger
from fsa_gateway.provider import Completion, ModelProvider
from fsa_guardrails import scan_pii
from fsa_telemetry import GuardrailRecord, InteractionLog, LlmRecord
from fsa_telemetry.metrics import LLM_COST, LLM_LATENCY, LLM_TOKENS

log = get_logger(__name__)


def _pii_redactor(text: str) -> str:
    """The default redactor: run the PII rail before anything is persisted.

    `direction="log_scrub"` rather than `"output"` on purpose. Scrubbing runs once per
    string field per record, so counting it as an output-rail evaluation made
    `guardrail_trips_total{rail="pii",direction="output"}` a function of log volume —
    it read 769 against 42 requests. That inflated series would have made
    `GuardrailBypassSuspected` unable to fire for the PII rail no matter what, and it
    poisoned the baseline `PromptInjectionSpike` compares against.
    """
    return scan_pii(text, direction="log_scrub").sanitised


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
        # A real date, not "since this process started". Without it the budget was
        # neither daily nor durable: a long-lived process was capped at $5 forever,
        # and a restart — which scale-to-zero makes routine — granted a fresh $5.
        self._spent_on: date = datetime.now(UTC).date()
        # check-and-reserve must be atomic. 50 concurrent $1 calls against a $5 cap
        # previously spent $50 with zero denials.
        self._lock = threading.Lock()

    @property
    def spent_usd(self) -> float:
        return round(self._spent, 6)

    def _reserve(self, estimate: float, route_name: str) -> None:
        """Reserve budget BEFORE the call, or refuse.

        The old check compared already-recorded spend against the cap and never
        estimated the call it was about to authorise, so a single call could spend
        anything. The real bound was `daily_budget + unbounded`.
        """
        with self._lock:
            today = datetime.now(UTC).date()
            if today != self._spent_on:
                self._spent, self._spent_on = 0.0, today
            if self._spent + estimate > self._daily_budget:
                raise BudgetExceededError(
                    "call would exceed the daily LLM budget",
                    spent_usd=round(self._spent, 4),
                    estimate_usd=round(estimate, 4),
                    budget_usd=self._daily_budget,
                    route=route_name,
                )
            self._spent += estimate  # reserve now, reconcile after

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

        # ── input rail BEFORE the provider sees the prompt ──────────────────
        # This was inverted: the redactor ran only inside the interaction log, so the
        # card number reached the provider and the only clean copy was the one we
        # already controlled. Scrub outbound first.
        input_verdict = scan_pii(prompt, direction="input")
        if input_verdict.action == "block":
            raise ArgusError("prompt rejected by the PII rail", route=route_name)
        safe_prompt = input_verdict.sanitised
        self._log.write(
            GuardrailRecord(
                run_id=run_id,
                rail="pii",
                direction="input",
                action=input_verdict.action,
                reasons=list(input_verdict.reasons),
                latency_ms=0.0,
            )
        )

        # ── reserve budget against a worst-case estimate ────────────────────
        input_price, output_price = route.provider.price_per_million()
        estimate = (
            len(safe_prompt) / 4 * input_price + route.max_tokens * output_price
        ) / 1_000_000
        if estimate > route.max_cost_usd_per_call:
            raise BudgetExceededError(
                "call would exceed the per-call cost cap",
                estimate_usd=round(estimate, 5),
                cap_usd=route.max_cost_usd_per_call,
                route=route_name,
            )
        self._reserve(estimate, route_name)

        completion = route.provider.complete(
            safe_prompt,
            max_tokens=route.max_tokens,
            temperature=route.temperature,
            json_schema=json_schema,
        )
        # Reconcile the reservation against what was actually billed. A provider that
        # reports no usage is charged the estimate, never zero — otherwise a missing
        # `usageMetadata` field makes every call free and the cap never binds.
        actual = completion.cost_usd if completion.cost_usd > 0 else estimate
        with self._lock:
            self._spent += actual - estimate

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
                rendered_prompt=safe_prompt,
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
