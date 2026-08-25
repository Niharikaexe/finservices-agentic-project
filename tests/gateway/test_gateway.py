"""Model gateway regression tests — each one is a budget or redaction hole an audit found."""

from __future__ import annotations

import threading

import pytest

from fsa_gateway import BudgetExceededError, EchoProvider, ModelGateway, Route
from fsa_telemetry import new_run_id


class ExpensiveProvider(EchoProvider):
    name = "expensive"

    def price_per_million(self) -> tuple[float, float]:
        return (1_000_000.0, 1_000_000.0)


def _gateway(**kwargs) -> ModelGateway:  # type: ignore[no-untyped-def]
    return ModelGateway(
        {"triage": Route("triage", EchoProvider(), max_cost_usd_per_call=1.0)}, **kwargs
    )


def test_prompt_pii_is_scrubbed_before_the_provider_sees_it() -> None:
    """The control was inverted: only the local log was redacted, so the card number
    reached the provider and the only clean copy was the one we already controlled."""
    seen: list[str] = []

    class Recording(EchoProvider):
        def complete(self, prompt, **kwargs):  # type: ignore[no-untyped-def]
            seen.append(prompt)
            return super().complete(prompt, **kwargs)

    gw = ModelGateway({"triage": Route("triage", Recording())})
    gw.complete(
        "triage",
        "Reimburse card 4539 1488 0343 6467 for Rajesh",
        run_id=new_run_id(),
        prompt_template_id="t.v1",
    )
    assert "4539" not in seen[0], "card number reached the provider"
    assert "[REDACTED:CARD_NUMBER]" in seen[0]


def test_a_single_call_cannot_exceed_the_budget() -> None:
    """The check compared already-recorded spend against the cap and never estimated
    the call it was authorising, so one call could spend anything."""
    gw = ModelGateway(
        {"triage": Route("triage", ExpensiveProvider(), max_cost_usd_per_call=10_000.0)},
        daily_budget_usd=0.01,
    )
    with pytest.raises(BudgetExceededError):
        gw.complete("triage", "x" * 4000, run_id=new_run_id(), prompt_template_id="t.v1")
    assert gw.spent_usd == 0.0


def test_per_call_cap_is_enforced_before_the_call_not_logged_after() -> None:
    gw = ModelGateway(
        {"triage": Route("triage", ExpensiveProvider(), max_cost_usd_per_call=0.001)},
        daily_budget_usd=100.0,
    )
    with pytest.raises(BudgetExceededError):
        gw.complete("triage", "x" * 4000, run_id=new_run_id(), prompt_template_id="t.v1")


def test_concurrent_calls_cannot_race_past_the_budget() -> None:
    """50 concurrent calls against the cap previously spent 10x it with zero denials."""
    gw = ModelGateway(
        {"triage": Route("triage", ExpensiveProvider(), max_cost_usd_per_call=1.0)},
        daily_budget_usd=5.0,
    )
    denied = []

    def call() -> None:
        try:
            gw.complete("triage", "x" * 4000, run_id=new_run_id(), prompt_template_id="t.v1")
        except BudgetExceededError:
            denied.append(1)

    threads = [threading.Thread(target=call) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert gw.spent_usd <= 5.0, f"budget overrun: {gw.spent_usd}"
    assert denied, "no calls were denied despite exceeding the cap"


def test_every_call_is_logged() -> None:
    gw = _gateway()
    gw.complete("triage", "hello", run_id=new_run_id(), prompt_template_id="t.v1")
    kinds = {r.kind for r in gw.interaction_log.records}
    assert "llm" in kinds and "guardrail" in kinds


class FailingProvider(EchoProvider):
    """A provider that always rejects — a rate limit, an outage, a bad key."""

    name = "failing"

    #: Priced so the per-call cap lets the call through. Priced any higher and the
    #: cap rejects it before the provider is reached, and the test passes without
    #: ever exercising the release path it exists to check.
    def price_per_million(self) -> tuple[float, float]:
        return (1.0, 1.0)

    def complete(self, prompt: str, **kwargs: object):  # type: ignore[no-untyped-def, override]
        raise RuntimeError("429 Too Many Requests")


def test_a_failed_call_does_not_keep_its_reservation() -> None:
    """Budget is reserved before the call and reconciled after. When the call raised,
    the reconciliation never ran and the reservation was charged forever.

    Live numbers that exposed it: 24 requests against a rate-limited free tier
    reported $0.037 spent against $0.0057 actually billed. Sustained, a provider
    outage exhausts the daily cap without producing a single answer — the cost control
    becomes the outage."""
    gateway = ModelGateway(
        {"qa": Route("qa", FailingProvider(), max_cost_usd_per_call=1.0)},
        daily_budget_usd=10.0,
    )
    for _ in range(5):
        with pytest.raises(RuntimeError):
            gateway.complete("qa", "a prompt", run_id=new_run_id(), prompt_template_id="t")

    assert gateway.spent_usd == 0.0, "nothing was billed, so nothing may be charged"


def test_the_budget_still_binds_after_failures() -> None:
    """The release must not be a hole in the cap: a working call afterwards is still
    charged, and the cap still refuses."""
    gateway = ModelGateway(
        {
            "bad": Route("bad", FailingProvider(), max_cost_usd_per_call=1.0),
            # A per-call cap high enough to be irrelevant, so it is the DAILY cap
            # under test here and not the per-call one.
            "good": Route("good", ExpensiveProvider(), max_cost_usd_per_call=100_000.0),
        },
        daily_budget_usd=0.5,
    )
    with pytest.raises(RuntimeError):
        gateway.complete("bad", "x" * 40, run_id=new_run_id(), prompt_template_id="t")
    assert gateway.spent_usd == 0.0

    # ExpensiveProvider charges $1 per token, so one call cannot fit a $0.50 day.
    with pytest.raises(BudgetExceededError):
        gateway.complete("good", "x" * 40, run_id=new_run_id(), prompt_template_id="t")


def test_echo_matches_the_question_not_the_retrieved_context() -> None:
    """The stub returned the same canned answer to every question from every user.

    Needles were matched against the whole prompt, and every retrieval in this corpus
    returns the global policy containing the words "Client Entertainment" — so a
    `client entertainment` needle matched whatever was asked. Two users comparing
    answers then saw identical output and read it as an authorisation leak. It was the
    stub, but a demo that cannot tell those apart is worse than no demo.
    """
    prompt = (
        "--- BEGIN POLICY EXCERPTS ---\n"
        "[T&E-4.1] Client Entertainment: claims must not exceed Rs 5,000.\n"
        "--- BEGIN EMPLOYEE QUESTION ---\n"
        "What is the lodging cap?\n"
    )
    responses = {"client entertainment": "WRONG", "lodging": "RIGHT"}

    scoped = EchoProvider(responses=responses, match_after="BEGIN EMPLOYEE QUESTION")
    assert scoped.complete(prompt).text == "RIGHT"

    # Unscoped is the old behaviour, kept because it is right for non-RAG callers.
    assert EchoProvider(responses=responses).complete(prompt).text == "WRONG"


def test_echo_with_no_match_returns_a_parseable_refusal() -> None:
    """The old fallback echoed prose, which parsed as nothing — so an unmatched
    question surfaced as `unparseable_output`, a model failure the model never had."""
    import json

    text = EchoProvider(responses={"nothing": "x"}).complete("an unrelated question").text
    parsed = json.loads(text)
    assert parsed["confidence"] == 0.0
    assert "stub" in parsed["answer"].lower()
    assert parsed["citations"], "the citations contract holds even for the stub"
