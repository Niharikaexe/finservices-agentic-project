"""Gemini provider tests.

These are offline: the HTTP call is stubbed, because a suite that needs a live key is
a suite that gets skipped the day the key rotates. What is being tested is our
handling of the *shapes* the API returns — every response body below was copied from a
real call, so the stub cannot drift into fiction.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from copilot_service.schemas import PolicyAnswer
from fsa_gateway.provider import GeminiProvider, ModelUnavailableError, _to_gemini_schema


def _stub(monkeypatch: pytest.MonkeyPatch, body: dict[str, Any], status: int = 200) -> list[dict]:
    """Replace the POST with one that records its payload and returns `body`."""
    sent: list[dict] = []

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        sent.append(kwargs["json"])
        return httpx.Response(status, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-a-real-one")
    return sent


def _ok_body(text: str, *, thoughts: int = 0, finish: str = "STOP") -> dict[str, Any]:
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": finish}],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 20,
            "thoughtsTokenCount": thoughts,
        },
    }


# ── thinking tokens ─────────────────────────────────────────────────────────


def test_thinking_tokens_are_billed_as_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bug this exists to prevent: a live call returned `candidatesTokenCount: 1`
    beside `thoughtsTokenCount: 70`. Costing on the former under-reported that call by
    70x, and a daily budget built on it does not bind."""
    _stub(monkeypatch, _ok_body("{}", thoughts=700))
    completion = GeminiProvider(input_price=1.0, output_price=2.0).complete("hi")

    assert completion.completion_tokens == 720, "thinking tokens must be counted as output"
    # 100 in @ $1/M + 720 out @ $2/M
    assert completion.cost_usd == pytest.approx((100 * 1.0 + 720 * 2.0) / 1_000_000)


def test_a_non_reasoning_response_is_unaffected(monkeypatch: pytest.MonkeyPatch) -> None:
    """No `thoughtsTokenCount` must behave exactly as before, not charge a default."""
    body = _ok_body("{}")
    del body["usageMetadata"]["thoughtsTokenCount"]
    _stub(monkeypatch, body)
    assert GeminiProvider().complete("hi").completion_tokens == 20


def test_budget_sees_the_thinking_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end: the gateway's reconciliation must charge the thinking-inclusive
    figure, or the cap is a suggestion."""
    from fsa_gateway import ModelGateway, Route
    from fsa_telemetry import new_run_id

    _stub(monkeypatch, _ok_body("{}", thoughts=700))
    provider = GeminiProvider(input_price=1.0, output_price=2.0)
    gateway = ModelGateway(
        {"qa": Route("qa", provider, max_cost_usd_per_call=1.0)}, daily_budget_usd=1.0
    )
    gateway.complete("qa", "hi", run_id=new_run_id(), prompt_template_id="t")

    assert gateway.spent_usd == pytest.approx((100 + 720 * 2.0) / 1_000_000, abs=1e-9)


# ── truncation ──────────────────────────────────────────────────────────────


def test_budget_exhausted_by_thinking_is_named_not_guessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty body after MAX_TOKENS is a capacity problem, and the fix is to raise
    the token budget. Reported as a parse failure it sends you to the JSON parser."""
    _stub(monkeypatch, _ok_body("", thoughts=2048, finish="MAX_TOKENS"))
    assert GeminiProvider().complete("hi").finish_reason == "truncated_in_thinking"


def test_truncation_with_partial_text_stays_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a *wholly* empty body is the thinking case; partial output is ordinary
    truncation and must not be relabelled."""
    _stub(monkeypatch, _ok_body('{"answer": "par', thoughts=10, finish="MAX_TOKENS"))
    assert GeminiProvider().complete("hi").finish_reason == "max_tokens"


# ── a retired model id ──────────────────────────────────────────────────────


def test_a_retired_model_raises_something_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real 404 body, verbatim. Google retires model ids; a bare HTTPStatusError here
    degrades every request to a refusal and the dashboard still looks healthy."""
    _stub(
        monkeypatch,
        {
            "error": {
                "code": 404,
                "message": (
                    "This model models/gemini-2.0-flash is no longer available. Please "
                    "update your code to use models/gemini-3.6-flash"
                ),
                "status": "NOT_FOUND",
            }
        },
        status=404,
    )
    with pytest.raises(ModelUnavailableError) as caught:
        GeminiProvider(model="gemini-2.0-flash").complete("hi")
    assert "gemini-3.6-flash" in str(caught.value), "the replacement id must survive to the log"


def test_the_key_is_never_an_argument_and_never_in_the_error() -> None:
    """A missing key must not be papered over, and the message must not leak one."""
    import os

    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        os.environ.pop(name, None)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY not set"):
        GeminiProvider()._api_key()


# ── schema translation ──────────────────────────────────────────────────────


def test_the_answer_schema_survives_translation() -> None:
    """`PolicyAnswer` is the contract. If translation drops the citation constraint,
    the decoder stops enforcing the one rule the whole product rests on."""
    schema = _to_gemini_schema(PolicyAnswer.model_json_schema())

    assert schema["type"] == "object"
    assert set(schema["required"]) >= {"answer", "citations", "confidence"}
    citations = schema["properties"]["citations"]
    assert citations["minItems"] == 1, "no uncited verdicts — this is the architectural rule"
    # The nested Citation model arrives as a $ref; Gemini does not resolve those.
    assert citations["items"]["properties"]["rule_ref"]["type"] == "string"


def test_no_ref_or_defs_reaches_the_wire() -> None:
    """Gemini rejects the whole request on an unknown key rather than ignoring it, so
    one stray `$ref` fails every call rather than degrading one field."""

    def walk(node: object) -> None:
        if isinstance(node, dict):
            assert "$ref" not in node and "$defs" not in node, node
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(_to_gemini_schema(PolicyAnswer.model_json_schema()))


def test_optional_becomes_nullable() -> None:
    """`int | None` is `anyOf: [integer, null]` in JSON Schema and `nullable` here."""
    field = _to_gemini_schema(PolicyAnswer.model_json_schema())["properties"][
        "applicable_limit_minor"
    ]
    assert field["type"] == "integer"
    assert field["nullable"] is True


def test_the_schema_is_actually_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Translation is worthless if the request never carries it."""
    sent = _stub(monkeypatch, _ok_body("{}"))
    GeminiProvider().complete("hi", json_schema=PolicyAnswer.model_json_schema())

    config = sent[0]["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseSchema"]["properties"]["citations"]["minItems"] == 1


def test_no_schema_means_no_constraint(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _stub(monkeypatch, _ok_body("{}"))
    GeminiProvider().complete("hi")
    assert "responseSchema" not in sent[0]["generationConfig"]


# ── retry on transient failures ─────────────────────────────────────────────


def _sequence(
    monkeypatch: pytest.MonkeyPatch, responses: list[tuple[int, dict[str, Any]]]
) -> dict[str, list]:
    """Serve `responses` in order, recording sleeps instead of taking them."""
    calls: dict[str, list] = {"posts": [], "sleeps": []}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        status, body = responses[min(len(calls["posts"]), len(responses) - 1)]
        calls["posts"].append(kwargs["json"])
        headers = body.pop("__headers__", {})
        return httpx.Response(
            status, json=body, headers=headers, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("fsa_gateway.provider.time.sleep", lambda s: calls["sleeps"].append(s))
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-a-real-one")
    return calls


_THROTTLED = (429, {"error": {"code": 429, "message": "Too Many Requests"}})


def test_a_rate_limit_is_retried_and_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Observed live on a free-tier key: the traffic generator, running strictly
    serially, still tripped 429 and every affected request degraded to a refusal. A
    rate limit clears on its own, so refusing the user is the wrong response to it."""
    calls = _sequence(monkeypatch, [_THROTTLED, _THROTTLED, (200, _ok_body('{"ok": 1}'))])
    completion = GeminiProvider().complete("hi")

    assert completion.text == '{"ok": 1}'
    assert completion.retries == 2, "the retry count must survive for the SLI"
    assert calls["sleeps"] == [1.0, 2.0], "exponential, not a busy loop"


def test_retries_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unbounded retry against a sustained rate limit is a self-inflicted outage: the
    request never returns and the caller's timeout budget is spent waiting."""
    calls = _sequence(monkeypatch, [_THROTTLED])
    with pytest.raises(httpx.HTTPStatusError):
        GeminiProvider(max_retries=2).complete("hi")
    assert len(calls["posts"]) == 3, "the original call plus exactly two retries"


def test_the_provider_backoff_hint_wins_over_our_guess(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the server says how long to wait, guessing is how a fleet of clients
    synchronises into a thundering herd."""
    throttled = (429, {"error": {}, "__headers__": {"retry-after": "7"}})
    calls = _sequence(monkeypatch, [throttled, (200, _ok_body("{}"))])
    GeminiProvider().complete("hi")
    assert calls["sleeps"] == [7.0]


def test_a_permanent_failure_is_never_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A retired model id returns the same 404 forever. Retrying it spends rate-limit
    budget to reach the identical answer more slowly, turning a config error into a
    latency problem that hides the config error."""
    calls = _sequence(
        monkeypatch, [(404, {"error": {"code": 404, "message": "no longer available"}})]
    )
    with pytest.raises(ModelUnavailableError):
        GeminiProvider().complete("hi")
    assert len(calls["posts"]) == 1
    assert calls["sleeps"] == []


def test_a_clean_call_reports_no_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _sequence(monkeypatch, [(200, _ok_body("{}"))])
    assert GeminiProvider().complete("hi").retries == 0
    assert calls["sleeps"] == []


def test_field_descriptions_reach_the_model() -> None:
    """Descriptions on `PolicyAnswer` are not documentation — the provider renders
    them into `responseSchema`, so they are the instruction the model reads.

    This is load-bearing. Without the minor-units description, flash-lite returned
    1500000 for Rs 15,000 and 5000 for Rs 5,000 in consecutive calls: inferring the
    unit from the field name and getting it right by luck. Drop these and the failure
    is silent, intermittent, and off by 100x on money.
    """
    schema = _to_gemini_schema(PolicyAnswer.model_json_schema())
    limit = schema["properties"]["applicable_limit_minor"]
    assert "MINOR" in limit["description"]
    assert "500000" in limit["description"], "the worked example is what fixed it"
    assert schema["properties"]["citations"]["items"]["properties"]["rule_ref"]["description"]
