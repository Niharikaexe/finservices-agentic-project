"""Model providers behind one interface.

The JD's words: *"Keep the architecture modular and vendor-neutral, so we are never
locked to a single framework or model provider."* This file is that sentence in code.

A provider is a function from a request to a `Completion` plus its cost and token
accounting. Nothing above this layer knows which vendor answered — the copilot, the
triage graph and the eval harness all see the same shape, so switching Azure OpenAI to
Gemini to a self-hosted vLLM endpoint is a config line.

Secrets are read from the environment, never passed as arguments and never written to
a record. `fsa_common.Settings` holds them as `SecretStr`, so a stray f-string in a log
line prints `**********`.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

from fsa_common import ArgusError, get_logger

log = get_logger(__name__)


class ModelUnavailableError(ArgusError):
    """The configured model id was rejected by the provider.

    Distinct from a transient provider error because the remedy is different: this one
    never succeeds on retry, it needs a config change.
    """

    http_status = 502
    code = "model_unavailable"


class ProviderQuotaExhaustedError(ArgusError):
    """The account is out of credit or over a hard quota. Not a rate limit.

    Google returns 429 RESOURCE_EXHAUSTED for both "you are going too fast" and "your
    prepayment credits are depleted", and only the message distinguishes them. They
    could not be less alike operationally: the first clears in seconds and retrying is
    correct, the second never clears and retrying spends the caller's latency budget
    three times over to arrive at the same failure. Observed here — every request
    burning 9 seconds on backoff against an account with no credit left.
    """

    http_status = 402
    code = "provider_quota_exhausted"


@dataclass(frozen=True, slots=True)
class Completion:
    """One model response, provider-agnostic."""

    text: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    ttft_ms: float | None
    latency_ms: float
    finish_reason: str = "stop"
    raw: dict[str, Any] | None = None
    #: How many times the call was retried before it succeeded. Zero on the happy
    #: path. Worth carrying because a route that quietly retries twice on every call
    #: has three times the rate-limit footprint and three times the tail latency, and
    #: neither shows up in a success rate.
    retries: int = 0


class ModelProvider(Protocol):
    name: str
    model: str

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion: ...

    def price_per_million(self) -> tuple[float, float]:
        """(input, output) USD per million tokens. Used for cost accounting."""
        ...


# ── deterministic provider, for tests and CI ────────────────────────────────


class EchoProvider:
    """Returns a canned or rule-derived answer. No network, no key, no cost.

    Every test in this repo that exercises the agent path uses this, so the suite runs
    offline and deterministically. An eval suite that needs a live model is an eval
    suite that is skipped when the key rotates.
    """

    name = "echo"

    def __init__(
        self,
        model: str = "echo-1",
        responses: dict[str, str] | None = None,
        *,
        match_after: str | None = None,
    ) -> None:
        """`match_after` narrows needle matching to the text after that marker.

        Without it, needles are matched against the whole prompt — including the
        retrieved context. In a RAG prompt that is almost always wrong: every
        retrieval for this corpus returns the global policy, which contains the words
        "Client Entertainment", so a `client entertainment` needle matched no matter
        what the user actually asked. The stub then returned the same canned answer to
        every question from every user, which reads exactly like an authorisation
        failure and is nothing of the sort.
        """
        self.model = model
        self._responses = responses or {}
        self._match_after = match_after

    def price_per_million(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        del max_tokens, temperature, json_schema
        started = time.perf_counter()

        haystack = prompt
        if self._match_after is not None:
            _, marker, tail = prompt.partition(self._match_after)
            if marker:
                haystack = tail

        for needle, response in self._responses.items():
            if needle.lower() in haystack.lower():
                text = response
                break
        else:
            # A well-formed refusal, not a prose echo. The prose form parsed as
            # nothing, so an unmatched question surfaced as `unparseable_output` —
            # a model failure the model never had. Say plainly that this is a stub.
            text = json.dumps(
                {
                    "answer": (
                        "This is the offline stub provider. It has no canned answer "
                        "for that question — set GOOGLE_API_KEY in .env for real "
                        "answers."
                    ),
                    "citations": [{"document_id": "none", "rule_ref": "none", "quoted_span": ""}],
                    "applicable_limit_minor": None,
                    "confidence": 0.0,
                }
            )
        latency = (time.perf_counter() - started) * 1000
        return Completion(
            text=text,
            provider=self.name,
            model=self.model,
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=max(1, len(text) // 4),
            cost_usd=0.0,
            ttft_ms=latency,
            latency_ms=latency,
        )


# ── Gemini ──────────────────────────────────────────────────────────────────

#: Statuses worth trying again. 429 is a rate limit and 5xx is the provider having a
#: moment; both clear on their own. Everything else is a permanent condition where a
#: retry produces the same failure later, having spent quota to get there.
_RETRYABLE = frozenset({429, 500, 502, 503, 504})

#: Substrings that mark a 429 as billing rather than throttling. Matching on message
#: text is fragile and it is the only signal the API gives — there is no distinct
#: status code and no distinct `reason`. The failure mode of a wrong match is mild in
#: both directions: a missed match costs three retries, and a false match turns a
#: transient throttle into one clean refusal instead of a delayed one.
_BILLING_MARKERS = ("credits are depleted", "billing", "prepayment", "quota_exceeded_for_project")


def _billing_failure(response: Any) -> str | None:
    """Return the provider's message if this 429 is about money, not speed."""
    try:
        message = response.json().get("error", {}).get("message", "")
    except ValueError:
        return None
    lowered = message.lower()
    return message if any(marker in lowered for marker in _BILLING_MARKERS) else None


def _retry_after(response: Any) -> float | None:
    """Honour the provider's own backoff hint when it gives one.

    Guessing an interval when the server has told you the answer is how a fleet of
    clients synchronises into a thundering herd against a rate limit.
    """
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, min(float(raw), 30.0))
    except ValueError:
        return None  # the HTTP-date form; the exponential fallback covers it


#: Keys Gemini's `responseSchema` accepts. It takes an OpenAPI 3.0 subset, not full
#: JSON Schema, and rejects the request outright on an unknown key rather than
#: ignoring it — so unsupported keys are dropped rather than passed through.
_GEMINI_SCHEMA_KEYS = frozenset(
    {
        "type",
        "format",
        "description",
        "nullable",
        "enum",
        "items",
        "properties",
        "required",
        "minItems",
        "maxItems",
        "propertyOrdering",
    }
)


def _to_gemini_schema(schema: dict[str, Any], defs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Translate a standard JSON Schema into Gemini's `responseSchema` dialect.

    This lives on the provider, not on the caller, on purpose: the caller hands over
    `PolicyAnswer.model_json_schema()` — one generic artefact derived from the model
    that already governs the answer — and each provider bends it into its own dialect.
    Put the translation upstream instead and the pipeline grows a `if provider ==`,
    which is the vendor lock-in this package exists to avoid.

    Three concrete impedance mismatches are handled. Pydantic emits nested models as
    `$ref` into `$defs`, which Gemini does not resolve, so refs are inlined. Optional
    fields become `anyOf: [T, null]`, which Gemini spells `nullable: true`. And
    `min_length` on a list arrives as `minItems` but on a string as `minLength`, which
    Gemini has no equivalent for — dropping it is safe because `PolicyAnswer` is still
    validated by pydantic afterwards. The decoder constraint is an optimisation that
    removes most parse failures; it is never the thing enforcing the contract.
    """
    defs = defs if defs is not None else schema.get("$defs", {})

    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[-1]
        return _to_gemini_schema(defs.get(name, {}), defs)

    if "anyOf" in schema:
        variants = [v for v in schema["anyOf"] if v.get("type") != "null"]
        nullable = len(variants) != len(schema["anyOf"])
        # `description` and friends sit as SIBLINGS of `anyOf`, not inside the
        # variants. Recursing into the variant alone silently dropped them — and
        # since descriptions are what the model actually reads, the field whose
        # instruction mattered most (minor units) was the one that lost it.
        siblings = {k: v for k, v in schema.items() if k in _GEMINI_SCHEMA_KEYS}
        if len(variants) == 1:
            out = {**_to_gemini_schema(variants[0], defs), **siblings}
            if nullable:
                out["nullable"] = True
            return out
        # A genuine union. Gemini cannot express it, so fall back to an unconstrained
        # value and let pydantic reject a bad one downstream.
        return {**siblings, "type": "string", "nullable": nullable}

    out = {k: v for k, v in schema.items() if k in _GEMINI_SCHEMA_KEYS}
    if "properties" in schema:
        out["properties"] = {
            name: _to_gemini_schema(sub, defs) for name, sub in schema["properties"].items()
        }
        # Gemini emits object keys in whatever order it likes unless told. Pinning the
        # order makes the raw output stable, which makes an eval diff readable.
        out["propertyOrdering"] = list(schema["properties"])
    if "items" in schema:
        out["items"] = _to_gemini_schema(schema["items"], defs)
    return out


class GeminiProvider:
    """Google Gemini over the REST API.

    The key comes from `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) in the environment,
    loaded from a gitignored `.env` locally and from Key Vault in cloud. It is never a
    constructor argument, so it cannot end up in a stack trace or a config dump.

    **This is a reasoning model, and that changes the accounting.** Gemini 3.x spends
    "thinking" tokens before it emits an answer. They are reported separately as
    `thoughtsTokenCount`, they are *billed as output tokens*, and they are drawn from
    the same `maxOutputTokens` budget as the answer. A first live call here returned
    `candidatesTokenCount: 1` alongside `thoughtsTokenCount: 70` — costing on the
    former alone under-reports spend by the ratio between them, and a budget built on
    that number does not bind. Both consequences are handled below.

    Prices default high on purpose. Over-estimating the unit price makes the gateway's
    pre-call reservation refuse *sooner* than reality requires, which is the safe
    direction to be wrong in; under-estimating silently raises the real cap. Check
    them against Google's current pricing page before quoting a figure to anyone.
    """

    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-3.5-flash-lite",
        *,
        input_price: float = 0.10,
        output_price: float = 0.40,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self._input_price = input_price
        self._output_price = output_price
        self._timeout = timeout_seconds
        self._max_retries = max_retries

    def price_per_million(self) -> tuple[float, float]:
        return (self._input_price, self._output_price)

    def _api_key(self) -> str:
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GOOGLE_API_KEY not set. Put it in a gitignored .env — never in code, "
                "never in a commit, never pasted into a chat."
            )
        return key

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        import httpx

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )
        config: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if json_schema is not None:
            # Structured output: the model is constrained to the schema rather than
            # asked nicely for JSON. This is what removes a whole class of parse-retry
            # logic, and it is the provider feature worth checking before you pick one.
            config["responseMimeType"] = "application/json"
            config["responseSchema"] = _to_gemini_schema(json_schema)

        payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": config}
        headers = {"x-goog-api-key": self._api_key(), "content-type": "application/json"}

        started = time.perf_counter()
        attempt = 0
        while True:
            response = httpx.post(url, headers=headers, json=payload, timeout=self._timeout)
            if response.status_code == 429:
                billing = _billing_failure(response)
                if billing is not None:
                    raise ProviderQuotaExhaustedError(
                        "the provider account is out of credit or over a hard quota",
                        model=self.model,
                        detail=billing[:300],
                    )
            if response.status_code not in _RETRYABLE or attempt >= self._max_retries:
                break
            # Only 429 and 5xx get here. A 400 or a 404 is a permanent condition and
            # retrying it burns the rate-limit budget to arrive at the same answer
            # slower — which is how a broken config turns into an outage.
            delay = _retry_after(response) or min(2.0**attempt, 8.0)
            log.warning(
                "provider throttled; backing off",
                model=self.model,
                status=response.status_code,
                attempt=attempt + 1,
                delay_s=round(delay, 2),
            )
            time.sleep(delay)
            attempt += 1
        latency = (time.perf_counter() - started) * 1000

        if response.status_code == 404:
            # Google retires model ids and the REST error names the replacement. A bare
            # HTTPStatusError here reaches the pipeline as a generic `provider_error`,
            # so the service degrades every request to a refusal and the dashboard
            # shows a healthy-looking system answering nothing. Surface the cause.
            detail = response.json().get("error", {}).get("message", response.text[:200])
            raise ModelUnavailableError(
                "the configured Gemini model is not available to this key",
                model=self.model,
                detail=detail,
            )
        response.raise_for_status()
        body = response.json()

        candidates = body.get("candidates", [])
        text = ""
        finish = "stop"
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            finish = candidates[0].get("finishReason", "stop").lower()

        usage = body.get("usageMetadata", {})
        prompt_tokens = int(usage.get("promptTokenCount", 0))
        # Thinking tokens are output tokens for billing, so they are output tokens for
        # accounting. Folding them in here rather than at the call sites means the
        # gateway's budget, the cost metric and the interaction log are all correct by
        # construction, and none of them has to know this model reasons.
        completion_tokens = int(usage.get("candidatesTokenCount", 0)) + int(
            usage.get("thoughtsTokenCount", 0)
        )
        cost = (
            prompt_tokens * self._input_price + completion_tokens * self._output_price
        ) / 1_000_000

        if finish == "max_tokens" and not text.strip():
            # The whole output budget went on thinking and nothing was emitted. Left
            # alone this reads downstream as a malformed response, which sends you
            # debugging the JSON parser instead of raising `max_tokens`. Name it.
            finish = "truncated_in_thinking"

        return Completion(
            text=text,
            provider=self.name,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            ttft_ms=None,
            latency_ms=latency,
            finish_reason=finish,
            retries=attempt,
        )


class AzureOpenAIProvider:
    """Azure OpenAI. Same shape; wired in M2 deployment."""

    name = "azure_openai"

    def __init__(
        self, deployment: str, *, input_price: float = 0.15, output_price: float = 0.60
    ) -> None:
        self.model = deployment
        self._input_price = input_price
        self._output_price = output_price

    def price_per_million(self) -> tuple[float, float]:
        return (self._input_price, self._output_price)

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        raise NotImplementedError("wire in M2; EchoProvider and GeminiProvider cover today")
