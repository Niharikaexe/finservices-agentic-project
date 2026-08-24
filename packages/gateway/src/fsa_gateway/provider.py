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

import os
import time
from dataclasses import dataclass
from typing import Any, Protocol


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

    def __init__(self, model: str = "echo-1", responses: dict[str, str] | None = None) -> None:
        self.model = model
        self._responses = responses or {}

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
        for needle, response in self._responses.items():
            if needle.lower() in prompt.lower():
                text = response
                break
        else:
            text = "ECHO: " + prompt[-200:]
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


class GeminiProvider:
    """Google Gemini over the REST API.

    The key comes from `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) in the environment,
    loaded from a gitignored `.env` locally and from Key Vault in cloud. It is never a
    constructor argument, so it cannot end up in a stack trace or a config dump.
    """

    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        *,
        input_price: float = 0.10,
        output_price: float = 0.40,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.model = model
        self._input_price = input_price
        self._output_price = output_price
        self._timeout = timeout_seconds

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
            config["responseSchema"] = json_schema

        started = time.perf_counter()
        response = httpx.post(
            url,
            headers={"x-goog-api-key": self._api_key(), "content-type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": config},
            timeout=self._timeout,
        )
        latency = (time.perf_counter() - started) * 1000
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
        completion_tokens = int(usage.get("candidatesTokenCount", 0))
        cost = (
            prompt_tokens * self._input_price + completion_tokens * self._output_price
        ) / 1_000_000

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
