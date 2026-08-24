"""Strict interaction logging — the substrate every eval pipeline sits on.

The JD asks for evaluation and release gates covering answer and retrieval quality,
safety, cost, latency and drift. None of that is computable after the fact unless you
captured the interaction at the time. **This module is the capture.**

What gets written, per run:

  RetrievalRecord   query, principal role/tenant, how many documents the ACL permitted,
                    every chunk id returned with its score and citation, the `as_of`
                    date, latency
  LlmRecord         prompt template id AND the rendered prompt, model, provider,
                    parameters, output, token counts, cost, TTFT, total latency
  GuardrailRecord   rail, direction, action, reasons, latency
  ToolRecord        tool, argument digest, outcome, latency

Four properties make this an audit log rather than a debug log:

  **Append-only.** Records are never mutated. A verdict that changed is a new record.

  **Redacted at write time.** PII goes through the rail *before* it is persisted. You
  cannot log a card number into the eval store and promise to clean it later — the
  copy already exists, and in a PCI or GDPR context that copy is the incident.

  **Schema-versioned.** `SCHEMA_VERSION` is on every record. Eval pipelines break
  silently when the log shape drifts; a version field turns that into a loud failure.

  **Correlated.** Everything in one `run_id`, so a bad answer can be traced back
  through its guardrail verdicts to the exact chunks retrieved and the exact prompt.

Vendor neutrality (also from the JD): records carry `provider` and `model` as plain
fields, so a multi-provider gateway logs Azure OpenAI, Bedrock and a self-hosted vLLM
endpoint into one schema and cost comparison is a groupby, not a migration.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(UTC).isoformat()


def new_run_id() -> str:
    return f"run-{uuid.uuid4().hex[:16]}"


@dataclass(frozen=True, slots=True)
class RetrievalRecord:
    """One permission-aware retrieval.

    `allowed_document_count` is here because it is the number that proves filtering
    happened before ranking: if it is 5 and the corpus has 20 documents, the other 15
    never entered the similarity computation. An eval that only logs the returned
    chunks cannot distinguish that from post-retrieval discard.
    """

    run_id: str
    query: str
    principal_role: str
    tenant_id: str
    principal_department: str | None
    as_of: str
    allowed_document_count: int
    returned_chunk_ids: list[str]
    returned_citations: list[str]
    scores: list[float]
    k: int
    latency_ms: float
    kind: Literal["retrieval"] = "retrieval"
    schema_version: int = SCHEMA_VERSION
    at: str = field(default_factory=_now)


@dataclass(frozen=True, slots=True)
class LlmRecord:
    """One model call.

    Both `prompt_template_id` and `rendered_prompt` are stored. The template is what
    you changed between releases; the rendered prompt is what the model actually saw.
    Diffing eval runs without the template id means you cannot attribute a regression
    to a prompt change, and storing only the template means you cannot reproduce the
    call.
    """

    run_id: str
    provider: str
    model: str
    prompt_template_id: str
    rendered_prompt: str
    output: str
    parameters: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    ttft_ms: float | None
    latency_ms: float
    workflow: str
    finish_reason: str = "stop"
    structured_output_valid: bool | None = None
    repair_attempts: int = 0
    kind: Literal["llm"] = "llm"
    schema_version: int = SCHEMA_VERSION
    at: str = field(default_factory=_now)


@dataclass(frozen=True, slots=True)
class GuardrailRecord:
    run_id: str
    rail: str
    direction: Literal["input", "output", "tool"]
    action: Literal["allow", "strip", "block"]
    reasons: list[str]
    latency_ms: float
    kind: Literal["guardrail"] = "guardrail"
    schema_version: int = SCHEMA_VERSION
    at: str = field(default_factory=_now)


@dataclass(frozen=True, slots=True)
class ToolRecord:
    run_id: str
    tool: str
    argument_digest: str  # a hash, never the raw arguments
    outcome: Literal["ok", "error", "denied", "timeout"]
    latency_ms: float
    authz_decision: str | None = None
    kind: Literal["tool"] = "tool"
    schema_version: int = SCHEMA_VERSION
    at: str = field(default_factory=_now)


Record = RetrievalRecord | LlmRecord | GuardrailRecord | ToolRecord


class InteractionLog:
    """Append-only JSONL sink, redacting on the way in.

    JSONL rather than a database because the eval pipeline reads it as a batch and
    because an append-only file is trivially auditable — you can prove nothing was
    edited. In cloud this becomes Blob append blobs plus OTel spans to Phoenix; the
    record shape does not change, which is the point of having a shape.
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        redactor: Callable[[str], str] | None = None,
    ) -> None:
        """`redactor` is injected, not imported.

        The PII rail lives in `fsa_guardrails`, which sits *above* telemetry in the
        layering — import-linter fails the build if this module reaches for it. That
        constraint produced the better design anyway: the log does not care how text
        is scrubbed, only that it is, so a service with a different classification
        policy passes its own redactor and nothing here changes.

        Passing `None` disables redaction and is only correct for text you generated
        yourself. `fsa_gateway` wires the real rail in.
        """
        self._path = path
        self._redactor = redactor
        self._buffer: list[Record] = []
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def _scrub(self, text: str) -> str:
        return self._redactor(text) if self._redactor is not None else text

    def _redact_record(self, record: Record) -> Record:
        """Return a redacted copy. Redaction happens ONCE, here, before the record is
        either buffered or written.

        An earlier version of this scrubbed only the serialised payload, so the file
        on disk was clean while `log.records` still held the raw card number in
        memory — and the eval pipeline reads `.records`. Redacting the record itself
        removes the possibility: there is no unredacted copy to leak.
        """
        if self._redactor is None:
            return record
        payload: dict[str, Any] = asdict(record)
        for key in ("query", "rendered_prompt", "output"):
            value = payload.get(key)
            if isinstance(value, str):
                payload[key] = self._scrub(value)
        return type(record)(**payload)

    # ── writing ─────────────────────────────────────────────────────────────
    def write(self, record: Record) -> None:
        safe = self._redact_record(record)
        self._buffer.append(safe)
        if self._path is not None:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(safe), separators=(",", ":")) + "\n")

    @property
    def records(self) -> list[Record]:
        return list(self._buffer)

    def of_kind(self, kind: str) -> list[Record]:
        return [r for r in self._buffer if r.kind == kind]

    # ── convenience ─────────────────────────────────────────────────────────
    @contextmanager
    def timed(self) -> Iterator[dict[str, float]]:
        """`with log.timed() as t:` then read `t["ms"]` afterwards."""
        holder: dict[str, float] = {}
        started = time.perf_counter()
        try:
            yield holder
        finally:
            holder["ms"] = (time.perf_counter() - started) * 1000

    def summary(self) -> dict[str, Any]:
        """What the eval pipeline and the cost dashboard both read."""
        llm = [r for r in self._buffer if isinstance(r, LlmRecord)]
        retrieval = [r for r in self._buffer if isinstance(r, RetrievalRecord)]
        rails = [r for r in self._buffer if isinstance(r, GuardrailRecord)]
        return {
            "schema_version": SCHEMA_VERSION,
            "runs": len({r.run_id for r in self._buffer}),
            "retrievals": len(retrieval),
            "llm_calls": len(llm),
            "guardrail_evaluations": len(rails),
            "guardrail_trips": sum(1 for r in rails if r.action != "allow"),
            "total_cost_usd": round(sum(r.cost_usd for r in llm), 6),
            "total_tokens": sum(r.prompt_tokens + r.completion_tokens for r in llm),
            "retrieval_p95_ms": (
                round(sorted(r.latency_ms for r in retrieval)[int(len(retrieval) * 0.95)], 3)
                if retrieval
                else None
            ),
            "citations_emitted": sum(len(r.returned_citations) for r in retrieval),
            "uncited_retrievals": sum(1 for r in retrieval if not r.returned_citations),
        }
