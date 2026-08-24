"""Interaction log regression tests."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from fsa_guardrails import scan_pii
from fsa_telemetry import GuardrailRecord, InteractionLog, LlmRecord, RetrievalRecord


def _redactor(text: str) -> str:
    return scan_pii(text, direction="output").sanitised


CARD = "4539 1488 0343 6467"


def _llm(**overrides: object) -> LlmRecord:
    base: dict[str, object] = {
        "run_id": "r1",
        "provider": "echo",
        "model": "e1",
        "prompt_template_id": "t.v1",
        "rendered_prompt": "clean",
        "output": "clean",
        "parameters": {},
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "cost_usd": 0.0,
        "ttft_ms": 1.0,
        "latency_ms": 1.0,
        "workflow": "w",
    }
    base.update(overrides)
    return LlmRecord(**base)  # type: ignore[arg-type]


def test_every_string_field_is_scrubbed_not_just_three() -> None:
    """`returned_citations`, `reasons`, `parameters` and friends all carried PII to
    disk while the docstring claimed everything was redacted."""
    log = InteractionLog(redactor=_redactor)
    log.write(
        RetrievalRecord(
            run_id="r1",
            query="q",
            principal_role="employee",
            tenant_id="t01",
            principal_department="d1",
            as_of="2026-03-15",
            allowed_document_count=1,
            returned_chunk_ids=[f"chunk {CARD}"],
            returned_citations=[f"cite {CARD}"],
            scores=[0.9],
            k=5,
            latency_ms=1.0,
        )
    )
    log.write(
        GuardrailRecord(
            run_id="r1",
            rail="injection",
            direction="input",
            action="strip",
            reasons=[f"span {CARD}"],
            latency_ms=1.0,
        )
    )
    log.write(_llm(parameters={"system": f"user card {CARD}"}))
    blob = json.dumps([r.__dict__ if hasattr(r, "__dict__") else str(r) for r in log.records])
    assert "4539" not in blob, "PII survived in a non-headline field"


def test_records_are_deep_copied_so_they_cannot_be_mutated_after_write() -> None:
    """`reasons` is a mutable list on a frozen dataclass — storing the caller's object
    by reference meant "append-only" was not true."""
    reasons = ["ok"]
    log = InteractionLog(redactor=None)
    log.write(
        GuardrailRecord(
            run_id="r1",
            rail="pii",
            direction="output",
            action="allow",
            reasons=reasons,
            latency_ms=1.0,
        )
    )
    reasons.append("TAMPERED")
    assert log.records[0].reasons == ["ok"]  # type: ignore[union-attr]


def test_unserialisable_parameters_do_not_desync_the_buffer_and_the_file(tmp_path: Path) -> None:
    """A Decimal in `parameters` used to raise out of write() after the buffer had
    been appended — and after the provider had been paid."""
    path = tmp_path / "log.jsonl"
    log = InteractionLog(path, redactor=None)
    log.write(_llm(parameters={"bad": Decimal("1.5")}))
    lines = path.read_text().splitlines()
    assert len(lines) == len(log.records) == 1
    assert json.loads(lines[0])["kind"] == "serialisation_error"


def test_non_finite_cost_does_not_poison_the_log(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    log = InteractionLog(path, redactor=None)
    log.write(_llm(cost_usd=float("nan")))
    assert "NaN" not in path.read_text()
