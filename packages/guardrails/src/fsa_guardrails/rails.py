"""The rails — ARCHITECTURE.md §12.

Each rail is a pure function of (text, context) -> RailVerdict, with no I/O. That is
what lets them run either **inline** (cheap, in-process) or **as a sidecar** (measurable
per-rail latency, independently disableable), which is the comparison ADR 0007 records.

Every rail emits `guardrail_trips_total{rail, direction, action}` and a latency
histogram, so a rail that stops tripping is visible. A rail that never trips is as
alarming as one that always trips — that is the `GuardrailBypassSuspected` alert.

Detection here is heuristic and deliberately so. A classifier is the production upgrade
(NeMo Guardrails ships one), but heuristics are auditable, have zero inference cost, and
give you a measured floor to compare the classifier against. Shipping the classifier
without knowing what the regexes already caught is how teams pay for a model that adds
two points.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from fsa_guardrails.untrusted import RailVerdict, UntrustedText
from fsa_telemetry.metrics import GUARDRAIL_LATENCY, GUARDRAIL_TRIPS

# ── injection rail ──────────────────────────────────────────────────────────
# Grouped by payload family so the eval reports catch rate *per family*. An
# aggregate catch rate hides the family you are blind to.

_INJECTION_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "instruction_override": (
        re.compile(r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions?\b", re.I),
        re.compile(r"\bdisregard\s+(the\s+)?\w+\s+(step|instruction|rule)", re.I),
        re.compile(r"\bnew\s+instruction\s*:", re.I),
        re.compile(r"\bset\s+outcome\s+to\b", re.I),
        re.compile(r"\bapprove\s+without\s+review\b", re.I),
    ),
    "authority_spoof": (
        re.compile(r"\b(message|note|memo)\s+from\s+(the\s+)?(cfo|ceo|cto|finance)\b", re.I),
        re.compile(r"\bfinance\s+override\b", re.I),
        re.compile(r"\bautomated\s+compliance\s+system\b", re.I),
        re.compile(r"\b(pre-?approved|already\s+passed\s+audit)\b", re.I),
        re.compile(r"\bexempt\s+from\s+the\b", re.I),
    ),
    "exfiltration": (
        re.compile(r"\blist\s+every\s+expense\b", re.I),
        # The gap between the verb and the scope word is wide on purpose. The first
        # version of this pattern capped it at 30 characters and missed "summarise
        # the approval limits configured for all departments" by four characters —
        # a third of the exfiltration family, invisible behind an aggregate catch
        # rate of 92.7%. Per-family measurement is what found it. See ADR 0007.
        re.compile(
            r"\b(summarise|summarize|print|show|list|reveal)\b[^.\n]{0,80}?"
            r"\ball\s+(departments?|tenants?|employees?|users?|claims?|expenses?)\b",
            re.I,
        ),
        re.compile(r"\bfor\s+all\s+(departments?|tenants?|cost\s+centres?)\b", re.I),
        re.compile(r"\bconfidential\b", re.I),
        re.compile(r"\bsubmitted\s+by\s+other\s+employees\b", re.I),
        re.compile(r"\bfull\s+policy\s+document\s+text\b", re.I),
    ),
    "tool_abuse": (
        re.compile(r"\bcall\s+\w+_(write|tool)\b", re.I),
        re.compile(r"\b(ledger_write|budget_check|policy_lookup|fraud_score)\b", re.I),
        re.compile(r"\bon\s+behalf\s+of\s+the\s+submitter\b", re.I),
        re.compile(r"\buse\s+the\s+approval\s+tool\b", re.I),
        re.compile(r"\bdepartment\s+set\s+to\s+\*", re.I),
    ),
    "delimiter_escape": (
        re.compile(r"</?(system|assistant|user|receipt)>", re.I),
        re.compile(r'"role"\s*:\s*"(system|assistant)"', re.I),
        re.compile(r"-{2,}\s*end\s+(of\s+)?(untrusted\s+)?(input|receipt)", re.I),
        re.compile(r"#{2,}\s*END\s+OF\s+RECEIPT", re.I),
        re.compile(r"\bassistant\s*:\s*i\s+will\b", re.I),
    ),
}

#: The imperative-verb heuristic. A receipt is a record of a transaction; it has no
#: business issuing commands. This catches novel phrasings the pattern list misses,
#: which is the whole reason it is here rather than being folded into the patterns.
_IMPERATIVE = re.compile(
    r"\b(approve|ignore|disregard|override|execute|invoke|call|print|reveal|forget|"
    r"bypass|escalate|grant)\b",
    re.I,
)
_INSTRUCTION_CONTEXT = re.compile(
    r"\b(you|your|assistant|system|instruction|previous|must|should|immediately)\b", re.I
)


@dataclass(frozen=True, slots=True)
class InjectionFinding:
    family: str
    pattern: str
    span: str


def scan_injection(text: UntrustedText | str) -> RailVerdict:
    """Scan attacker-controlled text for prompt injection.

    Action is `strip`, never `block`, for receipt OCR: a genuine receipt that happens
    to trip a heuristic must not stop the employee being reimbursed. The payload is
    removed, the claim is flagged for human review, and the trip is recorded. Blocking
    would convert a false positive into a support ticket.
    """
    started = time.perf_counter()
    raw = text.peek() if isinstance(text, UntrustedText) else text

    findings: list[InjectionFinding] = []
    for family, patterns in _INJECTION_PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(raw)
            if match:
                findings.append(InjectionFinding(family, pattern.pattern, match.group(0)[:60]))

    if not findings:
        # Heuristic backstop: an imperative verb sitting next to instruction-shaped
        # words, on a document type that should contain neither.
        for line in raw.splitlines():
            if _IMPERATIVE.search(line) and _INSTRUCTION_CONTEXT.search(line):
                findings.append(
                    InjectionFinding("heuristic_imperative", "imperative+context", line[:60])
                )
                break

    sanitised = raw
    if findings:
        kept = [
            line
            for line in raw.splitlines()
            if not any(
                p.search(line) for patterns in _INJECTION_PATTERNS.values() for p in patterns
            )
            and not (_IMPERATIVE.search(line) and _INSTRUCTION_CONTEXT.search(line))
        ]
        sanitised = "\n".join(kept)

    action = "strip" if findings else "allow"
    GUARDRAIL_TRIPS.labels(rail="injection", direction="input", action=action).inc()
    GUARDRAIL_LATENCY.labels(rail="injection").observe(time.perf_counter() - started)

    return RailVerdict(
        rail="injection",
        action=action,  # type: ignore[arg-type]
        sanitised=sanitised,
        reasons=tuple(f"{f.family}:{f.span}" for f in findings),
    )


# ── PII rail ────────────────────────────────────────────────────────────────
# UAE and Indian identifiers, because that is where the data actually is. A rail
# tuned only for US SSNs is a rail that does nothing in Dubai.

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "card_number": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "emirates_id": re.compile(r"\b784[- ]?\d{4}[- ]?\d{7}[- ]?\d\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "aadhaar": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "trn": re.compile(r"\bTRN[:\s]*\d{15}\b", re.I),
    "email": re.compile(r"\b[\w.%-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "iban_like_account": re.compile(r"\bA/?C(?:\s*NO)?[:\s]*\d{8,18}\b", re.I),
}


def _luhn_ok(digits: str) -> bool:
    """Card numbers pass Luhn; a 16-digit invoice number usually does not.

    Without this check the card rail fires on every long numeric string on a receipt —
    order numbers, GSTINs, phone numbers — and a rail with a 90% false-positive rate
    gets switched off within a week, which is worse than not having it.
    """
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def scan_pii(text: str, *, direction: str = "output") -> RailVerdict:
    """Detect and redact identifiers. Runs on input *and* output.

    On input it stops a card number entering a prompt (and therefore a provider's
    logs). On output it stops one leaving in an answer. Both directions matter and
    teams routinely implement only one.
    """
    started = time.perf_counter()
    reasons: list[str] = []
    redacted = text

    for label, pattern in _PII_PATTERNS.items():
        for match in list(pattern.finditer(text)):
            value = match.group(0)
            if label == "card_number":
                digits = re.sub(r"\D", "", value)
                if not (13 <= len(digits) <= 19 and _luhn_ok(digits)):
                    continue
            reasons.append(f"{label}:{len(value)}chars")
            redacted = redacted.replace(value, f"[REDACTED:{label.upper()}]")

    action = "strip" if reasons else "allow"
    GUARDRAIL_TRIPS.labels(rail="pii", direction=direction, action=action).inc()
    GUARDRAIL_LATENCY.labels(rail="pii").observe(time.perf_counter() - started)
    return RailVerdict("pii", action, redacted, tuple(reasons))  # type: ignore[arg-type]


# ── cross-tenant leakage rail ───────────────────────────────────────────────


_ENTITY = re.compile(r"\b(t\d{2}-(?:u|exp|dept|v|doc)-[A-Za-z0-9-]+|t\d{2}-doc-[a-z0-9-]+)\b")


def scan_tenant_leakage(
    text: str, *, tenant_id: str, permitted_ids: set[str] | None = None
) -> RailVerdict:
    """The rail that catches *your own* bugs.

    Every entity identifier in an output is checked against the caller's tenant and,
    optionally, the exact set of objects they were authorised to see. If retrieval has
    a bug and slips another tenant's chunk through, this trips on the way out.

    Deliberately injecting a retrieval bug and verifying this rail catches it is a
    genuinely good game-day scenario, and it is the strongest argument for output rails
    existing at all: input rails defend against attackers, output rails defend against
    you.
    """
    started = time.perf_counter()
    reasons: list[str] = []
    redacted = text

    for match in _ENTITY.finditer(text):
        entity = match.group(0)
        entity_tenant = entity.split("-", 1)[0]
        if entity_tenant != tenant_id:
            reasons.append(f"cross_tenant:{entity_tenant}")
            redacted = redacted.replace(entity, "[REDACTED:CROSS_TENANT]")
        elif permitted_ids is not None and entity not in permitted_ids:
            reasons.append(f"unpermitted_entity:{entity_tenant}")
            redacted = redacted.replace(entity, "[REDACTED:UNPERMITTED]")

    # `block`, not `strip`. A cross-tenant identifier in an output means retrieval or
    # authorisation is broken; redacting and continuing would ship a plausible answer
    # built on data the caller had no right to. Fail the request.
    action = "block" if reasons else "allow"
    GUARDRAIL_TRIPS.labels(rail="tenant_leakage", direction="output", action=action).inc()
    GUARDRAIL_LATENCY.labels(rail="tenant_leakage").observe(time.perf_counter() - started)
    return RailVerdict("tenant_leakage", action, redacted, tuple(reasons))  # type: ignore[arg-type]
