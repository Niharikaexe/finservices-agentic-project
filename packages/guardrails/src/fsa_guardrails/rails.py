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

import hashlib
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


#: Families where removing the matched text cannot neutralise the attack. A fake
#: `END RECEIPT` fence is a multi-line structure: deleting the line that matched left
#: the fence intact, so stripping made the payload *more* effective while recording a
#: successful catch. These block instead.
_MUST_BLOCK = frozenset({"delimiter_escape"})


def scan_injection(text: UntrustedText | str) -> RailVerdict:
    r"""Scan attacker-controlled text for prompt injection.

    Action is `strip` for most families: a genuine receipt that trips a heuristic must
    not stop an employee being reimbursed, so the payload is removed and the claim is
    flagged for human review. Blocking would turn a false positive into a support
    ticket.

    Two corrections from the audit, both of which made the old `strip` dishonest:

      * **Removal is span-based, not line-based.** Patterns contain `\s+`, which
        matches a newline, so a payload wrapped across two lines matched as a finding
        while neither individual line matched — and the line filter then removed
        nothing. The verdict said `strip`; the payload was released intact.
      * **`delimiter_escape` blocks.** See `_MUST_BLOCK`.

    After stripping we re-scan the result and escalate to `block` if anything still
    trips, so "we removed it" is verified rather than assumed.
    """
    started = time.perf_counter()
    raw = text.peek() if isinstance(text, UntrustedText) else text

    findings: list[InjectionFinding] = []
    spans: list[tuple[int, int]] = []
    for family, patterns in _INJECTION_PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(raw):
                findings.append(InjectionFinding(family, pattern.pattern, match.group(0)[:60]))
                spans.append((match.start(), match.end()))

    if not findings:
        # Heuristic backstop: an imperative verb sitting next to instruction-shaped
        # words, on a document type that should contain neither. Collect every such
        # line, not just the first — `reasons` must account for every removal, or an
        # employee cannot be told what was deleted from their receipt.
        offset = 0
        for line in raw.splitlines(keepends=True):
            stripped = line.rstrip("\n")
            if _IMPERATIVE.search(stripped) and _INSTRUCTION_CONTEXT.search(stripped):
                findings.append(
                    InjectionFinding("heuristic_imperative", "imperative+context", stripped[:60])
                )
                spans.append((offset, offset + len(stripped)))
            offset += len(line)

    action = "allow"
    sanitised = raw
    if findings:
        families = {f.family for f in findings}
        if families & _MUST_BLOCK:
            action = "block"
            sanitised = ""
        else:
            merged: list[tuple[int, int]] = []
            for start, end in sorted(spans):
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            for start, end in reversed(merged):
                sanitised = sanitised[:start] + sanitised[end:]
            action = "strip"
            # Verify the removal actually worked rather than trusting it.
            if any(
                p.search(sanitised) for patterns in _INJECTION_PATTERNS.values() for p in patterns
            ):
                action, sanitised = "block", ""

    GUARDRAIL_TRIPS.labels(rail="injection", direction="input", action=action).inc()
    GUARDRAIL_LATENCY.labels(rail="injection").observe(time.perf_counter() - started)

    return RailVerdict(
        rail="injection",
        action=action,  # type: ignore[arg-type]
        sanitised=sanitised,
        # A digest, not the span. The heuristic span was up to 60 characters of raw
        # attacker-chosen OCR, and it landed verbatim in GuardrailRecord.reasons —
        # which is an audit log, not a place to copy untrusted text into.
        reasons=tuple(
            f"{f.family}:{hashlib.sha256(f.span.encode()).hexdigest()[:12]}" for f in findings
        ),
    )


# ── PII rail ────────────────────────────────────────────────────────────────
# UAE and Indian identifiers, because that is where the data actually is. A rail
# tuned only for US SSNs is a rail that does nothing in Dubai.

_SEP = r"[\s.\-/]"

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    # Emirates ID before card: it is a fixed 15-digit shape and would otherwise be
    # eaten by the generic digit-run scanner below.
    "emirates_id": re.compile(rf"\b784{_SEP}?\d{{4}}{_SEP}?\d{{7}}{_SEP}?\d\b"),
    "iban": re.compile(
        rf"\b[A-Z]{{2}}\d{{2}}(?:{_SEP}?[A-Z0-9]{{4}}){{2,7}}{_SEP}?[A-Z0-9]{{0,4}}\b", re.I
    ),
    "pan": re.compile(rf"\b[A-Z]{{5}}{_SEP}?\d{{4}}{_SEP}?[A-Z]\b", re.I),
    "aadhaar": re.compile(rf"\b\d{{4}}{_SEP}?\d{{4}}{_SEP}?\d{{4}}\b"),
    "trn": re.compile(rf"\bTRN[:\s]*(?:\d{_SEP}?){{15,20}}", re.I),
    "email": re.compile(r"[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(rf"\+\d{{1,3}}{_SEP}?(?:\d{_SEP}?){{7,12}}\d"),
    "account_number": re.compile(
        rf"\b(?:A/?C|ACCOUNT)(?:\s*(?:NO|NUMBER))?[:\s]*(?:\d{_SEP}?){{8,18}}", re.I
    ),
}

#: Any run of digits and separators long enough to hide a card. Scanned separately
#: from the table above with a sliding Luhn window — see `_card_spans`.
_DIGIT_RUN = re.compile(r"[\d\s.\-]{13,}")

#: Above this, refuse rather than scan. The replace loop used to be quadratic in
#: input length: a 400KB OCR blob took 4.5s and a 5MB one about eleven minutes of
#: single-threaded CPU, in the request path, on attacker-controlled input.
MAX_SCAN_CHARS = 256 * 1024


def _luhn_ok(digits: str) -> bool:
    """Card numbers pass Luhn; a 16-digit invoice number usually does not.

    Without this check the card rail fires on every long numeric string on a receipt
    — order numbers, GSTINs, phone numbers — and a rail with a 90% false-positive
    rate gets switched off within a week, which is worse than not having it.
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


def _card_spans(text: str) -> list[tuple[int, int]]:
    """Find card numbers by sliding a Luhn window over each digit run.

    The previous implementation used one greedy regex and gave up on the first Luhn
    failure. Because `[ -]?` swallowed the separator, a valid card followed by any
    short numeric token — `CARD 4539148803436467 99`, the ordinary shape of a receipt
    line — matched as a 19-digit blob, failed Luhn, and was skipped entirely. The
    inner 16 digits were never re-examined and the card was released unredacted.
    """
    spans: list[tuple[int, int]] = []
    for run in _DIGIT_RUN.finditer(text):
        chunk = run.group(0)
        # map each digit back to its offset in `text`
        offsets = [run.start() + i for i, ch in enumerate(chunk) if ch.isdigit()]
        digits = "".join(ch for ch in chunk if ch.isdigit())
        i = 0
        while i < len(digits):
            for length in range(19, 12, -1):  # longest match wins
                if i + length > len(digits):
                    continue
                if _luhn_ok(digits[i : i + length]):
                    spans.append((offsets[i], offsets[i + length - 1] + 1))
                    i += length
                    break
            else:
                i += 1
    return spans


def scan_pii(text: str, *, direction: str = "output") -> RailVerdict:
    """Detect and redact identifiers. Runs on input *and* output.

    On input it stops a card number entering a prompt (and therefore a provider's
    logs). On output it stops one leaving in an answer. Both directions matter and
    teams routinely implement only one.

    Redaction is span-based and applied right-to-left in a single pass. The previous
    version called `str.replace` per match, which was quadratic and — worse — let an
    earlier label overwrite a region so a later pattern's `.replace` silently no-opped
    while still appending a reason, producing verdicts that claimed a catch they had
    not made.
    """
    started = time.perf_counter()

    if len(text) > MAX_SCAN_CHARS:
        GUARDRAIL_TRIPS.labels(rail="pii", direction=direction, action="block").inc()
        GUARDRAIL_LATENCY.labels(rail="pii").observe(time.perf_counter() - started)
        return RailVerdict("pii", "block", "", (f"oversized_input:{len(text)}",))

    spans: list[tuple[int, int, str]] = [(a, b, "card_number") for a, b in _card_spans(text)]
    for label, pattern in _PII_PATTERNS.items():
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end(), label))

    # Longest-first, then drop anything overlapping an already-accepted span, so one
    # identifier is never half-redacted and mislabelled by a competing pattern.
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    accepted: list[tuple[int, int, str]] = []
    cursor = -1
    for start, end, label in spans:
        if start >= cursor:
            accepted.append((start, end, label))
            cursor = end

    redacted = text
    for start, end, label in reversed(accepted):
        redacted = f"{redacted[:start]}[REDACTED:{label.upper()}]{redacted[end:]}"

    reasons = tuple(f"{label}:{end - start}chars" for start, end, label in accepted)
    action = "strip" if reasons else "allow"
    GUARDRAIL_TRIPS.labels(rail="pii", direction=direction, action=action).inc()
    GUARDRAIL_LATENCY.labels(rail="pii").observe(time.perf_counter() - started)
    return RailVerdict("pii", action, redacted, reasons)  # type: ignore[arg-type]


# ── cross-tenant leakage rail ───────────────────────────────────────────────


#: `t\d+`, not `t\d{2}` — the two-digit form silently stopped matching at the
#: hundredth tenant, so the rail would have quietly switched itself off as the
#: platform grew. The entity-type class is open for the same reason.
_ENTITY = re.compile(r"\b(t\d+-[a-z]+-[A-Za-z0-9-]+)\b")


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
