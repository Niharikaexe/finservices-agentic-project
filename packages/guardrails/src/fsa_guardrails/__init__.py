"""fsa_guardrails — rails, taint tracking and PII redaction.

The six rails from ARCHITECTURE.md §12 land in M3 and run as a sidecar so their
latency cost is measurable per rail. What exists now is the taint type, because the
simulator starts producing attacker-controlled receipt text in the very first
milestone of work.
"""

from fsa_guardrails.rails import scan_injection, scan_pii, scan_tenant_leakage
from fsa_guardrails.untrusted import (
    GuardrailBlockedError,
    RailAction,
    RailVerdict,
    UntrustedText,
)

__all__ = [
    "GuardrailBlockedError",
    "RailAction",
    "RailVerdict",
    "UntrustedText",
    "scan_injection",
    "scan_pii",
    "scan_tenant_leakage",
]
