"""fsa_sim.adversarial — fraudsters and attackers (ARCHITECTURE.md §15).

Everything here *transforms* the honest world produced by `fsa_sim.world`. Nothing
here draws a fresh population, for the reason set out in ADR 0004.

The asset this module creates is ground truth: we know exactly which claims are
fraudulent and exactly which receipts carry a payload. Catch rate, detection lag and
false-positive cost are therefore measurable — which is precisely what a real
production system cannot do.
"""

from fsa_sim.adversarial.fraud import inject_fraud
from fsa_sim.adversarial.injection import PAYLOADS, inject_payloads

__all__ = ["PAYLOADS", "inject_fraud", "inject_payloads"]
