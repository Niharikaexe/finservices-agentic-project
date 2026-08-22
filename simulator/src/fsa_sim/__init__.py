"""fsa_sim — the synthetic world.

Argus has no live customers, so the simulator *is* the production environment. It is a
first-class component, not a test fixture (ARCHITECTURE.md §15).

Its defining property: **we hold ground truth.** We know exactly which claims are
fraudulent and exactly which receipts carry an injection payload. That is what makes
catch rate and detection lag measurable — and it is precisely what a real production
system never has.

Layout
    world/        synthetic tenants, org trees, vendors, spend behaviour
    adversarial/  fraud typologies, injection payloads, ACL probes   (M3/M4)
    traffic/      persona-driven Locust load with real OAuth tokens  (M3)
    chaos/        dependency failure injection                       (M6)
"""

__all__ = ["world"]

from fsa_sim import world
