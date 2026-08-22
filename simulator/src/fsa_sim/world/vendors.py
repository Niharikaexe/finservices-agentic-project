"""Vendor catalogue.

Vendors matter more than they look. Three of the seven fraud feature families in
ARCHITECTURE.md §11 are vendor-derived: vendor first-seen recency, vendor concentration
for an employee, and vendor-employee affinity. A world where everyone buys from a
uniform random vendor makes all three features useless — so the catalogue is built with
a **long tail**: a handful of vendors take most of the volume, and a fat tail of
one-off vendors provides the noise that a "ghost vendor" must hide in.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from fsa_sim.world.entities import Category, Tenant, Vendor

_VENDOR_STEMS: dict[Category, tuple[str, ...]] = {
    Category.MEALS: ("Cafe", "Kitchen", "Bistro", "Dhaba", "Canteen", "Eatery"),
    Category.CLIENT_ENTERTAINMENT: ("Grill", "Lounge", "Club", "Brasserie", "Terrace"),
    Category.TRAVEL_AIR: ("Airways", "Air", "Jet", "Aviation"),
    Category.TRAVEL_GROUND: ("Cabs", "Rides", "Rail", "Transit", "Motors"),
    Category.LODGING: ("Hotel", "Residency", "Suites", "Inn", "Stays"),
    Category.MILEAGE: ("Fuel", "Petro", "Energy"),
    Category.SOFTWARE: ("Labs", "Software", "Cloud", "Systems", "Tech"),
    Category.OFFICE_SUPPLIES: ("Stationers", "Depot", "Supplies", "Traders"),
    Category.TRAINING: ("Academy", "Institute", "Learning", "Skills"),
    Category.TELECOM: ("Telecom", "Mobile", "Networks", "Connect"),
}
_VENDOR_PREFIXES: tuple[str, ...] = (
    "Blue",
    "Metro",
    "Prime",
    "Urban",
    "Sunrise",
    "Apex",
    "Nova",
    "Vertex",
    "Anchor",
    "Copper",
    "Silver",
    "Kestrel",
    "Banyan",
    "Harbour",
    "Summit",
    "Orchid",
)


def build_vendors(
    tenant: Tenant,
    rng: np.random.Generator,
    *,
    world_start: date,
    per_category: int = 12,
) -> list[Vendor]:
    """One catalogue per tenant. `first_seen_on` is spread over the two years before
    the world starts, so "vendor first seen 3 days ago" is a genuinely rare, and
    therefore genuinely suspicious, signal."""
    vendors: list[Vendor] = []
    seq = 0
    for category, stems in _VENDOR_STEMS.items():
        for _ in range(per_category):
            seq += 1
            prefix = _VENDOR_PREFIXES[int(rng.integers(len(_VENDOR_PREFIXES)))]
            stem = stems[int(rng.integers(len(stems)))]
            vendors.append(
                Vendor(
                    vendor_id=f"{tenant.tenant_id}-v-{seq:04d}",
                    tenant_id=tenant.tenant_id,
                    name=f"{prefix} {stem}",
                    category=category,
                    first_seen_on=world_start - timedelta(days=int(rng.integers(30, 730))),
                )
            )
    return vendors


def popularity_weights(n: int, rng: np.random.Generator, *, alpha: float = 1.6) -> np.ndarray:
    """Zipf-ish weights over `n` vendors: a few dominate, a long tail barely appears.

    Returns a probability vector, shuffled so the popular vendors are not the first
    ones in the list (which would let a model learn vendor *index*, a classic synthetic
    -data leak).
    """
    ranks = np.arange(1, n + 1, dtype=float)
    weights = 1.0 / ranks**alpha
    rng.shuffle(weights)
    return weights / weights.sum()
