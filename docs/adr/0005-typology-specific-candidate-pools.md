# 0005. Fraud typologies draw from typology-specific candidate pools

- **Status:** accepted
- **Date:** 2026-08-23

## Context

`inject_fraud` originally picked its targets uniformly at random from the honest
population, drew a typology from the configured mix, and applied it to whatever claim
came up. That is the obvious implementation and it produced a broken dataset.

The first generated world showed it immediately. Median claim amounts, honest vs
fraudulent, by category:

| Category | Honest | Fraud | Ratio |
|---|---|---|---|
| `travel_ground` | ₹377 | ₹26,438 | **70x** |
| `meals` | ₹665 | ₹21,000 | **32x** |
| `mileage` | ₹1,000 | ₹26,992 | **27x** |
| `office_supplies` | ₹300 | ₹10,500 | **35x** |

The cause was not subtle once visible. `_split` set every leg to just under the
₹30,000 approval threshold regardless of the original claim — so a ₹380 taxi ride
became three ₹29,000 legs. `_ghost_vendor` drew a round amount from a fixed
₹5,000–₹40,000 range with no reference to the claim it replaced. Both wrote an
absolute amount into a world whose honest amounts are category-relative.

A LightGBM model trained on that data would have reported an excellent PR-AUC while
having learned one rule: *amount ≈ 26,000 ⇒ fraud*. It would have been measuring the
generator, which is the exact failure ADR 0004 was written to prevent — and it slipped
through anyway, because ADR 0004 constrains *where fraud comes from* and says nothing
about *which claims a given typology may be applied to*.

## Decision

Each typology draws from the population where it is actually possible:

- **SPLIT** — only claims already above `approval_threshold_minor`, and the legs now
  **sum to the original amount**. A splitter has a large real expense they want
  approved without a second signature; they do not conjure ₹29,000 out of a ₹380 taxi.
- **COLLUSION** — only the colluding employee's own claims.
- Everything else draws from the general pool.
- A typology whose pool is exhausted degrades to `INFLATION` rather than being forced
  onto an implausible claim.

`_ghost_vendor` now scales **relative to the claim it replaces** (2–6x, then snapped to
a round figure) instead of drawing from a fixed absolute range, so a ghost invoice in
office supplies is large-for-office-supplies rather than uniformly ₹20,000.

## Consequences

After the change, on the same seed:

| | Before | After |
|---|---|---|
| Worst honest:fraud median ratio | 70x | 1.9x |
| AUC of `amount_minor` alone | — | 0.694 |
| AUC of within-category amount z-score | — | 0.624 |
| `collusion` claims generated | 0 | 48 |

An AUC of ~0.69 from amount alone is the right shape: fraudulent claims *are* somewhat
larger, which is true in reality, but amount is nowhere near sufficient. The model now
has to reach for duplicate signals, velocity and employee history to do better — which
is the entire point of building those families.

`collusion` previously never fired at all: it was drawn from the mix and then
immediately degraded, because the odds of a uniformly-chosen target belonging to the
one colluding employee are negligible. Pools fixed that as a side effect.

Cost: `inject_fraud` is meaningfully more complex than a single loop, and the planning
pass allocates a shuffled index list per typology. Worth it.

## Alternatives considered

- **Lower the approval threshold until most claims exceed it.** Would have made splits
  common, but ARCHITECTURE.md §15 specifies ₹30,000 and, more importantly, it treats a
  symptom: ghost-vendor amounts would still have been category-blind.
- **Post-hoc rebalancing** — generate as before, then rescale fraudulent amounts to
  match the honest per-category distribution. Rejected: it would erase the amount
  signal entirely, which is equally dishonest in the opposite direction. Inflated
  claims genuinely *are* larger.
- **Leave it and document it as a known limitation.** Tempting, since the pipeline
  "worked". Rejected because every downstream number — PR-AUC, SHAP explanations, the
  agent's `fraud_reasons` — would have been quietly meaningless.

## Note for the writeup

This is the leakage story worth telling in an interview, and it is better than the
deliberate-leakage exercise in §11 because it was **not** deliberate. The lesson is
that the defence which caught it was not a test — no test asserted anything about
amount distributions — it was *looking at the data*: one `groupby(category, is_fraud)`
on medians, run before training anything.
