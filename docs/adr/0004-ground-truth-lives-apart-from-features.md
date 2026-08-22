# 0004. Ground truth is generated separately from behaviour, and carries its own timestamp

- **Status:** accepted
- **Date:** 2026-08-22

## Context

The simulator holds ground truth — we know which claims are fraudulent (§15). That is
the project's biggest asset and its biggest hazard. Two failure modes were visible
before writing a line of it:

1. **Generator leakage.** If fraudulent claims are produced inline with honest ones,
   it is very easy to give them a subtly different amount or timing distribution for
   reasons *unrelated to the fraud itself* — a different RNG call order is enough. A
   model then learns the generator, PR-AUC looks superb, and the number is fiction.
2. **Label-time leakage.** Fraud is confirmed by an audit weeks after submission
   (§11). A dataset carrying a plain `is_fraud` column invites a join that uses labels
   which had not arrived at prediction time.

## Decision

- `fsa_sim.world` generates **only honest spend**. `fsa_sim.adversarial` then
  *transforms* selected honest claims into fraudulent ones. Fraud is a mutation of the
  baseline population, never a separate draw from it.
- `ExpenseRecord` carries **no label**. Labels live in `FraudLabel` with
  `confirmed_at`, and un-audited claims have `confirmed_at = None`.
- The only sanctioned label join is `fsa_ml.features.pointintime.label_as_of`, which
  drops unconfirmed and future labels rather than coercing them to negatives.

## Consequences

- The honest world can be regenerated at a different fraud rate without changing a
  single legitimate claim — which is exactly what the M5 drift experiment (a new
  typology appearing at a known date) needs.
- "Not yet confirmed" is never treated as "not fraud". This *reduces* the training set
  and will make offline numbers look worse than a naive pipeline's. That is the point;
  the naive number was not real.
- Slightly more machinery than a single `generate()` that emits labelled rows.

## Alternatives considered

- **One generator emitting labelled claims.** Simpler, and the standard way synthetic
  fraud datasets are built. Rejected for reason (1) above.
- **`is_fraud` on the record, with discipline about when to read it.** Discipline is
  not a control. Separating the files makes the leak require deliberate effort.
