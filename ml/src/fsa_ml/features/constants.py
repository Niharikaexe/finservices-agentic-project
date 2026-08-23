"""Domain constants the feature layer needs.

These duplicate values that also live in `fsa_sim.world.config`, and that duplication
is deliberate and enforced: `import-linter` forbids `fsa_ml` from importing `fsa_sim`
(ARCHITECTURE.md §5). The reason is not tidiness — it is that the model must not be
able to read the simulator's parameters. A feature layer that imports the generator's
`approval_threshold_minor` is one refactor away from importing the generator's fraud
mix, and at that point the model is being told the answer.

In production these come from tenant configuration. Here they are constants, and the
mismatch risk is real: if the simulator's threshold changes and this does not, the
split detector goes blind. That is exactly the coupling a real feature store has to
manage too, so having it visible is honest.
"""

from __future__ import annotations

from typing import Final

# The approval limit a "splitter" tries to stay under. ₹30,000 in paise.
APPROVAL_THRESHOLD_MINOR: Final = 3_000_000

# How close to the limit counts as "suspiciously just under" it.
NEAR_THRESHOLD_FLOOR: Final = 0.80

# Duplicate near-match tolerance on amount, as a fraction.
DUPLICATE_AMOUNT_TOLERANCE: Final = 0.02

# Below this many prior claims, employee-historical features are not trustworthy and
# the row is flagged cold-start rather than being silently scored on noise.
COLD_START_CLAIMS: Final = 5
