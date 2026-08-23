# Notebooks and Colab

## The rule

**Pipeline code lives in `ml/`. Notebooks call it; they never contain it.**

This is not notebook snobbery — it is a specific production constraint. The feature
code that trains the model must be *the same code* that serves it, or you get
training/serving skew: the scorer computes `eh_amount_zscore_cat_90d` slightly
differently from the trainer, and the model silently degrades in a way no test catches.
`services/fraud-scorer` imports `fsa_ml.features`. It cannot import a notebook cell.

So a notebook may:
- explore and plot (`df.groupby(...)`, distributions, calibration curves)
- call `fsa_ml.features.build.assemble()` and inspect what comes back
- try a modelling idea before it earns a place in the repo

A notebook may not:
- define a feature that ends up in the model
- hold the only copy of a training loop
- be the thing you point at in an interview when asked "how does this get to production"

When an experiment works, it moves into `ml/` with a test. That migration is the
workflow, not an afterthought.

## Should you use Google Colab?

**For this project, mostly no — with two real exceptions.**

Everything so far runs in ~25 seconds on a laptop: 82k claims, 47 features, and a
LightGBM fit on 48k rows is CPU work measured in seconds. Colab adds friction (re-clone
and re-install on every disconnect, a filesystem that vanishes, no `make lint`, git via
web UI) and buys nothing.

Use Colab when:

1. **M7, the LoRA receipt-extraction fine-tune.** That genuinely needs a GPU. Colab's
   free T4 is a legitimate alternative to renting a spot `Standard_NC8as_T4_v3`, and
   materially cheaper — the cost log in `docs/cost-log.md` should say which you used.
2. **Your machine cannot run the stack.** Docker with Postgres, Redpanda, OpenFGA and
   the observability plane wants ~8GB. If that is the constraint, Colab is a reasonable
   place to do the ML half while the platform half waits.

## Colab bootstrap

If you do use it, this cell gives you the real package rather than a copy of it:

```python
# 1. Clone the repo (private repo: use a fine-grained PAT with read-only Contents
#    scope, and paste it into Colab's Secrets panel, never into a cell).
from google.colab import userdata
import subprocess, os

token = userdata.get("GITHUB_PAT")
!git clone -q https://{token}@github.com/Niharikaexe/finservices-agentic-project.git
%cd finservices-agentic-project
!git checkout -q claude/ml-training-pipeline-setup-85035p

# 2. Install the workspace packages editable, so edits in the file browser take effect.
!pip install -q -e packages/common -e ml -e simulator
!pip install -q lightgbm shap optuna

# 3. Generate the world (~15s) — or upload data/worlds/seed-42 from your machine.
!python -m fsa_sim.cli generate --seed 42 --out data/worlds
```

Then work against the real API:

```python
from datetime import datetime
from pathlib import Path
from fsa_ml.features.build import assemble_evaluation

train, test = assemble_evaluation(
    Path("data/worlds/seed-42"),
    train_as_of=datetime(2026, 9, 1),
    eval_as_of=datetime(2027, 6, 1),
)
train.features.describe().T
```

**Getting work back out:** edit files, then commit from a cell
(`!git add -A && git commit -m "..." && git push`). Do not paste cell contents into
the repo by hand — that is how a notebook version and a repo version diverge.

## Suggested split

| Where | What |
|---|---|
| Repo + `make test` | features, training, evaluation, anything the scorer imports |
| Notebook (local Jupyter or Colab) | looking at distributions, calibration curves, SHAP plots, "what if I tried…" |
| Colab specifically | the M7 GPU fine-tune; a fallback if Docker will not run locally |

A local notebook is strictly better than Colab for the first two, because
`uv run jupyter lab` already sees the installed workspace with no bootstrap cell at all.
