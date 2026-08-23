"""ML platform CLI.

    make install-ml
    make train-fraud
    uv run --group ml python -m fsa_ml.cli train-fraud \
        --world data/worlds/seed-42 --train-as-of 2026-09-01 --eval-as-of 2027-06-01

`--train-as-of` and `--eval-as-of` are required rather than defaulted to `now()`, and
that is the point: a training run you cannot date is a run you cannot reproduce, and
`datetime.now()` is the value that turns a backtest into fiction.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from fsa_common import configure_logging, get_logger

log = get_logger(__name__)


def _date(value: str) -> datetime:
    return datetime.fromisoformat(value)


def cmd_features(args: argparse.Namespace) -> int:
    from fsa_ml.features.build import assemble_evaluation

    train, test = assemble_evaluation(Path(args.world), args.train_as_of, args.eval_as_of)
    log.info(
        "features assembled",
        train_rows=len(train.features),
        train_positives=int(train.labels.sum()),
        train_rate=round(train.positive_rate, 4),
        test_rows=len(test.features),
        test_positives=int(test.labels.sum()),
        n_features=len(train.features.columns),
    )
    print(f"\n{len(train.features.columns)} features:")
    for name in train.features.columns:
        print(f"  {name}")
    return 0


def cmd_train_fraud(args: argparse.Namespace) -> int:
    from fsa_ml.fraud.train import train_from_disk

    try:
        model = train_from_disk(Path(args.world), args.train_as_of, args.eval_as_of, seed=args.seed)
    except NotImplementedError as exc:
        log.error(
            "fraud training is not implemented yet",
            missing=str(exc),
            hint="implement fsa_ml.fraud.train.train",
        )
        return 1

    print("\nheld-out performance")
    print(model.metrics.render())
    print("\nbaseline (within-category amount percentile)")
    print(model.baseline_metrics.render())
    lift = model.metrics.pr_auc / max(model.baseline_metrics.pr_auc, 1e-9)
    print(f"\nPR-AUC lift over baseline: {lift:.2f}x")
    if lift < 1.15:
        print("  ^ the model does not clearly beat a three-line rule. That is a result,")
        print("    not a failure — report it rather than tuning until it goes away.")
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(prog="fsa-ml", description="Argus ML platform")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, handler, help_text in (
        ("features", cmd_features, "assemble and describe the training matrix"),
        ("train-fraud", cmd_train_fraud, "train and evaluate the fraud model"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--world", default="data/worlds/seed-42")
        cmd.add_argument("--train-as-of", type=_date, required=True)
        cmd.add_argument("--eval-as-of", type=_date, required=True)
        cmd.add_argument("--seed", type=int, default=42)
        cmd.set_defaults(func=handler)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
