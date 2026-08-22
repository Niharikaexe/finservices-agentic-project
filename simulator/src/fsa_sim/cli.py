"""Simulator CLI.

    uv run python -m fsa_sim.cli generate --seed 42 --out data/worlds
    uv run python -m fsa_sim.cli inspect  --dir data/worlds/seed-42

argparse rather than Typer/Click on purpose: the simulator is a workspace member that
services do not depend on, and every dependency added here is one that a CI job has to
install. The stdlib does this fine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fsa_common import configure_logging, get_logger
from fsa_sim.world import WorldConfig, generate_world, save_world
from fsa_sim.world.persistence import load_table

log = get_logger(__name__)


def cmd_generate(args: argparse.Namespace) -> int:
    config = WorldConfig(seed=args.seed)
    out_dir = Path(args.out) / f"seed-{config.seed}"
    log.info("generating world", seed=config.seed, tenants=len(config.shapes))
    try:
        world = generate_world(config, with_spend=not args.org_only)
    except NotImplementedError as exc:
        log.error(
            "world generation is not implemented yet",
            missing=str(exc),
            hint="implement fsa_sim.world.spend, or pass --org-only to skip spend",
        )
        return 1
    counts = save_world(world, out_dir)
    log.info("world written", path=str(out_dir), **counts)
    for key, value in world.summary().items():
        print(f"{key:>18}: {value:,}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    directory = Path(args.dir)
    for table in ("tenants", "departments", "users", "vendors", "expenses"):
        try:
            frame = load_table(directory, table)
        except FileNotFoundError:
            print(f"{table:>14}: (absent)")
            continue
        print(f"{table:>14}: {len(frame):>9,} rows  {list(frame.columns)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(prog="fsa-sim", description="Argus world simulator")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate and persist a world")
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--out", default="data/worlds")
    gen.add_argument(
        "--org-only",
        action="store_true",
        help="skip spend generation (useful before spend.py is written)",
    )
    gen.set_defaults(func=cmd_generate)

    ins = sub.add_parser("inspect", help="summarise a generated world on disk")
    ins.add_argument("--dir", required=True)
    ins.set_defaults(func=cmd_inspect)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
