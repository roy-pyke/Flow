"""Run, verify or replay a versioned local research experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.research.experiments import replay_bundle, run_experiment, verify_bundle
from backend.app.research.problems import capabilities, load_spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run config into a new immutable bundle")
    run.add_argument("config", type=Path)
    run.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify", help="Check artifact hashes, schema and field layout")
    verify.add_argument("bundle", type=Path)
    replay = commands.add_parser("replay", help="Recompute a bundle into a new output")
    replay.add_argument("bundle", type=Path)
    replay.add_argument("--output", type=Path, required=True)
    commands.add_parser("capabilities", help="List currently implemented problem types")
    args = parser.parse_args()
    try:
        if args.command == "run":
            result = run_experiment(load_spec(args.config), args.output, base_dir=args.config.parent)
        elif args.command == "verify":
            result = verify_bundle(args.bundle)
        elif args.command == "replay":
            result = replay_bundle(args.bundle, args.output)
        else:
            result = capabilities()
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f"Research experiment failed: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
