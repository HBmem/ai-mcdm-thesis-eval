import argparse
import sys

from pathlib import Path

from src.experiments.loader import load_experiments
from src.experiments.runner import build_plan


ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent deterministic MCDM evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("validate", "run"):
        command = sub.add_parser(name)
        command.add_argument("--experiments", type=Path, default=ROOT / "experiments")
        command.add_argument("--scenario", help="Scenario (S8/S08) or exact variant key; includes dependencies")
        command.add_argument("--ai", type=bool, help="")
        if name == "run":
            command.add_argument("--output", type=Path, default=ROOT / "results")
    
    args = parser.parse_args(argv)

    try:
        experiments = load_experiments(args.experiments)
        print("Experiments loaded successfully")
        plan = build_plan(experiments, args.scenario)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
if __name__ == "__main__":
    sys.exit(main())