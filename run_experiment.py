"""CLI entry point; paths to experiment resources are relative to their JSON file."""
import argparse
from pathlib import Path
import sys

from src.load_experiments import load_experiments
from src.runner import build_plan, execute_batch
from src.comprehension import AIConfig
from src.preflight import PreflightError, check_model_access

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent deterministic MCDM evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run"):
        command = sub.add_parser(name)
        command.add_argument("--experiments", type=Path, default=ROOT / "experiments")
        command.add_argument("--scenario", help="Scenario (S8/S08) or exact variant key; includes dependencies")
        if name == "run":
            command.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(load_experiments(args.experiments), args.scenario)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    if args.command == "validate":
        print(f"Validated {len(plan)} runs: " + ", ".join(c.key for c in plan))
        return 0
    try:
        ai_config = AIConfig.from_env()
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    try:
        with ai_config.client() as client:
            print("Checking AI key and model access...")
            preflight = check_model_access(ai_config, client)
            print("AI access check passed.")
            path, results = execute_batch(plan, args.output, ai_config=ai_config,
                                          ai_client=client, preflight=preflight)
    except PreflightError as error:
        print(f"AI startup check failed: {error}", file=sys.stderr)
        return error.exit_code
    except OSError as error:
        print(f"Artifact error: {error}", file=sys.stderr)
        return 1
    for r in results:
        print(f"{r.case_key}: {r.status}")
        for message in r.execution.get("messages", []):
            print(f"  {message}")
    print(f"Artifacts: {path}")
    return 0 if all(r.status == "passed" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
