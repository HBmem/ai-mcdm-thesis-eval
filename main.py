"""CLI entry point"""
import argparse
import sys

from pathlib import Path

from src.experiments.loader import load_experiments
from src.experiments.runner import prepare_runs, execute_experiments
from src.ai.config import AIConfig
from src.ai.preflight import PreflightError, check_model_access


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
            command.add_argument("--ai", action=argparse.BooleanOptionalAction, default=False, help="Run AI stages after numerical calculations (default: disabled).")
    
    args = parser.parse_args(argv)

    # Load and prepare experiments
    try:
        experiments = load_experiments(args.experiments)
        print("Experiments loaded successfully")
        runs = prepare_runs(
            experiments=experiments,
            scenario=args.scenario
        )
        print("Experiments prepared successfully")
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    if args.command == "validate":
        print(f"Validated {len(runs)} runs: " + ", ".join(c.key for c in runs))
        return 0

    if args.ai:
        try:
            # Configure AI connection
            ai_config = AIConfig.from_env(ROOT / ".env")

            # Check AI connection
            with ai_config.client() as client:
                print("Checking AI key and model access...")
                preflight = check_model_access(ai_config, client)
                print("AI access check passed.")

            # TODO: Execute numerical and AI stages here
            # Pass ai_config and client to your executor.
            # Keep model calls inside this context.
            path, results = execute_experiments(
                runs,
                args.output,
                ai_config=ai_config,
                ai_client=client,
                preflight=preflight
            )
        except ValueError as error:
            print(f"Configuration error: {error}", file=sys.stderr)
            return 2
        except PreflightError as error:
            print(f"AI startup check failed: {error}", file=sys.stderr)
            return error.exit_code
    else:
        try:
            path, results = execute_experiments(runs, args.output)
        except ValueError as error:
            print(f"Configuration error: {error}", file=sys.stderr)
            return 2


if __name__ == "__main__":
    sys.exit(main())