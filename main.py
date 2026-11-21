"""CLI entry point"""
import argparse
import sys

from pathlib import Path
from contextlib import ExitStack

from src.experiments.loader import load_experiments
from src.experiments.runner import prepare_runs, execute_experiments
from src.experiments.storage import ArtifactWriteError
from src.ai.config import AIConfig
from src.ai.component import build_ai_component
from src.ai.preflight import PreflightError


ROOT = Path(__file__).resolve().parent


# Exit code policy:
# 0: validation succeeded, or execution completed with all runs passed.
# 1: a run failed/was blocked, or artifacts could not be written.
# 2: invalid configuration or an unsupported request such as AI execution.
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

    try:
        with ExitStack() as stack:
            ai_component = None

            if args.ai:
                config = AIConfig.from_env(ROOT / ".env")
                print("Checking AI key and model access...")
                ai_component = stack.enter_context(build_ai_component(config))
                print("AI access check passed.")

            batch_directory, results = execute_experiments(
                runs,
                args.output,
                ai_component=ai_component,
            )

            for result in results:
                print(f"{result.execution['execution_key']}: {result.status}")
    
                for message in result.execution.get("messages", []):
                    print(f"  {message}")
    
            print(f"Results: {batch_directory}")
    
            return (
                0
                if all(result.status == "passed" for result in results)
                else 1
            )

    except PreflightError as error:
        print(f"AI startup check failed: {error}", file=sys.stderr)
        return error.exit_code

    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    except ArtifactWriteError as error:
        print(f"Artifact error: {error}", file=sys.stderr)
        return 1
    
    # if args.ai:
    #     try:
    #         # Configure AI connection
    #         ai_config = AIConfig.from_env(ROOT / ".env")

    #         # Check AI connection
    #         with ai_config.client() as client:
    #             print("Checking AI key and model access...")
    #             preflight = check_model_access(ai_config, client)
    #             print("AI access check passed.")

    #             batch_directory, results = execute_experiments(
    #                 runs,
    #                 args.output,
    #                 ai_config=ai_config,
    #                 ai_client=client,
    #                 preflight=preflight
    #             )

    #     except ValueError as error:
    #         print(f"Configuration error: {error}", file=sys.stderr)
    #         return 2
    #     except PreflightError as error:
    #         print(f"AI startup check failed: {error}", file=sys.stderr)
    #         return error.exit_code
    #     except ArtifactWriteError as error:
    #         print(f"Artifact error: {error}",file=sys.stderr)
    #         return 1
    #     for result in results:
    #         print(f"{result.execution['execution_key']}: {result.status}")

    #         for message in result.execution.get("messages", []):
    #             print(f"  {message}")

    #     print(f"Results: {batch_directory}")

    #     return (
    #         0
    #         if all(result.status == "passed" for result in results)
    #         else 1
    #     )
    
    # else:
    #     try:
    #         batch_directory, results = execute_experiments(
    #             runs,
    #             args.output,
    #         )

    #     except ValueError as error:
    #         print(f"Configuration error: {error}", file=sys.stderr)
    #         return 2

    #     except ArtifactWriteError as error:
    #         print(f"Artifact error: {error}",file=sys.stderr)
    #         return 1

    #     for result in results:
    #         print(f"{result.execution['execution_key']}: {result.status}")

    #         for message in result.execution.get("messages", []):
    #             print(f"  {message}")

    #     print(f"Results: {batch_directory}")

    #     return (
    #         0
    #         if all(result.status == "passed" for result in results)
    #         else 1
    #     )


if __name__ == "__main__":
    sys.exit(main())