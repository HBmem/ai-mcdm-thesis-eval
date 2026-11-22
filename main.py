"""CLI entry point"""
import argparse
import os
import sys

from pathlib import Path
from contextlib import ExitStack

from src.experiments.loader import load_experiments
from src.experiments.progress import (
    AIRepetitionProgress,
    ProgressEvent,
    RunProgress,
    StageProgress,
)
from src.experiments.runner import prepare_runs, execute_experiments
from src.experiments.storage import ArtifactWriteError
from src.ai.config import AIConfig
from src.ai.component import build_ai_component
from src.ai.preflight import PreflightError


ROOT = Path(__file__).resolve().parent


class CLIProgressReporter:
    def __init__(self, stream=sys.stdout):
        self.stream = stream
        self.use_color = (
            hasattr(stream, "isatty")
            and stream.isatty()
            and "NO_COLOR" not in os.environ
        )

    def color(self, code: str, text: str) -> str:
        if not self.use_color:
            return text

        return f"\033[{code}m{text}\033[0m"

    def _duration(self, seconds: float | None) -> str:
        return (
            f" in {seconds:.2f}s"
            if seconds is not None
            else ""
        )

    def _outcome(self, state: str) -> tuple[str, str]:
        outcomes = {
            "completed": ("✓", "1;32"),
            "partial": ("◐", "1;33"),
            "blocked": ("■", "1;33"),
            "skipped": ("○", "1;33"),
            "failed": ("✗", "1;31"),
        }
        return outcomes.get(state, ("•", "0"))

    def __call__(self, event: ProgressEvent) -> None:
        if isinstance(event, RunProgress):
            position = f"[{event.current}/{event.total}]"

            if event.state == "started":
                message = self.color(
                    "0;36",
                    f"  ├─ {position} {event.execution_key}",
                )
            else:
                symbol, color = self._outcome(event.state)
                message = self.color(
                    color,
                    (
                        f"  │  {symbol} {event.execution_key}: "
                        f"{event.state}"
                        f"{self._duration(event.duration_seconds)}"
                    ),
                )

            if event.detail:
                message += f" — {event.detail}"

            print(message, file=self.stream, flush=True)
            return

        if isinstance(event, AIRepetitionProgress):
            position = f"[{event.repetition}/{event.total}]"

            if event.state == "started":
                message = self.color(
                    "0;36",
                    (
                        f"  │  ├─ AI repetition {position} "
                        f"for {event.execution_key}"
                    ),
                )
            else:
                symbol, color = self._outcome(event.state)
                message = self.color(
                    color,
                    (
                        f"  │  │  {symbol} repetition "
                        f"{position}: {event.state}"
                        f"{self._duration(event.duration_seconds)}"
                    ),
                )

            if event.detail:
                message += f" — {event.detail}"

            print(message, file=self.stream, flush=True)
            return

        label = (
            "AI"
            if event.stage == "ai"
            else event.stage.replace("_", " ").title()
        )

        if event.state == "started":
            message = self.color(
                "1;36",
                f"\n╭─ {label}",
            )

        elif event.state == "completed":
            message = self.color(
                "1;32",
                (
                    f"╰─ ✓ {label} completed"
                    f"{self._duration(event.duration_seconds)}"
                ),
            )

        elif event.state == "skipped":
            message = self.color(
                "1;33",
                f"\n○ {label} skipped",
            )

        elif event.state == "partial":
            message = self.color(
                "1;33",
                (
                    f"╰─ ◐ {label} partially completed"
                    f"{self._duration(event.duration_seconds)}"
                ),
            )

        else:
            message = self.color(
                "1;31",
                f"╰─ ✗ {label} failed",
            )

        if event.detail:
            message += f" — {event.detail}"

        print(message, file=self.stream, flush=True)


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
            reporter = CLIProgressReporter()

            if args.ai:
                config = AIConfig.from_env(ROOT / ".env")
                print("Checking AI key and model access...")
                ai_component = stack.enter_context(build_ai_component(config))
                print("AI access check passed.")

            batch_directory, results = execute_experiments(
                runs,
                args.output,
                ai_component=ai_component,
                on_progress=reporter,
            )

            for result in results:
                print(f"{result.execution['execution_key']}: {result.status}")
    
                for message in result.execution.get("messages", []):
                    print(f"  {message}")
    
            print(f"Results: {batch_directory}")
    
            numerical_succeeded = all(
                result.status == "passed"
                for result in results
            )

            ai_succeeded = all(
                result.execution.get("ai_status")
                in {"completed", "not_requested"}
                for result in results
            )

            return (
                0
                if numerical_succeeded and ai_succeeded
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


if __name__ == "__main__":
    sys.exit(main())
