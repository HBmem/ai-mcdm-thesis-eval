"""Run the production schema-3 loader from the repository root.

    .venv/bin/python -m docs.examples.experiment_loading
"""
from src.experiments.loader import load_experiments
from src.experiments.participants import _participants
from src.experiments.variations import resolve_cases


if __name__ == "__main__":
    for experiment in load_experiments():
        print(experiment.session.id, len(experiment.session.participants), "participants")
