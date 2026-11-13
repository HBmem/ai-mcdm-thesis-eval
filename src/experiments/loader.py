from pathlib import Path

from src.experiments.models import Experiment
def load_experiments(experiments_dir: Path | str | None = None) -> list[Experiment]:
    """
    
    """

    directory = (
        Path(experiments_dir)
        if experiments_dir is not None
        else Path(__file__).resolve().parent.parent / "experiments"
    )

    if not directory.is_dir():
        raise ValueError(f"Experiment directory does not exists: {directory}")

    experiment
