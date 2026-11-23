from dataclasses import dataclass
from typing import Literal, TypeAlias

from src.ai.comprehension.progress import DCLCallProgress
from src.ai.progress import PreflightProgress


StageName = Literal[
    "numerical",
    "validation",
    "ai",
]

StageState = Literal[
    "started",
    "completed",
    "partial",
    "failed",
    "skipped",
]


@dataclass(frozen=True, slots=True)
class StageProgress:
    stage: StageName
    state: StageState
    duration_seconds: float | None = None
    detail: str | None = None


RunState = Literal[
    "started",
    "completed",
    "failed",
    "blocked",
    "skipped",
    "partial",
]


@dataclass(frozen=True, slots=True)
class RunProgress:
    stage: StageName
    execution_key: str
    state: RunState
    current: int
    total: int
    duration_seconds: float | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class AIRepetitionProgress:
    execution_key: str
    repetition: int
    total: int
    state: RunState
    duration_seconds: float | None = None
    detail: str | None = None


ProgressEvent: TypeAlias = (
    StageProgress
    | RunProgress
    | AIRepetitionProgress
    | DCLCallProgress
    | PreflightProgress
)
