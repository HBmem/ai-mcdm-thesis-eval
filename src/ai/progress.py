from dataclasses import dataclass
from typing import Callable, Literal


@dataclass(frozen=True, slots=True)
class PreflightProgress:
    attempt: int
    max_attempts: int
    state: Literal[
        "request_started",
        "retry_wait",
    ]
    delay_seconds: float | None = None
    code: str | None = None
    limit_source: str | None = None

PreflightProgressCallback = Callable[
    [PreflightProgress],
    None,
]
