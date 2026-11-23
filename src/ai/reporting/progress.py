from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias


@dataclass(frozen=True, slots=True)
class ReportingCallProgress:
    stage: Literal[
        "data_explanation_reporting"
    ]
    attempt: int
    max_attempts: int

    state: Literal[
        "request_started",
        "retry_wait",
    ]

    delay_seconds: float | None = None
    code: str | None = None
    limit_source: str | None = None


ReportingProgressCallback: TypeAlias = Callable[
    [ReportingCallProgress],
    None,
]
