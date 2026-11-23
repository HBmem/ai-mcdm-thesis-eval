import random
import time

from datetime import timezone
from email.utils import parsedate_to_datetime

from src.ai.backend import ProviderErrorDetails
from src.ai.comprehension.models import RetryDecision


def _retry_after_seconds(
    value: str,
    *,
    now: float,
) -> float | None:
    try:
        return max(0.0, float(value))
    except ValueError:
        pass

    try:
        retry_at = parsedate_to_datetime(value)

        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(
                tzinfo=timezone.utc
            )

        return max(
            0.0,
            retry_at.timestamp() - now,
        )
    except (TypeError, ValueError, OverflowError):
        return None


def _reset_delay_seconds(
    value: str,
    *,
    now: float,
) -> float | None:
    try:
        reset = float(value)
    except ValueError:
        return None

    # OpenRouter uses epoch milliseconds.
    if reset > 10_000_000_000:
        reset /= 1000

    return max(0.0, reset - now)


def choose_retry_delay(
    *,
    attempt: int,
    provider_details: ProviderErrorDetails | None,
    base_seconds: float,
    max_wait_seconds: float,
) -> RetryDecision | None:
    headers = (
        provider_details.headers
        if provider_details is not None
        else {}
    )
    now = time.time()

    retry_after = headers.get("retry-after")
    if retry_after is not None:
        delay = _retry_after_seconds(
            retry_after,
            now=now,
        )
        if delay is not None:
            return (
                RetryDecision(
                    delay_seconds=delay,
                    source="retry_after",
                )
                if delay <= max_wait_seconds
                else None
            )

    reset = headers.get("x-ratelimit-reset")
    if reset is not None:
        delay = _reset_delay_seconds(
            reset,
            now=now,
        )
        if delay is not None:
            # Avoid all waiting requests restarting together.
            delay += random.uniform(1.0, 5.0)

            return (
                RetryDecision(
                    delay_seconds=delay,
                    source="rate_limit_reset",
                )
                if delay <= max_wait_seconds
                else None
            )

    ceiling = min(
        base_seconds * (2 ** (attempt - 1)),
        max_wait_seconds,
    )

    return RetryDecision(
        delay_seconds=random.uniform(
            ceiling * 0.5,
            ceiling,
        ),
        source="exponential_backoff",
    )
