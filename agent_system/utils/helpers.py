"""Small, reusable helper functions."""

from __future__ import annotations

import time
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import structlog

logger = structlog.get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def async_timed(
    func: Callable[P, Coroutine[Any, Any, R]],
) -> Callable[P, Coroutine[Any, Any, tuple[R, float]]]:
    """Decorator that returns ``(result, elapsed_ms)``."""

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[R, float]:
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(
            "async_timed",
            function=func.__qualname__,
            elapsed_ms=round(elapsed_ms, 2),
        )
        return result, elapsed_ms

    return wrapper


def truncate(text: str, max_length: int = 500) -> str:
    """Truncate *text* to *max_length* characters, appending '…' if cut."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 1] + "…"