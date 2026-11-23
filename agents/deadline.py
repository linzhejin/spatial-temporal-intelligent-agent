"""Share the run's monotonic deadline with synchronous provider calls."""
from contextlib import contextmanager
from contextvars import ContextVar
import time

_deadline = ContextVar("agent_deadline", default=None)


@contextmanager
def shared_deadline(value):
    token = _deadline.set(value)
    try:
        yield
    finally:
        _deadline.reset(token)


def limited_timeout(default):
    deadline = _deadline.get()
    if deadline is None:
        return float(default)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("run deadline exceeded")
    return min(float(default), remaining)


def is_bounded():
    return _deadline.get() is not None


def current_deadline():
    """Return the active monotonic deadline, if the caller is bounded."""
    return _deadline.get()
