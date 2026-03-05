"""Simple performance timing for pipeline stages."""

import time
from typing import Dict, List
from contextlib import contextmanager


class PerformanceMonitor:
    """Track elapsed time per stage."""

    def __init__(self):
        self._times: Dict[str, List[float]] = {}
        self._current: Dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        """Context manager to time a stage."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._times.setdefault(name, []).append(elapsed)

    @contextmanager
    def track(self, name: str):
        """Context manager to time a stage (alias for stage, spec API)."""
        with self.stage(name):
            yield

    def print_summary(self) -> None:
        """Log summary of recorded stage times."""
        for name, times in self._times.items():
            total = sum(times)
            count = len(times)
            avg = total / count if count else 0
            # Use logging if available, else print
            try:
                from src.utils.logger import get_logger
                get_logger(__name__).info(
                    "  %s: %.3fs (avg over %d run(s))", name, avg, count
                )
            except Exception:
                print(f"  {name}: {avg:.3f}s (avg over {count} run(s))")

    def record(self, name: str, elapsed_seconds: float) -> None:
        """Record a stage duration manually."""
        self._times.setdefault(name, []).append(elapsed_seconds)

    def summary(self) -> Dict[str, float]:
        """Return mean time per stage (seconds)."""
        return {
            name: sum(times) / len(times) if times else 0.0
            for name, times in self._times.items()
        }

    def reset(self) -> None:
        """Clear all recorded times."""
        self._times.clear()
        self._current.clear()
