"""Small in-process throttle for repeated failed login attempts.

This is the application safety net. A production reverse proxy should also apply a coarse
per-IP request limit so a flood is rejected before it consumes an application worker.
"""

import threading
import time
from collections import deque


class LoginLimiter:
    def __init__(self, max_failures: int, window_seconds: float, lockout_seconds: float) -> None:
        self.max_failures = max_failures
        self.window_seconds = window_seconds
        self.lockout_seconds = lockout_seconds
        self._failures: dict[str, deque[float]] = {}
        self._locked_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def retry_after(self, key: str) -> int:
        now = time.monotonic()
        with self._lock:
            until = self._locked_until.get(key, 0.0)
            if until <= now:
                self._locked_until.pop(key, None)
                return 0
            return max(1, int(until - now + 0.999))

    def failed(self, key: str) -> int:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            failures = self._failures.setdefault(key, deque())
            while failures and failures[0] < cutoff:
                failures.popleft()
            failures.append(now)
            if len(failures) < self.max_failures:
                return 0
            until = now + self.lockout_seconds
            self._locked_until[key] = until
            failures.clear()
            return max(1, int(self.lockout_seconds + 0.999))

    def succeeded(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)
            self._locked_until.pop(key, None)
