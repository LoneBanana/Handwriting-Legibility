"""In-memory sliding-window rate limits (per process).

Used to slow down password guessing and to cap Claude usage per learner. For several
API replicas behind a load balancer, put an equivalent limit in the proxy or a shared
store; per-process limits still bound each replica.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, status


class RateLimiter:
    def __init__(self, limit: int, window_seconds: float):
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str, message: str = "Too many attempts. Please wait a moment and try again.") -> None:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                retry = int(self.window - (now - hits[0])) + 1
                raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, detail=message,
                                    headers={"Retry-After": str(retry)})
            hits.append(now)
            if len(self._hits) > 50_000:  # drop idle keys so memory stays bounded
                for stale in [k for k, v in self._hits.items() if not v or now - v[-1] > self.window]:
                    del self._hits[stale]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


# Shared limiters (tests call .reset()).
login_limiter = RateLimiter(limit=10, window_seconds=15 * 60)        # per email + client address
register_limiter = RateLimiter(limit=20, window_seconds=60 * 60)     # per client address
llm_limiter = RateLimiter(limit=60, window_seconds=60 * 60)          # Claude calls per learner per hour
