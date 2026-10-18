"""Ingestion progress shared by terminal and browser clients."""
from __future__ import annotations

import sys
import time
from collections.abc import Callable

Progress = Callable[[str, int, int], None]


class TerminalProgress:
    def __init__(self):
        self.stage = ""
        self.started = self.updated = time.monotonic()

    def __call__(self, stage: str, done: int, total: int) -> None:
        now = time.monotonic()
        changed = stage != self.stage
        if changed:
            self.stage, self.started = stage, now
        if not changed and done != total and now - self.updated < 1:
            return
        self.updated = now
        elapsed = now - self.started
        fraction = done / total if total else 0
        eta = f" | ETA {(elapsed / done * (total - done)):.0f}s" if done and done < total else ""
        message = f"{stage}: {done:,}/{total:,} ({fraction:.0%}) | {elapsed:.0f}s{eta}"
        interactive = sys.stderr.isatty()
        print(message.ljust(100) if interactive else message,
              file=sys.stderr, end="\r" if interactive and done < total else "\n", flush=True)
