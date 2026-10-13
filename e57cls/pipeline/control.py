"""Versioned JSONL events and cooperative worker commands."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import threading
import time


class Cancelled(Exception):
    """Cooperative cancellation at a work-unit boundary."""


class Control:
    def __init__(self, output: Path, listen=True):
        self.paused = threading.Event()
        self.cancelled = threading.Event()
        self.events = (output/"events.jsonl").open("a", encoding="utf-8")
        self.started = time.monotonic()
        self.lock = threading.Lock()
        import psutil
        self.process = psutil.Process()
        self.peak_rss = self.process.memory_info().rss
        self.peak_handles = 0
        if listen:
            threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        for line in sys.stdin:
            try:
                command = json.loads(line)
                if command.get("version") != 1:
                    continue
                kind = command.get("command")
                if kind == "pause":
                    self.paused.set()
                elif kind == "resume":
                    self.paused.clear()
                elif kind == "cancel":
                    self.cancelled.set()
                    self.event('state',state='cancelling')
            except (ValueError, TypeError):
                logging.warning("Invalid worker command ignored")

    def event(self, kind, **payload):
        event = {"version": 1, "event": kind, "utc": datetime.now(timezone.utc).isoformat(),
                 "elapsed_seconds": time.monotonic()-self.started, **payload}
        line = json.dumps(event, ensure_ascii=False, allow_nan=False)
        with self.lock:
            print(line, flush=True)
            self.events.write(line+"\n")
            self.events.flush()

    def checkpoint(self):
        self.peak_rss = max(self.peak_rss,self.process.memory_info().rss)
        if hasattr(self.process,'num_handles'):
            self.peak_handles = max(self.peak_handles,self.process.num_handles())
        if self.cancelled.is_set():
            self.event("state", state="cancelled")
            raise Cancelled()
        if self.paused.is_set():
            self.event("state", state="paused")
            while self.paused.is_set() and not self.cancelled.is_set():
                time.sleep(.1)
            self.checkpoint()
            self.event("state", state="running")

    def close(self):
        self.events.close()
