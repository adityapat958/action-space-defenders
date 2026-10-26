"""defend_log.py — non-blocking JSONL run logger for defend_rl / defend_lead.

One JSONL file per run in DEFEND_LOG_DIR (default: team4/logs/).
Each file: one metadata line, then one line per tick per drone.

Metadata line (written once on first tick or setup):
  {"meta": true, "weights": "...", "weights_sha256": "...",
   "env": {...}, "git_commit": "...", "t_start": float}

Tick line:
  {"t": float, "drone": "drone1",
   "own": {"pos": [x,y,z], "age": float, "src": "localize|telem_udp|none"},
   "attackers": [{"pos":[x,y,z], "vel":[vx,vy,vz], "age":float}, ...],
   "obs": [53 floats],
   "action": [4 floats],
   "rc": [lr, fb, ud, yaw],
   "flags": {"hover_stale": bool, "rc_capped": bool}}

All writes go through a queue + writer thread — non-blocking.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

import numpy as np

_REPO = Path(__file__).resolve().parents[3]  # Action_Space/
_DEFAULT_LOG_DIR = _REPO / "team4" / "logs"


def _sha256(path: str) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def _git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        return r.stdout.strip()
    except Exception:
        return ""


def _env_snapshot() -> dict:
    keys = [k for k in os.environ if k.startswith("DEFEND_") or k.startswith("LOCALIZE_")]
    return {k: os.environ[k] for k in keys}


def _j(obj) -> str:
    """Compact JSON serialiser that handles numpy arrays."""
    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        raise TypeError(repr(o))
    return json.dumps(obj, default=_default, separators=(",", ":"))


class DefendLogger:
    """Non-blocking JSONL logger.  One instance per policy (shared across drones)."""

    def __init__(self, policy_name: str = "defend", weights_path: str = ""):
        log_dir = Path(os.environ.get("DEFEND_LOG_DIR", str(_DEFAULT_LOG_DIR)))
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._path = log_dir / f"{policy_name}_{ts}.jsonl"
        self._q: queue.Queue = queue.Queue(maxsize=8192)
        self._writer = threading.Thread(target=self._loop, daemon=True, name="defend-log")
        self._writer.start()
        self._meta_done = False
        self._weights_path = weights_path
        self._t_start = time.time()

    def _loop(self) -> None:
        try:
            with open(self._path, "w", buffering=1) as f:
                while True:
                    try:
                        item = self._q.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    if item is None:
                        break
                    f.write(item + "\n")
        except OSError as exc:
            print(f"[defend_log] write error: {exc}", flush=True)

    def _put(self, obj: dict) -> None:
        try:
            self._q.put_nowait(_j(obj))
        except queue.Full:
            pass  # drop rather than block

    def _write_meta(self) -> None:
        self._meta_done = True
        self._put({
            "meta": True,
            "weights": self._weights_path,
            "weights_sha256": _sha256(self._weights_path),
            "env": _env_snapshot(),
            "git_commit": _git_commit(),
            "t_start": self._t_start,
        })

    def log_tick(
        self, t: float, drone: str,
        own_pos, own_age: float, own_src: str,
        attackers: list,  # list of (pos, vel, age)
        obs, action, rc: tuple,
        hover_stale: bool, rc_capped: bool,
    ) -> None:
        if not self._meta_done:
            self._write_meta()

        def _pos(a):
            return np.asarray(a, float).tolist() if a is not None else None

        atk_list = []
        for item in attackers:
            pos, vel, age = item
            atk_list.append({"pos": _pos(pos), "vel": _pos(vel), "age": float(age)})

        self._put({
            "t": round(float(t), 4),
            "drone": drone,
            "own": {
                "pos": _pos(own_pos),
                "age": round(float(own_age), 3),
                "src": own_src,
            },
            "attackers": atk_list,
            "obs": np.asarray(obs, float).tolist() if obs is not None else [],
            "action": np.asarray(action, float).tolist() if action is not None else [],
            "rc": list(rc),
            "flags": {"hover_stale": hover_stale, "rc_capped": rc_capped},
        })

    def close(self) -> None:
        self._q.put(None)
        self._writer.join(timeout=3.0)
        print(f"[defend_log] log written → {self._path}", flush=True)
