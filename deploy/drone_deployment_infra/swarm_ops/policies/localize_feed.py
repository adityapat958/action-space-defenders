"""localize_feed.py — adapter for 06_localize.py as PRIMARY pose source.

06_localize.py (upstream jaromero3rd/Action_Space) outputs poses in the tag-10
ORIGIN frame: x right, y up, z out of tag face toward the drone (= map frame at origin).
It logs to stderr: "drone @ origin  x +1.23  y +0.45  z +2.10 m  via [10]"

We add three delivery mechanisms (tried simultaneously, first wins per tick):
  1. UDP socket: a wrapper or patched 06_localize sends JSON datagrams to
       LOCALIZE_UDP_PORT (default 9101).
       Each datagram: {"drone":"drone1","x":f,"y":f,"z":f,"yaw_deg":f,"t":f}
  2. Shared file: LOCALIZE_POSE_FILE path (tailed).  Same JSON, one line per update.
  3. stdin pipe: LOCALIZE_FROM_STDIN=1.  Same JSON on stdin.

Frame: 06_localize already gives origin/map frame (no wire flip needed unlike
attack_feed.py which receives raw camera xyz).
  map_to_sim: sim_x = map_z, sim_y = map_x, sim_z = map_y + origin_height.
  1 m in front of tag 10: origin/map = (0,0,1) → sim = (1,0,1.22). ✓

Fallback: if no data within LOCALIZE_TIMEOUT_S (default 2.0 s),
  own_state_with_age returns (None, inf) → caller uses AttackFeed UDP.

Unit test: python localize_feed.py --test
"""
from __future__ import annotations

import json
import math
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

# --- path setup ---
_DI = Path(__file__).resolve().parents[2]   # drone_deployment_infra/
sys.path.insert(0, str(_DI))
from defend_live import Frame, load_map      # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_MAP = _REPO / "drone_deployment_infra" / "config" / "map.yaml"

_ORIGIN_HEIGHT = 1.22   # metres: tag-10 centre above floor

LOCALIZE_UDP_PORT = int(os.environ.get("LOCALIZE_UDP_PORT", 9101))
LOCALIZE_TIMEOUT_S = float(os.environ.get("LOCALIZE_TIMEOUT_S", 2.0))
LOCALIZE_POSE_FILE = os.environ.get("LOCALIZE_POSE_FILE", "")
LOCALIZE_FROM_STDIN = os.environ.get("LOCALIZE_FROM_STDIN", "") == "1"


class LocalePose(NamedTuple):
    drone: str
    pos_sim: np.ndarray   # (3,) sim frame
    yaw_sim: float        # rad CCW from +x
    age: float            # seconds since measurement
    t_recv: float         # wall time received


def _make_frame(map_path=None) -> Frame:
    path = Path(map_path) if map_path else _DEFAULT_MAP
    if path.exists():
        info = load_map(path)
        return Frame("yup_zout", origin_height=_ORIGIN_HEIGHT, tags=info["tags"])
    return Frame("yup_zout", origin_height=_ORIGIN_HEIGHT, tags={})


def _parse_line(line: str, frame: Frame) -> "LocalePose | None":
    """Parse JSON line from 06_localize wrapper → LocalePose in sim frame.

    JSON: {"drone":"drone1","x":f,"y":f,"z":f,"yaw_deg":f,"t":f}
    x,y,z in origin/map frame (no wire flip; 06_localize already applied tag_to_map).
    """
    line = line.strip()
    if not line or line[0] != "{":
        return None
    try:
        d = json.loads(line)
    except (ValueError, KeyError):
        return None
    try:
        x, y, z = float(d["x"]), float(d["y"]), float(d["z"])
        drone = str(d.get("drone", ""))
        yaw_deg = float(d.get("yaw_deg", 0.0))
        t_meas = float(d.get("t", time.time()))
    except (KeyError, TypeError, ValueError):
        return None

    # Origin frame = map frame at origin tag.  map_to_sim applies directly.
    p_map = np.array([x, y, z], float)
    pos_sim = frame.map_to_sim(p_map)
    if pos_sim is None:
        # No map loaded → use raw axis mapping
        pos_sim = np.array([z, x, y + _ORIGIN_HEIGHT], float)

    # yaw: 06_localize convention 0=facing tag, +CW (same as drone_telemetry)
    yaw_sim = math.radians(-yaw_deg)

    age = max(0.0, time.time() - t_meas)
    return LocalePose(drone=drone, pos_sim=pos_sim, yaw_sim=yaw_sim,
                      age=age, t_recv=time.time())


class LocalizeFeed:
    """Consumes 06_localize output; exposes per-drone poses in sim frame.

    Thread-safe. Falls back silently when 06_localize is not running.
    """

    def __init__(self, map_path=None,
                 udp_port: int = LOCALIZE_UDP_PORT,
                 pose_file: str = LOCALIZE_POSE_FILE,
                 from_stdin: bool = LOCALIZE_FROM_STDIN):
        self._frame = _make_frame(map_path)
        self._lock = threading.Lock()
        self._poses: dict[str, LocalePose] = {}
        self._running = True
        self._sock = None
        self._threads: list[threading.Thread] = []

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.settimeout(1.0)
            s.bind(("0.0.0.0", udp_port))
            self._sock = s
            t = threading.Thread(target=self._udp_loop, daemon=True, name="localize-udp")
            t.start(); self._threads.append(t)
        except OSError as exc:
            print(f"[localize_feed] UDP :{udp_port} unavailable: {exc}", flush=True)

        if pose_file:
            t = threading.Thread(target=self._file_loop, args=(pose_file,),
                                 daemon=True, name="localize-file")
            t.start(); self._threads.append(t)

        if from_stdin:
            t = threading.Thread(target=self._stdin_loop, daemon=True, name="localize-stdin")
            t.start(); self._threads.append(t)

    def _ingest(self, line: str) -> None:
        pose = _parse_line(line, self._frame)
        if pose is None:
            return
        with self._lock:
            existing = self._poses.get(pose.drone)
            if existing is None or pose.t_recv >= existing.t_recv:
                self._poses[pose.drone] = pose

    def _udp_loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            self._ingest(data.decode(errors="ignore"))

    def _file_loop(self, path: str) -> None:
        p = Path(path); pos = 0
        while self._running:
            try:
                if p.exists():
                    with open(p) as f:
                        f.seek(pos)
                        for line in f:
                            self._ingest(line)
                        pos = f.tell()
            except OSError:
                pass
            time.sleep(0.05)

    def _stdin_loop(self) -> None:
        for line in sys.stdin:
            if not self._running:
                break
            self._ingest(line)

    def own_state_with_age(self, drone_name: str) -> "tuple[LocalePose | None, float]":
        """Return (pose, age_s) or (None, inf) if never seen / timeout."""
        with self._lock:
            pose = self._poses.get(drone_name)
        if pose is None:
            return None, math.inf
        age = time.time() - pose.t_recv
        return pose, age

    def latest_attackers(self, own_name: str | None = None,
                         max_age: float = LOCALIZE_TIMEOUT_S) -> list[LocalePose]:
        with self._lock:
            snap = dict(self._poses)
        now = time.time()
        return [p for n, p in snap.items()
                if n != own_name and now - p.t_recv < max_age]

    def close(self) -> None:
        self._running = False
        if self._sock:
            self._sock.close()


# ---------------------------------------------------------------------------
def _run_test():
    """1 m in front of tag 10 in origin frame → sim (1, 0, 1.22)."""
    frame = _make_frame(None)
    t_now = time.time()
    # origin/map: (x=0, y=0, z=1) = 1 m along z (outward normal of tag 10)
    line = json.dumps({"drone": "drone1", "x": 0.0, "y": 0.0, "z": 1.0,
                       "yaw_deg": 0.0, "t": t_now})
    pose = _parse_line(line, frame)
    assert pose is not None, "parse failed"
    expected = np.array([1.0, 0.0, 1.22])
    diff = np.abs(pose.pos_sim - expected)
    print(f"[localize_feed TEST] sim={pose.pos_sim} expected={expected} "
          f"max_err={diff.max():.4f} m")
    assert diff.max() < 0.01, f"FAIL: {pose.pos_sim}"
    print("[localize_feed TEST] PASS")


if __name__ == "__main__":
    if "--test" in sys.argv:
        _run_test()
    else:
        print("Usage: python localize_feed.py --test")
