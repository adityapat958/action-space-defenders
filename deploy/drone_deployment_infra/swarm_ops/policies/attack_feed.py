"""attack_feed.py — background UDP listener for the drone_telemetry v1 wire format.

Reads datagrams from drone_gui / drone_telemetry.  Every snapshot holds ALL drones;
this module filters by team and converts poses to sim frame so the defend policies
receive the same (pos_sim, yaw_rad) they were trained on.

Frame pipeline:
  wire xyz: camera position in the SEEN tag's frame (x right, y down, z out toward drone)
    -- wire_to_map_tag (flip y,z: diag[1,-1,-1]) -->
  tag frame in map convention (x right, y up, z out of tag face) — matches map.yaml storage
    -- tag_to_map(tag, xyz) -->  map frame  (origin tag-10; y up, z out; from config/map.yaml)
    -- map_to_sim(p_map)    -->  sim frame  (z up, +x into room, +y = map +x)
  yaw: drone_telemetry deg (0=facing tag, +CW looking down) -> sim rad (CCW from +x)
  convention: "yup_zout" — matches the map.yaml frame string and how map_tags.py stored R,t
  origin_height: ~1.22 m (floor tags at y≈-1.22 in map frame → tag-10 centre 1.22 m above floor)

For the defender's OWN pose the caller should pass its drone name; this module returns
it separately so the policy can update own_pos / own_yaw.

Port default: 9100 (roles.yaml default destination).
Env var: DEFEND_TELEM_PORT overrides.
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

# ---- path to defend_live for Frame / load_map ----
_DI = Path(__file__).resolve().parents[2]   # drone_deployment_infra/
sys.path.insert(0, str(_DI))
from defend_live import Frame, load_map      # noqa: E402

_REPO = Path(__file__).resolve().parents[3]  # Action_Space/
_DEFAULT_MAP = _REPO / "drone_deployment_infra" / "config" / "map.yaml"

DEFAULT_PORT = int(os.environ.get("DEFEND_TELEM_PORT", 9100))
# After STALE_S the pose is treated as gone and the policy hovers.
# Between 0 and STALE_S the policy flies on EKF dead-reckoning at a reduced speed.
# Override with env var DEFEND_POSE_TIMEOUT_S (seconds, float).
STALE_S = float(os.environ.get("DEFEND_POSE_TIMEOUT_S", 5.0))
MAX_ATK_SLOTS = 4    # matches defend_live NUM_SLOTS


class DroneState(NamedTuple):
    name: str
    team: str           # "attack" | "defense" | ""
    pos_sim: np.ndarray | None   # (3,) sim frame, or None
    yaw_sim: float | None        # rad, CCW from +x, or None
    pose_age: float              # seconds (inf if never seen)
    height_m: float | None       # Tello h sensor (m above takeoff)
    fly: bool


_ORIGIN_HEIGHT = 1.22   # tag-10 centre above the floor [m]; floor tags sit at map y ≈ -1.22
# Wire → map-tag axis flip: drone_telemetry sends camera xyz with y-down, z negative in front of the tag.
# map.yaml stores tag poses with y-up, z-toward-drone (map_tags.py C_FLIP = diag[1,-1,-1]
# ). So flip y and z, exactly like C_FLIP.
_C_WIRE = np.array([1.0, -1.0, -1.0])  # (x,y,z)->(x,-y,-z). Wire = cam_in_tag: x right, y down, z NEGATIVE in front
# (drone_gui/tello_unit.py: ahead = -cam[2]; sim_drone truth z = -dist). Same as map_tags.py C_FLIP.


def _make_frame(map_path: str | Path | None) -> Frame:
    """Build a Frame from map.yaml using the correct y-up/z-out convention."""
    path = Path(map_path) if map_path else _DEFAULT_MAP
    if path.exists():
        info = load_map(path)
        return Frame("yup_zout", origin_height=_ORIGIN_HEIGHT, tags=info["tags"])
    # No map: identity frame (sim ≈ map; xy will be wrong but z/height still usable)
    return Frame("yup_zout", origin_height=_ORIGIN_HEIGHT, tags={})


def _parse_drone(d: dict, t_recv: float, frame: Frame) -> DroneState:
    """Convert one drone entry from the v1 JSON into a DroneState in sim frame."""
    p = d.get("pose")
    pos_sim, yaw_sim, pose_age = None, None, math.inf
    if p is not None and p.get("xyz") is not None:
        tag = int(p["tag"])
        xyz = p["xyz"]
        # Flip y: wire is y-down, map.yaml stores y-up (same z direction: toward drone)
        xyz_map = np.asarray(xyz, float) * _C_WIRE
        p_map = frame.tag_to_map(tag, xyz_map)
        if p_map is not None:
            pos_sim = frame.map_to_sim(p_map)
            if p.get("yaw") is not None:
                yaw_sim = frame.yaw_from_tag(tag, float(p["yaw"]))
            age = float(p.get("age", 0.0))
            pose_age = age   # seconds since the camera measured it (grows while stale)
    h = d.get("h")
    return DroneState(
        name=str(d.get("name", "")),
        team=str(d.get("team") or ""),
        pos_sim=pos_sim,
        yaw_sim=yaw_sim,
        pose_age=pose_age,
        height_m=float(h) if h is not None else None,
        fly=bool(d.get("fly", False)),
    )


class AttackFeed:
    """Listens on a UDP port for drone_telemetry v1 datagrams in a background thread.

    Thread-safe reads: latest_attackers() and own_state(name).
    Call close() when done (or let the daemon thread die with the process).
    """

    def __init__(self, port: int = DEFAULT_PORT, host: str = "0.0.0.0",
                 map_path: str | Path | None = None):
        self.port = port
        self._frame = _make_frame(map_path)
        self._lock = threading.Lock()
        self._snapshot: list[DroneState] = []
        self._t_snap: float = 0.0
        self._running = True
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(1.0)
        self._sock.bind((host, port))
        self._thread = threading.Thread(target=self._loop, daemon=True, name="attack-feed")
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            t_recv = time.time()
            try:
                msg = json.loads(data)
            except (ValueError, KeyError):
                continue
            if msg.get("v") != 1 or "drones" not in msg:
                continue
            snap = [_parse_drone(d, t_recv, self._frame) for d in msg["drones"]]
            with self._lock:
                self._snapshot = snap
                self._t_snap = t_recv

    def latest_attackers(self, max_age: float = STALE_S) -> list[DroneState]:
        """Return DroneStates for team='attack' drones with a fresh pose, up to MAX_ATK_SLOTS."""
        with self._lock:
            snap = list(self._snapshot)
        return [s for s in snap if s.team == "attack" and s.pos_sim is not None and s.pose_age <= max_age][:MAX_ATK_SLOTS]

    def latest_attackers_with_age(self) -> list[tuple["DroneState", float]]:
        """Return (DroneState, pose_age_s) for all attack drones whose pose arrived within STALE_S.

        pose_age < STALE_S: valid — use position.  Caller applies rc speed scale.
        pose_age >= STALE_S: treated as gone (drone dead / out of range).
        """
        with self._lock:
            snap = list(self._snapshot)
        return [(s, s.pose_age) for s in snap
                if s.team == "attack" and s.pos_sim is not None and s.pose_age < STALE_S][:MAX_ATK_SLOTS]

    def own_state(self, drone_name: str, max_age: float = STALE_S) -> DroneState | None:
        """Return the DroneState for the named defender (team='defense'), or None if stale/absent."""
        with self._lock:
            snap = list(self._snapshot)
        for s in snap:
            if s.name == drone_name:
                if s.pos_sim is not None and s.pose_age <= max_age:
                    return s
                return None   # found but stale
        return None

    def own_state_with_age(self, drone_name: str) -> tuple["DroneState | None", float]:
        """Return (DroneState_or_None, pose_age_s) regardless of staleness.

        Returns (None, inf) if the drone has never been seen.
        The age lets the caller decide whether to use the EKF prediction (age < STALE_S)
        or hover (age >= STALE_S).
        """
        with self._lock:
            snap = list(self._snapshot)
        for s in snap:
            if s.name == drone_name:
                if s.pos_sim is not None:
                    return s, s.pose_age
                return None, math.inf
        return None, math.inf

    def snap_age(self) -> float:
        """Seconds since the last datagram arrived (inf if none yet)."""
        t = self._t_snap
        return math.inf if t == 0.0 else time.time() - t

    def close(self) -> None:
        self._running = False
        self._sock.close()
