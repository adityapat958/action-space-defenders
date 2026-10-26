"""frame_check.py — Frame-verification flight policy (team4).

Sequence (per drone):
  1. Hold 1.0 m altitude above takeoff point.
  2. Fly to WP1 = 1.0 m in front of tag 10 (map [0, 0, 1]) at hold height.
  3. Yaw 180° (face into the room, away from tag 10).
  4. Fly to WP2 = 1.0 m off tag 6 along its outward normal, at hold height.
  5. Hover 3 s.
  6. Return None → land.

Every tick prints: map-frame xyz, sim xyz, yaw, current WP, distance.

Dry-run (no --fly): prints poses only so you can walk the drone by hand and
verify that the printed map xyz matches the real tag positions.

Uses AttackFeed for own_state (identical telemetry path as defend policies).
P-control in sim frame; world_to_rc for heading; rc cap 40, yaw cap 40; arrive ≤ 0.25 m.
Pose older than 1 s → hover in place.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# -- path setup: drone_deployment_infra/ must be on sys.path
_DD = Path(__file__).resolve().parents[2]   # drone_deployment_infra/
if str(_DD) not in sys.path:
    sys.path.insert(0, str(_DD))

from defend_live import Frame, load_map, world_to_rc   # noqa: E402
from swarm_ops.policies.attack_feed import AttackFeed, _make_frame, _ORIGIN_HEIGHT  # noqa: E402
from swarm_ops.policies.base import Policy, hold_height, clip  # noqa: E402

# ---- map path ----
_REPO = Path(__file__).resolve().parents[4]
_MAP_PATH = _REPO / "drone_deployment_infra" / "config" / "map.yaml"

# ---- flight constants ----
HOLD_M = 1.0          # hold altitude [m] in sim frame (sim z)
ARRIVE_M = 0.25       # waypoint arrival radius [m]
RC_CAP = 40           # max lateral/fwd rc
YAW_CAP = 40          # max yaw rc
KP_POS = 30.0         # rc per metre of position error
KP_YAW = 15.0         # rc per radian of yaw error
HOVER_S = 3.0         # hover time at WP2
STALE_S = 1.0         # pose age threshold → hover

# ---- states ----
_S_CLIMB    = "climb"
_S_WP1      = "wp1"
_S_YAW180   = "yaw180"
_S_WP2      = "wp2"
_S_HOVER    = "hover"
_S_DONE     = "done"


def _load_waypoints(frame: Frame) -> Tuple[np.ndarray, np.ndarray]:
    """Return (WP1_sim, WP2_sim) from the map."""
    info = load_map(_MAP_PATH)
    tags = info["tags"]

    # WP1: 1 m in front of tag 10 (map frame [0, 0, 1]) at hold height
    wp1_map = np.array([0.0, 0.0, 1.0])
    wp1_sim = frame.map_to_sim(wp1_map)
    wp1_sim[2] = HOLD_M   # fix to hold altitude

    # WP2: 1 m off tag 6 along its outward normal, at hold height
    if 6 in tags:
        R6, t6 = tags[6]
        normal6_map = R6[:, 2]   # z-column = outward normal in map frame
        wp2_map = t6 + normal6_map
        wp2_sim = frame.map_to_sim(wp2_map)
        wp2_sim[2] = HOLD_M
    else:
        # fallback: just 2 m into the room
        wp2_sim = np.array([2.0, 0.0, HOLD_M])

    return wp1_sim, wp2_sim


class FrameCheckPolicy(Policy):
    """Frame-verification waypoint flight using own_state from AttackFeed telemetry."""
    name = "frame_check"

    def __init__(self, telem_port: int = 9100,
                 map_path: str | None = None,
                 fly: bool = True):
        self._port = telem_port
        self._map_path = map_path or str(_MAP_PATH)
        self._fly = fly
        self._feed: AttackFeed | None = None
        self._frame: Frame | None = None
        self._wp1: np.ndarray | None = None
        self._wp2: np.ndarray | None = None
        # per-drone state
        self._state: dict[str, str] = {}
        self._hover_start: dict[str, float] = {}

    def setup(self, drones):
        super().setup(drones)
        self._frame = _make_frame(self._map_path)
        self._wp1, self._wp2 = _load_waypoints(self._frame)
        print(f"[frame_check] WP1 sim={self._wp1}, WP2 sim={self._wp2}")
        if self._fly:
            self._feed = AttackFeed(port=self._port, map_path=self._map_path)
        for d in drones:
            self._state[d.name] = _S_CLIMB

    def teardown(self):
        if self._feed:
            self._feed.close()

    # ------------------------------------------------------------------
    def step(self, d, t: float):
        state = self._state.get(d.name, _S_DONE)
        if state == _S_DONE:
            return None   # land

        # ---- get pose ----
        own = None
        pose_age = math.inf
        if self._feed:
            own, pose_age = self._feed.own_state_with_age(d.name)

        pos_sim: np.ndarray | None = own.pos_sim if own else None
        yaw_sim: float | None = own.yaw_sim if own else None

        # ---- dry-run: print and hover ----
        if not self._fly:
            if pos_sim is not None:
                p_map = self._frame.sim_to_map(pos_sim)
                print(
                    f"[frame_check dry-run] {d.name}  "
                    f"map=({p_map[0]:+.3f},{p_map[1]:+.3f},{p_map[2]:+.3f})  "
                    f"sim=({pos_sim[0]:+.3f},{pos_sim[1]:+.3f},{pos_sim[2]:+.3f})  "
                    f"yaw={math.degrees(yaw_sim):.1f}°" if yaw_sim else "yaw=?"
                )
            else:
                print(f"[frame_check dry-run] {d.name}  no pose yet")
            return (0, 0, 0, 0)

        # ---- stale pose → hover ----
        if pose_age > STALE_S or pos_sim is None:
            _log(d.name, pos_sim, yaw_sim, state, math.inf, self._frame)
            h_rc = hold_height(d, int(HOLD_M * 100))
            return (0, 0, h_rc, 0)

        # ---- state machine ----
        h_rc = hold_height(d, int(HOLD_M * 100))
        rc_a, rc_b, rc_c, rc_d = 0, 0, h_rc, 0

        if state == _S_CLIMB:
            if abs(pos_sim[2] - HOLD_M) < 0.10:
                self._state[d.name] = _S_WP1
            _log(d.name, pos_sim, yaw_sim, state, abs(pos_sim[2] - HOLD_M), self._frame)
            return (0, 0, h_rc, 0)

        elif state == _S_WP1:
            dist, (ra, rb, rd) = _nav(pos_sim, yaw_sim, self._wp1)
            _log(d.name, pos_sim, yaw_sim, state, dist, self._frame)
            if dist < ARRIVE_M:
                self._state[d.name] = _S_YAW180
            return (ra, rb, h_rc, rd)

        elif state == _S_YAW180:
            # Target yaw: face +z into room = sim +x direction → yaw_target = 0 rad
            # After 180° turn from facing tag (yaw≈π) we want yaw≈0
            # But actually after WP1 the drone faces tag 10 (sim -x direction → yaw≈π)
            # We want it to face away → yaw ≈ 0
            target_yaw = 0.0
            err = _yaw_err(yaw_sim, target_yaw) if yaw_sim is not None else 0.0
            _log(d.name, pos_sim, yaw_sim, state, abs(err), self._frame)
            if abs(err) < math.radians(15):
                self._state[d.name] = _S_WP2
                return (0, 0, h_rc, 0)
            yaw_rc = clip(KP_YAW * (-err), YAW_CAP)  # negative err: turn CW (rc_d positive)
            return (0, 0, h_rc, yaw_rc)

        elif state == _S_WP2:
            dist, (ra, rb, rd) = _nav(pos_sim, yaw_sim, self._wp2)
            _log(d.name, pos_sim, yaw_sim, state, dist, self._frame)
            if dist < ARRIVE_M:
                self._state[d.name] = _S_HOVER
                self._hover_start[d.name] = time.time()
            return (ra, rb, h_rc, rd)

        elif state == _S_HOVER:
            elapsed = time.time() - self._hover_start.get(d.name, time.time())
            dist = float(np.linalg.norm(pos_sim - self._wp2))
            _log(d.name, pos_sim, yaw_sim, state, dist, self._frame)
            if elapsed >= HOVER_S:
                self._state[d.name] = _S_DONE
                return None
            return (0, 0, h_rc, 0)

        return (0, 0, h_rc, 0)

    def describe(self, d):
        return f"{self.name}:{self._state.get(d.name, '?')}"


# ---- helpers ----

def _nav(pos_sim: np.ndarray, yaw_sim: Optional[float],
         target_sim: np.ndarray) -> Tuple[float, Tuple[int, int, int]]:
    """P-control to target; returns (dist, (rc_a, rc_b, rc_d))."""
    err = target_sim - pos_sim
    err[2] = 0.0   # height handled by hold_height
    dist = float(np.linalg.norm(err))
    if dist < 0.05:
        return dist, (0, 0, 0)
    # world velocity direction (capped)
    vel = err / dist * min(dist, 1.0)   # unit vector * min(dist,1)
    action = np.array([vel[0], vel[1], 0.0, 0.0])
    yaw = yaw_sim if yaw_sim is not None else 0.0
    rc = world_to_rc(action, yaw)
    ra = clip(rc[0], RC_CAP)
    rb = clip(rc[1], RC_CAP)
    rd = clip(rc[3], YAW_CAP)
    return dist, (ra, rb, rd)


def _yaw_err(current: float, target: float) -> float:
    """Signed angular difference in [-π, π]."""
    err = target - current
    while err > math.pi:
        err -= 2 * math.pi
    while err < -math.pi:
        err += 2 * math.pi
    return err


def _log(name: str, pos_sim: Optional[np.ndarray], yaw_sim: Optional[float],
         state: str, dist: float, frame: Frame) -> None:
    if pos_sim is None:
        print(f"[frame_check] {name}  no_pose  state={state}")
        return
    p_map = frame.sim_to_map(pos_sim)
    yaw_deg = math.degrees(yaw_sim) if yaw_sim is not None else float("nan")
    print(
        f"[frame_check] {name}  "
        f"map=({p_map[0]:+.3f},{p_map[1]:+.3f},{p_map[2]:+.3f})  "
        f"sim=({pos_sim[0]:+.3f},{pos_sim[1]:+.3f},{pos_sim[2]:+.3f})  "
        f"yaw={yaw_deg:+.1f}°  wp={state}  dist={dist:.3f}m"
    )
