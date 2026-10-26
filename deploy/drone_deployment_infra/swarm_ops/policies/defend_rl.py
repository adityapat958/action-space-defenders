"""defend_rl.py — DefendRLPolicy: RL MLP (sparse_s2) as a swarm_ops Policy.

Pose sources (in priority order):
  1. PRIMARY:  06_localize.py via LocalizeFeed (UDP :9101 / file / stdin).
               Outputs origin/map-frame poses; map_to_sim converts directly.
  2. FALLBACK: drone_telemetry UDP feed (AttackFeed, port 9100) — tag-frame wire
               poses, full wire-flip + tag_to_map + map_to_sim pipeline.
  3. TERTIARY: swarm bus formation/drone<N> (swarm_formation.py).

Obs builder, safety helpers, MLP forward: imported from defend_live.py.
RC always in [-100, 100]; height held by hold_height.

Pose-age / dead-reckoning rules (shared with defend_lead):
  age < STALE_S/2      → full rc (1.0 scale)
  STALE_S/2 .. STALE_S → linearly drops to 0.5
  >= STALE_S           → HOVER (rc 0,0,z,0)
  STALE_S = DEFEND_POSE_TIMEOUT_S env var (default 5.0 s).
  Max RC: DEFEND_MAX_RC (default 80).

Logging: set DEFEND_LOG_DIR (default team4/logs/) to capture per-tick JSONL.
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

_DI = Path(__file__).resolve().parents[2]
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_DI))

from defend_live import (  # noqa: E402
    SafetyCfg, Safety, MLPPolicy as _MLP, build_obs, world_to_rc,
)
from .attack_feed import AttackFeed, DEFAULT_PORT, STALE_S
from .localize_feed import LocalizeFeed, LOCALIZE_UDP_PORT, LOCALIZE_TIMEOUT_S
from .base import Policy, hold_height, HIGH_CM, clip
from .defend_log import DefendLogger

_DEFAULT_WEIGHTS = _REPO / "policies" / "defend_rl_sparse_s2" / "policy.npz"
_FORMATION_PREFIX = "formation/"
_ATK_SLOTS = 4
_SITE = np.array([3.0, 0.0, 1.25])
_MAX_RC = int(os.environ.get("DEFEND_MAX_RC", 80))
_MAX_YAW_RC = int(os.environ.get("DEFEND_MAX_YAW_RC", "50"))

# --- height controller (overrides base.hold_height) ---
# P-gain and deadband same as base.py (KP_HEIGHT=0.6, DEADBAND_CM=4).
# DEFEND_MAX_RC_Z=60 gets from 80 cm takeoff to 180 cm in ~2 s (vs ~4-5 s at 30).
_DEFEND_HEIGHT_CM = int(os.environ.get("DEFEND_HEIGHT_CM", 180))
_DEFEND_MAX_RC_Z  = int(os.environ.get("DEFEND_MAX_RC_Z", 60))
_KP_Z = 0.6
_DEADBAND_CM = 4


def _hold_height(d, target_cm: int = _DEFEND_HEIGHT_CM) -> int:
    """P-controller toward target_cm; cap at DEFEND_MAX_RC_Z (default 60)."""
    h = d.height_cm()
    if h is None or abs(target_cm - h) <= _DEADBAND_CM:
        return 0
    return int(max(-_DEFEND_MAX_RC_Z, min(_DEFEND_MAX_RC_Z, round(_KP_Z * (target_cm - h)))))


def pose_age_rc_scale(own_age: float) -> float:
    if own_age >= STALE_S:
        return 0.0
    half = STALE_S / 2.0
    if own_age <= half:
        return 1.0
    return 1.0 - 0.5 * (own_age - half) / half


def pose_age_atk_scale(atk_age: float) -> float:
    return pose_age_rc_scale(atk_age)


def _attackers_from_bus(bus, own_drone_nums):
    att_pos = np.zeros((_ATK_SLOTS, 3))
    att_vel = np.zeros((_ATK_SLOTS, 3))
    att_alive = np.zeros(_ATK_SLOTS, bool)
    if bus is None:
        return att_pos, att_vel, att_alive
    entries = bus.since(_FORMATION_PREFIX, max_age=STALE_S)
    slot = 0
    for topic, (data, _age) in entries.items():
        num_str = topic[len(_FORMATION_PREFIX):]
        try:
            num = int(num_str.replace("drone", ""))
        except ValueError:
            continue
        if num in own_drone_nums:
            continue
        if slot >= _ATK_SLOTS:
            break
        pose = data.get("pose") if isinstance(data, dict) else None
        vel = data.get("vel", [0, 0, 0]) if isinstance(data, dict) else [0, 0, 0]
        if pose is not None and len(pose) == 3:
            att_pos[slot] = pose
            att_vel[slot] = vel[:3]
            att_alive[slot] = True
            slot += 1
    return att_pos, att_vel, att_alive


class DefendRLPolicy(Policy):
    """RL MLP defender.  06_localize primary, drone_telemetry fallback."""
    name = "defend_rl"

    def __init__(self, weights=None, target_cm=None, telem_port=None, map_path=None):
        self.weights = weights or os.environ.get("DEFEND_RL_WEIGHTS") or str(_DEFAULT_WEIGHTS)
        self.target_cm = target_cm or _DEFEND_HEIGHT_CM
        self._telem_port = telem_port or int(os.environ.get("DEFEND_TELEM_PORT", DEFAULT_PORT))
        self._map_path = map_path
        self._mlp: _MLP | None = None
        self._safety = Safety(SafetyCfg(stale_hover_s=STALE_S, stale_land_s=1e9, link_timeout_s=1e9))
        self._prev_target = None
        self._loc_feed: LocalizeFeed | None = None
        self._feed: AttackFeed | None = None
        self._own_yaw = 0.0
        self._own_nums: list[int] = []
        self._logger: DefendLogger | None = None
        self._tick = 0

    def setup(self, drones):
        super().setup(drones)
        self._mlp = _MLP(self.weights)
        self._own_nums = [d.num for d in drones]
        self._cache, self._consumed = None, set()
        self._prev_att, self._prev_own, self._vel_own, self._why = None, {}, {}, {}
        self._prev_target = None
        self._own_name = drones[0].name if drones else "drone1"

        # Primary: 06_localize feed
        try:
            self._loc_feed = LocalizeFeed(map_path=self._map_path)
        except Exception as exc:
            print(f"[defend_rl] localize_feed init error: {exc}", flush=True)
            self._loc_feed = None

        # Fallback: drone_telemetry UDP
        try:
            self._feed = AttackFeed(port=self._telem_port, map_path=self._map_path)
        except OSError as exc:
            print(f"[defend_rl] WARNING: UDP {self._telem_port}: {exc}; bus-only", flush=True)
            self._feed = None

        # Logger
        self._logger = DefendLogger(
            policy_name=self.name,
            weights_path=self.weights,
        )

    def _get_own_pose(self, d):
        """Returns (pos_sim, yaw_sim, age, source_str). pos_sim=None → HOVER."""
        h_cm = d.height_cm()
        own_z = (h_cm / 100.0) if h_cm is not None else (self.target_cm / 100.0)

        # --- 1. 06_localize (primary) ---
        if self._loc_feed is not None:
            pose, age = self._loc_feed.own_state_with_age(d.name)
            if pose is not None and age < LOCALIZE_TIMEOUT_S:
                p = pose.pos_sim.copy(); p[2] = own_z
                return p, pose.yaw_sim, age, "localize"

        # --- 2. drone_telemetry UDP (fallback) ---
        if self._feed is not None:
            s, age = self._feed.own_state_with_age(d.name)
            if s is not None and s.pos_sim is not None:
                p = s.pos_sim.copy(); p[2] = own_z
                yaw = s.yaw_sim if s.yaw_sim is not None else self._own_yaw
                return p, yaw, age, "telem_udp"

        return None, self._own_yaw, math.inf, "none"

    def _get_attacker_state(self):
        """Returns (att_pos, att_vel, att_alive, atk_age, sources).

        Primary: 06_localize feed; fallback: UDP; tertiary: bus.
        """
        att_pos = np.zeros((_ATK_SLOTS, 3))
        att_vel = np.zeros((_ATK_SLOTS, 3))
        att_alive = np.zeros(_ATK_SLOTS, bool)
        atk_age = 0.0
        sources = []
        slot = 0

        # Primary: 06_localize
        if self._loc_feed is not None:
            for p in self._loc_feed.latest_attackers(own_name=self._own_name,
                                                     max_age=LOCALIZE_TIMEOUT_S):
                if slot >= _ATK_SLOTS:
                    break
                att_pos[slot] = p.pos_sim
                att_alive[slot] = True
                atk_age = max(atk_age, p.age)
                sources.append("localize")
                slot += 1

        # Fallback: drone_telemetry UDP
        if slot == 0 and self._feed is not None:
            for s, age in self._feed.latest_attackers_with_age():
                if slot >= _ATK_SLOTS:
                    break
                att_pos[slot] = s.pos_sim
                att_alive[slot] = True
                atk_age = max(atk_age, age)
                sources.append("telem_udp")
                slot += 1

        # Tertiary: bus
        if slot == 0:
            att_pos, att_vel, att_alive = _attackers_from_bus(self.bus, self._own_nums)
            if any(att_alive):
                sources.append("bus")

        return att_pos, att_vel, att_alive, atk_age, sources

    # ------------------------------------------------------------------ joint team step
    # The runner calls step(d) once per drone per loop. We compute ALL defenders together once per
    # loop (like training: each defender sees its 2 teammates), run the per-agent MLP on (D, 53),
    # apply defend_live's Safety (0.5 m hard / 0.8 m soft separation, geofence), cache per drone.
    def _team_tick(self, t):
        drones = sorted(self.drones, key=lambda x: x.num)
        D = len(drones)
        now = time.time()
        att_pos, att_vel, att_alive, atk_age, atk_src = self._get_attacker_state()
        # attacker velocity: finite difference per slot (feed has no map-frame vel)
        if self._prev_att is not None and now > self._prev_att[0]:
            dt = now - self._prev_att[0]
            for k in range(_ATK_SLOTS):
                if att_alive[k] and self._prev_att[2][k] and not att_vel[k].any():
                    v = (att_pos[k] - self._prev_att[1][k]) / dt
                    if np.linalg.norm(v) < 3.0:
                        att_vel[k] = v
        self._prev_att = (now, att_pos.copy(), att_alive.copy())

        own_pos = np.zeros((D, 3)); own_vel = np.zeros((D, 3)); own_yaw = np.zeros(D)
        ages = np.full(D, math.inf); alive = np.zeros(D, bool); srcs = []
        for i, dd in enumerate(drones):
            p, yaw, age, src = self._get_own_pose(dd)
            srcs.append(src)
            if p is None:
                continue
            own_pos[i], own_yaw[i], ages[i] = p, yaw, age
            alive[i] = age < STALE_S
            prev = self._prev_own.get(dd.num)
            if prev is not None and now > prev[0]:
                v = (p - prev[1]) / (now - prev[0])
                v_old = self._vel_own.get(dd.num, np.zeros(3))
                v = 0.5 * v + 0.5 * v_old if np.linalg.norm(v) < 3.0 else v_old
                self._vel_own[dd.num] = v
            self._prev_own[dd.num] = (now, p.copy())
            own_vel[i] = self._vel_own.get(dd.num, np.zeros(3))

        prev_t = self._prev_target if (self._prev_target is not None and len(self._prev_target) == D) else None
        obs, target, _ttg, _aim = build_obs(own_pos, own_vel, own_yaw, alive,
                                            att_pos, att_vel, att_alive, _SITE, prev_target=prev_t)
        self._prev_target = target
        actions = np.asarray(self._mlp.act({"obs": obs}), float)
        raw = actions.copy()
        cmds, why = self._safety.apply(actions, own_pos, np.where(alive, ages, 1e9),
                                       link_age=np.zeros(D), battery=[None] * D,
                                       flying=[True] * D)
        out = {}
        for i, dd in enumerate(drones):
            rc_z = _hold_height(dd, self.target_cm)
            kind, a = cmds[i]
            if not alive[i] or kind != "rc":
                rc = (0, 0, clip(rc_z, 100), 0); stale = True; capped = False
            else:
                rc_a, rc_b, _c, rc_d = world_to_rc(a, own_yaw[i])
                scale = min(pose_age_rc_scale(ages[i]),
                            pose_age_atk_scale(atk_age) if any(att_alive) else 1.0)
                lim = int(_MAX_RC * scale); capped = lim < _MAX_RC
                rc = (clip(rc_a, lim), clip(rc_b, lim), clip(rc_z, 100), clip(rc_d, _MAX_YAW_RC))
                stale = False
            out[dd.num] = rc
            if self._logger:
                self._logger.log_tick(
                    t=t, drone=dd.name,
                    own_pos=own_pos[i] if alive[i] else None, own_age=ages[i], own_src=srcs[i],
                    attackers=[(att_pos[k], att_vel[k], atk_age) for k in range(_ATK_SLOTS) if att_alive[k]],
                    obs=obs[i], action=raw[i], rc=rc, hover_stale=stale, rc_capped=capped,
                )
        self._why = dict(zip([dd.num for dd in drones], why))
        return out

    def step(self, d, t):
        self._tick += 1
        # new loop iteration = this drone already consumed the cached command
        if self._cache is None or d.num in self._consumed:
            self._cache = self._team_tick(t)
            self._consumed = set()
        self._consumed.add(d.num)
        return self._cache.get(d.num, (0, 0, clip(_hold_height(d, self.target_cm), 100), 0))

    def describe(self, d):
        n = 0
        if self._loc_feed:
            n = len(self._loc_feed.latest_attackers(own_name=self._own_name))
        if n == 0 and self._feed:
            n = len(self._feed.latest_attackers())
        return f"{self.name}({n} atk)"

    def teardown(self):
        if self._feed:
            self._feed.close()
        if self._loc_feed:
            self._loc_feed.close()
        if self._logger:
            self._logger.close()
