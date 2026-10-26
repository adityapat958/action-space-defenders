"""defend_lead.py — AssignLeadSafePolicy: classical assign-lead-safe as a swarm_ops Policy.

Pose sources (priority order):
  1. PRIMARY:  06_localize.py via LocalizeFeed (UDP :9101 / file / stdin).
  2. FALLBACK: drone_telemetry UDP feed (AttackFeed, port 9100).
  3. TERTIARY: swarm bus formation/drone<N>.

Pose-age / dead-reckoning rules (same as DefendRLPolicy):
  0 .. STALE_S/2 → full rc; STALE_S/2 .. STALE_S → 0.5× rc;
  >= STALE_S     → HOVER (rc 0,0,z,0).

Logging: DEFEND_LOG_DIR (default team4/logs/), one JSONL per run.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_DI = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_DI))

from defend_live import AssignLeadPolicy as _AssignLead, build_obs, world_to_rc  # noqa: E402
from .attack_feed import AttackFeed, DEFAULT_PORT, STALE_S
from .localize_feed import LocalizeFeed, LOCALIZE_UDP_PORT, LOCALIZE_TIMEOUT_S
from .base import Policy, hold_height, HIGH_CM, clip
from .defend_rl import (  # noqa: E402
    _attackers_from_bus, _SITE, _ATK_SLOTS, _MAX_RC,
    pose_age_rc_scale, pose_age_atk_scale,
    _hold_height, _DEFEND_HEIGHT_CM,
)
from .defend_log import DefendLogger

_SITE_ARR = _SITE


class AssignLeadSafePolicy(Policy):
    """Classical assign-lead-safe defender with 06_localize primary pose source."""
    name = "defend_lead"

    def __init__(self, target_cm=None, repel_radius=0.8, telem_port=None, map_path=None):
        self.target_cm = target_cm or _DEFEND_HEIGHT_CM
        self._lead = _AssignLead(repel_radius=repel_radius)
        self._prev_target = None
        self._own_yaw = 0.0
        self._telem_port = telem_port or DEFAULT_PORT
        self._map_path = map_path
        self._loc_feed: LocalizeFeed | None = None
        self._feed: AttackFeed | None = None
        self._own_nums: list[int] = []
        self._logger: DefendLogger | None = None

    def setup(self, drones):
        super().setup(drones)
        self._own_nums = [d.num for d in drones]
        self._own_name = drones[0].name if drones else "drone1"

        try:
            self._loc_feed = LocalizeFeed(map_path=self._map_path)
        except Exception as exc:
            print(f"[defend_lead] localize_feed error: {exc}", flush=True)

        try:
            self._feed = AttackFeed(port=self._telem_port, map_path=self._map_path)
        except OSError as exc:
            print(f"[defend_lead] UDP {self._telem_port}: {exc}; bus-only", flush=True)

        self._logger = DefendLogger(policy_name=self.name, weights_path="")

    def _get_own_pose(self, d):
        h_cm = d.height_cm()
        own_z = (h_cm / 100.0) if h_cm is not None else (self.target_cm / 100.0)

        if self._loc_feed is not None:
            pose, age = self._loc_feed.own_state_with_age(d.name)
            if pose is not None and age < LOCALIZE_TIMEOUT_S:
                p = pose.pos_sim.copy(); p[2] = own_z
                return p, pose.yaw_sim, age, "localize"

        if self._feed is not None:
            s, age = self._feed.own_state_with_age(d.name)
            if s is not None and s.pos_sim is not None:
                p = s.pos_sim.copy(); p[2] = own_z
                yaw = s.yaw_sim if s.yaw_sim is not None else self._own_yaw
                return p, yaw, age, "telem_udp"

        return None, self._own_yaw, math.inf, "none"

    def _get_attacker_state(self):
        att_pos = np.zeros((_ATK_SLOTS, 3))
        att_vel = np.zeros((_ATK_SLOTS, 3))
        att_alive = np.zeros(_ATK_SLOTS, bool)
        atk_age = 0.0; slot = 0

        if self._loc_feed is not None:
            for p in self._loc_feed.latest_attackers(own_name=self._own_name,
                                                     max_age=LOCALIZE_TIMEOUT_S):
                if slot >= _ATK_SLOTS: break
                att_pos[slot] = p.pos_sim; att_alive[slot] = True
                atk_age = max(atk_age, p.age); slot += 1

        if slot == 0 and self._feed is not None:
            for s, age in self._feed.latest_attackers_with_age():
                if slot >= _ATK_SLOTS: break
                att_pos[slot] = s.pos_sim; att_alive[slot] = True
                atk_age = max(atk_age, age); slot += 1

        if slot == 0:
            att_pos, att_vel, att_alive = _attackers_from_bus(self.bus, self._own_nums)

        return att_pos, att_vel, att_alive, atk_age

    def step(self, d, t):
        rc_z = _hold_height(d, self.target_cm)
        att_pos, att_vel, att_alive, atk_age = self._get_attacker_state()
        own_pos_arr, own_yaw, own_age, own_src = self._get_own_pose(d)

        hover_stale = own_pos_arr is None or own_age >= STALE_S
        if hover_stale:
            rc = (0, 0, clip(rc_z, 100), 0)
            if self._logger:
                self._logger.log_tick(
                    t=t, drone=d.name,
                    own_pos=own_pos_arr, own_age=own_age, own_src=own_src,
                    attackers=[(att_pos[i], att_vel[i], atk_age)
                                for i in range(_ATK_SLOTS) if att_alive[i]],
                    obs=np.zeros(53), action=np.zeros(4), rc=rc,
                    hover_stale=True, rc_capped=False,
                )
            return rc

        self._own_yaw = own_yaw
        own_pos = own_pos_arr.reshape(1, 3)
        own_vel = np.zeros((1, 3))
        own_yaw_arr = np.array([own_yaw])
        def_alive = np.array([True])

        obs, target, _ttg, aim = build_obs(
            own_pos, own_vel, own_yaw_arr, def_alive,
            att_pos, att_vel, att_alive, _SITE_ARR,
            prev_target=self._prev_target,
        )
        self._prev_target = target

        action = self._lead.act({
            "obs": obs, "own_pos": own_pos, "own_vel": own_vel,
            "def_alive": def_alive, "att_pos": att_pos, "att_vel": att_vel,
            "att_alive": att_alive, "site": _SITE_ARR, "aim": aim, "target": target,
        })
        a_vec = action[0]
        rc_a, rc_b, _rc_c, rc_d = world_to_rc(a_vec, own_yaw)

        own_scale = pose_age_rc_scale(own_age)
        atk_scale = pose_age_atk_scale(atk_age) if any(att_alive) else 1.0
        scale = min(own_scale, atk_scale)
        lim = int(_MAX_RC * scale)
        capped = lim < _MAX_RC

        rc = (clip(rc_a, lim), clip(rc_b, lim), clip(rc_z, 100), clip(rc_d, lim))

        if self._logger:
            self._logger.log_tick(
                t=t, drone=d.name,
                own_pos=own_pos_arr, own_age=own_age, own_src=own_src,
                attackers=[(att_pos[i], att_vel[i], atk_age)
                            for i in range(_ATK_SLOTS) if att_alive[i]],
                obs=obs[0] if obs.ndim == 2 else obs,
                action=np.array([rc_a, rc_b, _rc_c, rc_d]),
                rc=rc, hover_stale=False, rc_capped=capped,
            )

        return rc

    def describe(self, d):
        return self.name

    def teardown(self):
        if self._feed:
            self._feed.close()
        if self._loc_feed:
            self._loc_feed.close()
        if self._logger:
            self._logger.close()
