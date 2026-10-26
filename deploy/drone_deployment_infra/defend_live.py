#!/usr/bin/env python3
"""defend_live.py -- 3 Tello defenders vs N attackers on the organizer host (single file, numpy only).

Loop at 13.5 Hz (the sim's 500/37 policy rate):

    feed (attackers + our own poses, UDP JSON)  ->  map frame  ->  sim frame (z up)
      -> KalmanCV per own drone, alpha-beta per attacker (same filters as AS-Defend-Live-v0)
      -> obs (53 per defender, exactly the AS-Defend-Live-v0 layout)
      -> policy: assign-lead (scripted, default) | rl (numpy MLP from .npz) | hover
      -> safety layer (geofence, 0.5 m separation, stale pose, link loss, battery, kill)
      -> world-frame velocity -> heading-frame `rc a b c d` per drone over its USB dongle

Modes
    --dry-run                 no drone is ever contacted; rc is printed/logged. Feed: --feed synthetic
                              (built-in kinematic world) or a live UDP feed (watch the GUI drones).
    --replay FILE             feed a recording (JSONL, see Recorder) through the pipeline on a virtual
                              clock; prints feed latency/staleness + a frame sanity check. No drones.
    --live --i-know-this-flies   the only path that opens Tello sockets and takes off.

Keyboard kill (live and dry-run): type k / q / l + Enter, or Ctrl-C  ->  land all, exit.

Frames (see FRAME NOTES below): the tag map (config/map.yaml) is the origin tag's AprilTag frame.
pupil-apriltags poses (and drone_telemetry's README) use x right, y DOWN, z INTO the wall, while the
map.yaml `frame:` string says y up, z out. Both are supported (--map-convention); the default
`ydown_zin` follows the code that built the map. `--replay` prints a check (height vs Tello `h`) that
says which one the data supports. Sim frame: z up, +x away from the wall into the room, +y = map +x.

Policy weights (.npz, --policy rl --weights FILE):
    w0, b0, w1, b1, ..., wK, bK      Linear layers, w_i shape (in, out) (torch weights transposed)
    activation  (optional) 'elu' (skrl default) | 'tanh' | 'relu'
    obs_mean, obs_var (optional)      running standardizer: clip((x - mean) / sqrt(var + 1e-8), -5, 5)
    obs_std (optional, wins over obs_var)  skrl RunningStandardScaler exactly: clip((x - mean) / obs_std, -c, c),
                                      obs_std = sqrt(running_variance) + 1e-8, c = obs_clip (default 5)
    Per-defender policies (skrl MAPPO/IPPO keep one net per agent): every w/b/obs_* array gets a leading
    axis of size 3 (defender i uses row i); 2-D w = one shared net. Written by export_policy.py.
    The last layer's output is the action mean, clipped to [-1, 1]: world-frame [vx, vy, vz, yaw_rate].
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_MAP = HERE / "config" / "map.yaml"
DEFAULT_DONGLES = HERE / "config" / "dongles.json"

# --- AS-Defend-Live-v0 constants (tello_defend_env_cfg.TelloDefendLiveEnvCfg) ------------------
DT = 37.0 / 500.0  # [s] policy period (13.5 Hz)
NUM_DEFENDERS = 3
NUM_SLOTS = 4  # attacker slots in the obs
OBS_DIM = 9 + 2 + 7 * (NUM_DEFENDERS - 1) + 7 * NUM_SLOTS  # 53
MAX_LIN_SPEED = 1.0  # [m/s] TelloCommandCfg.max_lin_speed (rc 100)
MAX_YAW_RATE = 1.745  # [rad/s] TelloCommandCfg.max_yaw_rate
OBS_CLIP_POS = 12.0
OBS_CLIP_VEL = 3.0
SITE_HEIGHT = 2.5  # site centre is at z = SITE_HEIGHT / 2 in sim
ASSIGN_HYSTERESIS = 0.3  # [s]
FEED_ALPHA, FEED_BETA = 0.5, 0.2  # attacker alpha-beta filter (feed.py)
DEAD_TTG, TTG_CAP = 1.0e4, 100.0
GRAVITY_B = np.array([0.0, 0.0, -1.0])  # projected gravity of a level drone


# =============================================================================================
# Frames
# =============================================================================================
# Rows map a map-frame vector to sim (z up, +x into the room). Both are proper rotations.
CONVENTIONS = {
    # x right, y down, z into the wall (pupil-apriltags / drone_telemetry README; what built map.yaml)
    "ydown_zin": np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
    # x right, y up, z out of the wall (the map.yaml `frame:` string)
    "yup_zout": np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
}


def _parse_scalar(text: str):
    t = text.strip().strip("'\"")
    for cast in (int, float):
        try:
            return cast(t)
        except ValueError:
            pass
    return {"null": None, "true": True, "false": False}.get(t, t)


def mini_yaml(text: str):
    """Parse the YAML subset map.yaml uses (nested mappings, block lists incl. `- - x`), no PyYAML needed."""
    lines = []
    for raw in text.splitlines():
        s = raw.split(" #")[0].rstrip() if not raw.lstrip().startswith("#") else ""
        if s.strip():
            lines.append(s)
    pos = 0

    def indent(s):
        return len(s) - len(s.lstrip(" "))

    def block(ind):
        nonlocal pos
        if lines[pos].strip().startswith("- ") or lines[pos].strip() == "-":
            out = []
            while pos < len(lines) and indent(lines[pos]) == ind and lines[pos].strip().startswith("-"):
                rest = lines[pos].strip()[1:].strip()
                if rest.startswith("- ") or rest == "-":  # nested list on the same line: "- - 1.0"
                    lines[pos] = " " * (ind + 2) + rest
                    out.append(block(ind + 2))
                elif rest:
                    out.append(_parse_scalar(rest))
                    pos += 1
                else:
                    pos += 1
                    out.append(block(indent(lines[pos])))
            return out
        out = {}
        while pos < len(lines) and indent(lines[pos]) == ind:
            key, _, val = lines[pos].strip().partition(":")
            key = _parse_scalar(key)
            pos += 1
            if val.strip():
                out[key] = _parse_scalar(val)
            elif pos < len(lines) and indent(lines[pos]) > ind:
                out[key] = block(indent(lines[pos]))
            elif pos < len(lines) and indent(lines[pos]) == ind and lines[pos].strip().startswith("-"):
                out[key] = block(ind)
            else:
                out[key] = None
        return out

    return block(indent(lines[0])) if lines else {}


def load_map(path) -> dict:
    """{tag_id: (R 3x3, t 3)} plus optional 'geofence' / 'site' from map.yaml."""
    text = Path(path).read_text()
    try:
        import yaml  # optional

        data = yaml.safe_load(text)
    except ImportError:
        data = mini_yaml(text)
    tags = {int(k): (np.asarray(v["R"], float), np.asarray(v["t"], float)) for k, v in (data.get("tags") or {}).items()}
    return {"tags": tags, "geofence": data.get("geofence"), "site": data.get("site"),
            "origin_tag": data.get("origin_tag")}


class Frame:
    """Tag-frame / map-frame poses -> sim frame (z up). `origin_height` = height of the origin tag's centre
    above the floor [m] (sim z = 0 is the floor, as in the env)."""

    def __init__(self, convention: str = "ydown_zin", origin_height: float = 1.5, tags: dict | None = None):
        if convention not in CONVENTIONS:
            raise ValueError(f"unknown map convention {convention!r}; one of {sorted(CONVENTIONS)}")
        self.convention = convention
        self.R = CONVENTIONS[convention]
        self.offset = np.array([0.0, 0.0, origin_height])
        self.tags = tags if tags is not None else {}

    def map_to_sim(self, p_map) -> np.ndarray:
        return self.R @ np.asarray(p_map, float) + self.offset

    def sim_to_map(self, p_sim) -> np.ndarray:
        return self.R.T @ (np.asarray(p_sim, float) - self.offset)

    def tag_to_map(self, tag: int, p_tag):
        """Camera position in tag `tag`'s frame -> map frame; None if the tag is not mapped."""
        if tag not in self.tags:
            return None
        R, t = self.tags[tag]
        return R @ np.asarray(p_tag, float) + t

    def yaw_from_tag(self, tag: int, yaw_deg: float):
        """drone_telemetry yaw (deg, 0 = facing the tag, + = turned toward the tag's +x) -> sim yaw (rad, CCW)."""
        if tag not in self.tags:
            return None
        psi = math.radians(yaw_deg)
        f_tag = np.array([math.sin(psi), 0.0, math.cos(psi)])  # camera forward, tag frame (z into the tag)
        f_sim = self.R @ (self.tags[tag][0] @ f_tag)
        return math.atan2(f_sim[1], f_sim[0])


# =============================================================================================
# Filters (verbatim numpy ports of core/positional_kalman_filter.KalmanCV and feed.alpha_beta_step)
# =============================================================================================
@dataclass
class KalmanCfg:
    # 0.03 m = the env's feed noise; 8 m/s^2 lets the estimate follow a Tello's 4 m/s^2 steps (obs parity: own-vel
    # p95 error vs truth 0.10 m/s, was 0.8 m/s at 0.15/1.5, which cost the RL policy ~15% downed). --kalman to change.
    accel_sigma: float = 8.0
    meas_sigma: float = 0.03
    init_vel_sigma: float = 0.5
    gate_chi2: float = 16.27
    max_rejects: int = 3
    lost_after: float = 0.4


class KalmanCV:
    """Constant-velocity Kalman filter per axis, batched over the leading shape (numpy)."""

    def __init__(self, cfg: KalmanCfg, batch: tuple = ()):
        self.cfg = cfg
        z = np.zeros((*batch, 3))
        self.p, self.v, self.a, self.b, self.c = z.copy(), z.copy(), z.copy(), z.copy(), z.copy()
        self.ready = np.zeros(batch, bool)
        self.rejects = np.zeros(batch)

    def reset(self, mask) -> None:
        m3 = mask[..., None]
        self.p, self.v = np.where(m3, 0.0, self.p), np.where(m3, 0.0, self.v)
        self.ready = self.ready & ~mask
        self.rejects = np.where(mask, 0.0, self.rejects)

    def predict(self, dt: float, active=None) -> None:
        q = self.cfg.accel_sigma ** 2
        p = self.p + self.v * dt
        a = self.a + 2 * self.b * dt + self.c * dt ** 2 + q * dt ** 4 / 4
        b = self.b + self.c * dt + q * dt ** 3 / 2
        c = self.c + q * dt ** 2
        if active is None:
            self.p, self.a, self.b, self.c = p, a, b, c
            return
        m3 = (active & self.ready)[..., None]
        self.p, self.a, self.b, self.c = (np.where(m3, p, self.p), np.where(m3, a, self.a),
                                          np.where(m3, b, self.b), np.where(m3, c, self.c))

    def freeze(self, mask) -> None:
        self.v = np.where(mask[..., None], 0.0, self.v)

    def update(self, z, mask=None):
        cfg = self.cfg
        if mask is None:
            mask = np.ones(self.ready.shape, bool)
        r = cfg.meas_sigma ** 2
        y = z - self.p
        S = self.a + r
        d2 = (y * y / S).sum(-1)
        ok = mask & self.ready & (d2 <= cfg.gate_chi2)
        rejected = mask & self.ready & ~ok
        self.rejects = np.where(rejected, self.rejects + 1, np.where(ok, 0.0, self.rejects))
        ka, kb = self.a / S, self.b / S
        p, v = self.p + ka * y, self.v + kb * y
        a, b, c = self.a - ka * self.a, self.b - ka * self.b, self.c - kb * self.b
        start = mask & (~self.ready | (self.rejects >= cfg.max_rejects))
        s3, o3 = start[..., None], ok[..., None]
        self.p = np.where(s3, z, np.where(o3, p, self.p))
        self.v = np.where(s3, 0.0, np.where(o3, v, self.v))
        self.a = np.where(s3, r, np.where(o3, a, self.a))
        self.b = np.where(s3, 0.0, np.where(o3, b, self.b))
        self.c = np.where(s3, cfg.init_vel_sigma ** 2, np.where(o3, c, self.c))
        self.rejects = np.where(start, 0.0, self.rejects)
        self.ready = self.ready | start
        return ok | start


def alpha_beta_step(est_pos, est_vel, meas, measured, dt: float, alpha: float, beta: float):
    """numpy copy of tello_defend/feed.alpha_beta_step."""
    pred = est_pos + est_vel * dt
    resid = (meas - pred) * np.asarray(measured, float)[..., None]
    return pred + alpha * resid, est_vel + (beta / dt) * resid


# =============================================================================================
# Assignment + observation (numpy ports of tello_defend/assignment.py and _get_observations)
# =============================================================================================
def intercept_time(rel, v_att, speed: float):
    a = (v_att * v_att).sum(-1) - speed ** 2
    b = 2.0 * (rel * v_att).sum(-1)
    c = (rel * rel).sum(-1)
    direct = np.sqrt(c) / speed
    disc = b * b - 4.0 * a * c
    sq = np.sqrt(np.clip(disc, 0.0, None))
    safe_a = np.where(np.abs(a) < 1e-6, -1e-6, a)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-b - sq) / (2.0 * safe_a), (-b + sq) / (2.0 * safe_a)
    t = np.minimum(np.where(t1 > 0, t1, np.inf), np.where(t2 > 0, t2, np.inf))
    ok = (disc >= 0) & np.isfinite(t)
    t = np.where(ok, t, direct)
    aim = np.where(ok[..., None], rel + v_att * t[..., None], rel)
    return t, aim


_COMBOS: dict = {}


def all_assignments(D: int, A: int) -> np.ndarray:
    if (D, A) not in _COMBOS:
        _COMBOS[(D, A)] = np.array(list(itertools.product(range(A), repeat=D)), dtype=np.int64).reshape(-1, D)
    return _COMBOS[(D, A)]


def assignment_costs(ttg, alive, def_alive=None):
    """minmax_ttg cost of every joint assignment for ONE scene. ttg (D, A); alive (A,); def_alive (D,)."""
    D, A = ttg.shape
    ttg = np.clip(np.nan_to_num(ttg, nan=TTG_CAP, posinf=TTG_CAP), None, TTG_CAP)
    ttg = np.where(alive[None, :], ttg, DEAD_TTG)
    if def_alive is not None:
        ttg = np.where(def_alive[:, None], ttg, 0.0)
    combos = all_assignments(D, A)
    cost_per = ttg[np.arange(D)[None, :], combos]  # (K, D)
    base = cost_per.max(-1) + 0.01 * cost_per.sum(-1)
    onehot = np.zeros((len(combos), D, A))
    onehot[np.arange(len(combos))[:, None], np.arange(D)[None, :], combos] = 1.0
    if def_alive is not None:
        onehot = onehot * def_alive[None, :, None]
        need = min(int(alive.sum()), int(def_alive.sum()))
    else:
        need = min(int(alive.sum()), D)
    covered = (onehot.max(1) * alive[None, :]).sum(-1)
    return combos, base + 1.0e3 * np.clip(need - covered, 0, None)


def assign_targets(ttg, alive, previous=None, hysteresis=0.0, def_alive=None) -> np.ndarray:
    D, A = ttg.shape
    combos, cost = assignment_costs(ttg, alive, def_alive)
    best = int(cost.argmin())
    target = combos[best].copy()
    if previous is not None and hysteresis > 0.0:
        prev = np.asarray(previous, np.int64)
        prev_idx = int((prev * (A ** np.arange(D - 1, -1, -1))).sum())
        if alive[prev].all() and cost[prev_idx] - cost[best] < hysteresis:
            target = prev.copy()
    return target


def slot_order(ttg, alive, target) -> np.ndarray:
    """(D, A): slot -> attacker index; slot 0 = assigned, then live by ttg, dead last."""
    ttg = np.clip(np.nan_to_num(ttg, nan=TTG_CAP, posinf=TTG_CAP), None, TTG_CAP)
    key = np.where(alive[None, :], ttg, DEAD_TTG) + DEAD_TTG * (~alive)[None, :]
    key = key.copy()
    key[np.arange(len(target)), target] = -1.0
    return np.argsort(key, axis=-1, kind="stable")


def build_obs(own_pos, own_vel, own_yaw, def_alive, att_pos, att_vel, att_alive, site, prev_target=None,
              hysteresis=ASSIGN_HYSTERESIS):
    """AS-Defend-Live-v0 observation for every defender.

    own_* (D, 3) / (D,), def_alive (D,) bool; att_* (A=4, 3), att_alive (A,) bool; site (3,), all sim frame.
    Returns obs (D, 53), target (D,), ttg (D, A), aim (D, A, 3).
    """
    D = own_pos.shape[0]
    rel_all = att_pos[None, :, :] - own_pos[:, None, :]  # (D, A, 3)
    ttg, aim = intercept_time(rel_all, np.broadcast_to(att_vel[None], rel_all.shape), MAX_LIN_SPEED)
    target = assign_targets(ttg, att_alive, prev_target, hysteresis, def_alive=def_alive)
    order = slot_order(ttg, att_alive, target)
    valid_all = att_alive.astype(float)
    obs = np.zeros((D, OBS_DIM))
    for i in range(D):
        own = own_pos[i]
        idx = order[i]
        valid = valid_all[idx][:, None]
        rel_pos = np.clip((att_pos - own)[idx], -OBS_CLIP_POS, OBS_CLIP_POS) * valid
        rel_vel = np.clip((att_vel - own_vel[i])[idx], -OBS_CLIP_VEL, OBS_CLIP_VEL) * valid
        parts = [own - site, own_vel[i], GRAVITY_B, [math.sin(own_yaw[i]), math.cos(own_yaw[i])]]
        for j in range(D):
            if j == i:
                continue
            m = float(def_alive[j])
            parts += [np.clip(own_pos[j] - own, -OBS_CLIP_POS, OBS_CLIP_POS) * m,
                      np.clip(own_vel[j] - own_vel[i], -OBS_CLIP_VEL, OBS_CLIP_VEL) * m, [m]]
        # Pad missing teammate slots with zeros so obs dim is always OBS_DIM (= 53) regardless of D.
        # Trained policy saw NUM_DEFENDERS-1 = 2 teammate entries; fill absent ones with zeros.
        for _ in range(NUM_DEFENDERS - 1 - (D - 1)):
            parts += [np.zeros(3), np.zeros(3), [0.0]]
        parts += [rel_pos.reshape(-1), rel_vel.reshape(-1), valid[:, 0]]
        obs[i] = np.concatenate([np.asarray(p, float).reshape(-1) for p in parts])
    return obs, target, ttg, aim


# =============================================================================================
# Policies -> world-frame actions (D, 4) in [-1, 1]: [vx, vy, vz, yaw_rate]
# =============================================================================================
class HoverPolicy:
    name = "hover"

    def act(self, ctx: dict) -> np.ndarray:
        return np.zeros((ctx["obs"].shape[0], 4))


class AssignLeadPolicy:
    """assign-lead-safe from scripts/evaluate.py: lead-intercept the assigned attacker at full speed, plus
    teammate repulsion inside 0.8 m. Uses the obs' assignment (with hysteresis) so obs and expert agree."""

    name = "assign-lead"

    def __init__(self, repel_radius: float = 0.8):
        self.repel_radius = repel_radius

    def act(self, ctx: dict) -> np.ndarray:
        own_pos, def_alive, aim, target = ctx["own_pos"], ctx["def_alive"], ctx["aim"], ctx["target"]
        D = own_pos.shape[0]
        out = np.zeros((D, 4))
        any_att = ctx["att_alive"].any()
        for i in range(D):
            if any_att:
                a = aim[i, target[i]]
                direction = a / max(np.linalg.norm(a), 1e-6)
            else:  # nothing to chase: return toward the site and hold
                to_site = ctx["site"] - own_pos[i]
                direction = to_site / max(np.linalg.norm(to_site), 1.0)
            for j in range(D):
                if j == i or not def_alive[j]:
                    continue
                away = own_pos[i] - own_pos[j]
                d = max(np.linalg.norm(away), 1e-3)
                w = max(0.0, 2.0 * (self.repel_radius - d) / self.repel_radius)
                direction = direction + w * away / d
            direction = direction / max(np.linalg.norm(direction), 1.0)
            out[i, :3] = direction
        return out


class MLPPolicy:
    """Shared-parameter MLP from an .npz (format in the module docstring); runs once per defender."""

    name = "rl"

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.layers = []
        k = 0
        while f"w{k}" in z:
            self.layers.append((np.asarray(z[f"w{k}"], float), np.asarray(z[f"b{k}"], float)))
            k += 1
        if not self.layers:
            raise ValueError(f"{path}: no w0/b0 arrays")
        self.per_agent = self.layers[0][0].ndim == 3  # (n_agents, in, out): one net per defender
        if self.layers[0][0].shape[-2] != OBS_DIM or self.layers[-1][0].shape[-1] != 4:
            raise ValueError(f"{path}: MLP maps {self.layers[0][0].shape[-2]} -> {self.layers[-1][0].shape[-1]}, "
                             f"want {OBS_DIM} -> 4")
        if self.per_agent and self.layers[0][0].shape[0] != NUM_DEFENDERS:
            raise ValueError(f"{path}: {self.layers[0][0].shape[0]} per-agent nets, want {NUM_DEFENDERS}")
        self.activation = str(z["activation"]) if "activation" in z else "elu"
        self.mean = np.asarray(z["obs_mean"], float) if "obs_mean" in z else None
        self.var = np.asarray(z["obs_var"], float) if "obs_var" in z else None
        self.std = np.asarray(z["obs_std"], float) if "obs_std" in z else None
        if self.mean is not None and self.std is None:
            self.std = np.sqrt(self.var + 1e-8)
        self.obs_clip = float(z["obs_clip"]) if "obs_clip" in z else 5.0

    def _act_fn(self, x):
        if self.activation == "elu":
            return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
        if self.activation == "tanh":
            return np.tanh(x)
        if self.activation == "relu":
            return np.maximum(x, 0)
        raise ValueError(f"unknown activation {self.activation}")

    def forward(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, float)
        D = x.shape[0]
        if self.mean is not None:
            mean = self.mean[:D] if self.per_agent and self.mean.ndim > 1 else self.mean
            std  = self.std[:D]  if self.per_agent and self.std.ndim  > 1 else self.std
            x = np.clip((x - mean) / std, -self.obs_clip, self.obs_clip)
        for k, (w, b) in enumerate(self.layers):
            if self.per_agent:
                # slice to D agents when running fewer defenders than trained (e.g. 1v1 test)
                x = np.einsum("ai,aio->ao", x, w[:D]) + b[:D]
            else:
                x = x @ w + b
            if k < len(self.layers) - 1:
                x = self._act_fn(x)
        return np.clip(x, -1.0, 1.0)

    def act(self, ctx: dict) -> np.ndarray:
        return self.forward(ctx["obs"])


# =============================================================================================
# World action -> Tello rc (heading frame), identical to the env's rotation + TelloDynamics.to_rc
# =============================================================================================
def world_to_rc(action, yaw: float) -> tuple:
    """action = world [vx, vy, vz, yaw_rate] in [-1, 1]; yaw = measured sim yaw (rad, CCW from +x).

    Heading frame (env._pre_physics_step): fwd = c vx + s vy, left = -s vx + c vy. rc a = right = -left,
    b = forward, c = up, d = yaw clockwise = -yaw_rate. Integers in [-100, 100].
    """
    a = np.clip(np.asarray(action, float), -1.0, 1.0)
    c, s = math.cos(yaw), math.sin(yaw)
    fwd, left = c * a[0] + s * a[1], -s * a[0] + c * a[1]
    rc = np.clip(np.round(np.array([-left, fwd, a[2], -a[3]]) * 100.0), -100, 100)
    return tuple(int(v) for v in rc)


# =============================================================================================
# Safety layer
# =============================================================================================
@dataclass
class SafetyCfg:
    geofence: tuple = (0.6, 6.0, -3.5, 3.5, 0.4, 2.3)  # sim xmin xmax ymin ymax zmin zmax [m]
    fence_margin: float = 0.5  # [m] start removing outward motion this far inside the fence
    min_sep: float = 0.5  # [m] hard separation: below this, back off
    soft_sep: float = 0.8  # [m] below this, the closing component toward a teammate is removed
    stale_hover_s: float = 0.5
    stale_land_s: float = 2.0
    link_timeout_s: float = 1.5
    battery_floor: int = 15
    # cap on the commanded world xyz velocity NORM (fraction of rc 100). The env only clamps per axis, so its
    # action space allows norms up to sqrt(3); a 1.0 norm cap cost the RL exports ~10% downed/ep in the closed-loop
    # check (sparse_s0 S0 5.98 -> 6.62, sparse_s1 S1 7.22 -> 7.69, dense_s0 S0 6.56 -> 7.03; scripted unaffected,
    # its directions are unit-norm). Default = env parity; lower it with --max-speed for a cautious first flight.
    max_speed: float = math.sqrt(3.0)
    yaw_rate_max: float = 0.5  # cap on |yaw_rate| action


class Safety:
    """Turns policy actions into per-drone commands: ('rc', action) | ('hover', 0) | ('land', 0)."""

    def __init__(self, cfg: SafetyCfg):
        self.cfg = cfg
        self.landed = set()  # once landed for safety, stays down

    def apply(self, actions, own_pos, pose_age, link_age, battery, flying, killed=False):
        cfg = self.cfg
        D = actions.shape[0]
        lo = np.array(cfg.geofence[0::2], float)
        hi = np.array(cfg.geofence[1::2], float)
        out, reasons = [], []
        for i in range(D):
            a = np.clip(np.asarray(actions[i], float).copy(), -1.0, 1.0)
            why = []
            if killed:
                self.landed.add(i)
                why.append("kill")
            if link_age[i] > cfg.link_timeout_s:
                self.landed.add(i)
                why.append(f"link {link_age[i]:.1f}s")
            if battery[i] is not None and battery[i] < cfg.battery_floor:
                self.landed.add(i)
                why.append(f"battery {battery[i]}%")
            if pose_age[i] > cfg.stale_land_s:
                self.landed.add(i)
                why.append(f"pose {pose_age[i]:.1f}s")
            if i in self.landed or not flying[i]:
                out.append(("land", None) if i in self.landed else ("idle", None))
                reasons.append(",".join(why) or ("landed" if i in self.landed else "not flying"))
                continue
            if pose_age[i] > cfg.stale_hover_s:
                out.append(("hover", np.zeros(4)))
                reasons.append(f"stale pose {pose_age[i]:.2f}s")
                continue
            p = own_pos[i]
            # separation: remove closing motion inside soft_sep, back off inside min_sep
            for j in range(D):
                if j == i or j in self.landed or not flying[j] or pose_age[j] > cfg.stale_land_s:
                    continue
                d_vec = p - own_pos[j]
                d = float(np.linalg.norm(d_vec))
                u = d_vec / d if d > 1e-6 else np.array([0.0, 0.0, 1.0 if i < j else -1.0])
                if d < cfg.min_sep:
                    a[:3] = 0.6 * u
                    a[2] += 0.3 if i < j else -0.3  # vertical split, lower index climbs
                    why.append(f"sep {d:.2f}m from {j}")
                    break
                if d < cfg.soft_sep:
                    closing = float(a[:3] @ -u)
                    if closing > 0:
                        a[:3] += closing * u
                        why.append(f"soft-sep {j}")
            # geofence: outside -> push back in; within margin -> no outward component
            for k in range(3):
                if p[k] < lo[k]:
                    a[k] = max(a[k], 0.5)
                    why.append("fence-in")
                elif p[k] > hi[k]:
                    a[k] = min(a[k], -0.5)
                    why.append("fence-in")
                elif p[k] < lo[k] + cfg.fence_margin and a[k] < 0:
                    a[k] *= (p[k] - lo[k]) / cfg.fence_margin
                    why.append("fence")
                elif p[k] > hi[k] - cfg.fence_margin and a[k] > 0:
                    a[k] *= (hi[k] - p[k]) / cfg.fence_margin
                    why.append("fence")
            n = float(np.linalg.norm(a[:3]))
            if n > cfg.max_speed:
                a[:3] *= cfg.max_speed / n
            a[3] = float(np.clip(a[3], -cfg.yaw_rate_max, cfg.yaw_rate_max))
            out.append(("rc", a))
            reasons.append(",".join(why))
        return out, reasons


# =============================================================================================
# Feed adapters (pluggable: the organizer format is unknown)
# =============================================================================================
@dataclass
class Track:
    name: str
    team: str  # "attack" | "defense" | ""
    pos: np.ndarray | None  # sim frame, or None if unknown
    t_meas: float  # measurement time on OUR clock (t_recv - age)
    yaw: float | None = None  # sim yaw (rad)
    link: bool | None = None
    fly: bool | None = None
    bat: int | None = None
    h: float | None = None  # Tello height above takeoff


class TelemetryV1Adapter:
    """drone_telemetry wire format v1 (drone_telemetry/README.md): per-drone pose in a TAG frame."""

    name = "telemetry"

    def __init__(self, frame: Frame):
        self.frame = frame
        self.unmapped = set()

    def parse(self, msg: dict, t_recv: float) -> list:
        if msg.get("v") != 1 or "drones" not in msg:
            return []
        out = []
        for d in msg["drones"]:
            p = d.get("pose")
            pos, yaw, t_meas = None, None, -math.inf
            if p is not None and p.get("xyz") is not None:
                pm = self.frame.tag_to_map(int(p["tag"]), p["xyz"])
                if pm is None:
                    self.unmapped.add(int(p["tag"]))
                else:
                    pos = self.frame.map_to_sim(pm)
                    if p.get("yaw") is not None:
                        yaw = self.frame.yaw_from_tag(int(p["tag"]), float(p["yaw"]))
                    t_meas = t_recv - float(p.get("age", 0.0))
            out.append(Track(name=str(d.get("name")), team=str(d.get("team") or ""), pos=pos, t_meas=t_meas,
                             yaw=yaw, link=d.get("link"), fly=d.get("fly"), bat=d.get("bat"), h=d.get("h")))
        return out


class GenericAdapter:
    """Best-effort reader for an unknown organizer JSON: a list under drones/objects/targets/attackers/tracks
    (or the message itself), id from name/id, position from xyz/pos/position/[x,y,z], optional tag (then a
    tag-frame pose), time from t/stamp/timestamp (sender clock, used only for age if `age` is absent).
    `frame_in` = 'map' (default: the origin-tag frame), 'sim' (already z-up), or 'tag'."""

    name = "generic"
    LIST_KEYS = ("drones", "objects", "targets", "attackers", "tracks", "poses")

    def __init__(self, frame: Frame, frame_in: str = "map", team: str = "attack"):
        self.frame, self.frame_in, self.team = frame, frame_in, team

    def _items(self, msg):
        if isinstance(msg, list):
            return msg
        for k in self.LIST_KEYS:
            if isinstance(msg.get(k), list):
                return msg[k]
            if isinstance(msg.get(k), dict):  # {"id": {...}}
                return [dict(v, id=kk) for kk, v in msg[k].items()]
        return [msg]

    def parse(self, msg, t_recv: float) -> list:
        out = []
        for d in self._items(msg):
            if not isinstance(d, dict):
                continue
            name = d.get("name", d.get("id"))
            if name is None:
                continue
            p = d.get("pose") if isinstance(d.get("pose"), dict) else d
            xyz = next((p[k] for k in ("xyz", "pos", "position", "p") if k in p), None)
            if xyz is None and all(k in p for k in ("x", "y", "z")):
                xyz = [p["x"], p["y"], p["z"]]
            if isinstance(xyz, dict):
                xyz = [xyz["x"], xyz["y"], xyz["z"]]
            pos = None
            if xyz is not None:
                xyz = np.asarray(xyz, float)
                if "tag" in p or self.frame_in == "tag":
                    pm = self.frame.tag_to_map(int(p.get("tag", 0)), xyz)
                    pos = None if pm is None else self.frame.map_to_sim(pm)
                elif self.frame_in == "sim":
                    pos = xyz
                else:
                    pos = self.frame.map_to_sim(xyz)
            age = p.get("age")
            if age is None:
                age = 0.0
            t_meas = t_recv - float(age) if pos is not None else -math.inf
            team = str(d.get("team") or self.team)
            out.append(Track(name=str(name), team=team, pos=pos, t_meas=t_meas, fly=d.get("fly")))
        return out


ADAPTERS = {"telemetry": TelemetryV1Adapter, "generic": GenericAdapter}


def make_adapter(kind: str, frame: Frame, **kw):
    if kind == "auto":
        return AutoAdapter(frame, **kw)
    if kind == "telemetry":
        return TelemetryV1Adapter(frame)
    return GenericAdapter(frame, **kw)


class AutoAdapter:
    """telemetry v1 when the datagram says so, else generic."""

    name = "auto"

    def __init__(self, frame: Frame, **kw):
        self.tel, self.gen = TelemetryV1Adapter(frame), GenericAdapter(frame, **kw)

    def parse(self, msg, t_recv):
        if isinstance(msg, dict) and msg.get("v") == 1 and "drones" in msg:
            return self.tel.parse(msg, t_recv)
        return self.gen.parse(msg, t_recv)


# --- sources: yield (t_recv, msg) --------------------------------------------------------------
class UdpSource:
    def __init__(self, port: int, host: str = "0.0.0.0", recorder=None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.setblocking(False)
        self.recorder = recorder

    def poll(self, now: float) -> list:
        out = []
        while True:
            try:
                data, _ = self.sock.recvfrom(65535)
            except (BlockingIOError, InterruptedError):
                break
            try:
                msg = json.loads(data)
            except ValueError:
                continue
            out.append((now, msg))
            if self.recorder:
                self.recorder.write(now, msg)
        return out


class Recorder:
    """JSONL: one {"t_recv": wall clock, "msg": datagram} per line -- the --replay input."""

    def __init__(self, path):
        self.f = open(path, "a", buffering=1)

    def write(self, t_recv, msg):
        self.f.write(json.dumps({"t_recv": round(t_recv, 4), "msg": msg}, separators=(",", ":")) + "\n")


def read_recording(path) -> list:
    """Recording -> [(t_recv, msg)] sorted. Accepts Recorder lines or bare datagrams (uses msg['t'])."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict) and "msg" in obj and "t_recv" in obj:
                rows.append((float(obj["t_recv"]), obj["msg"]))
            elif isinstance(obj, dict) and "t" in obj:
                rows.append((float(obj["t"]), obj))
    rows.sort(key=lambda r: r[0])
    return rows


class SyntheticWorld:
    """Dry-run world: attackers fly at the site (weaving), defenders follow the commanded rc with a first-order
    lag. Emits telemetry v1 datagrams in the origin-tag frame (tag 10), so the full frame chain is exercised."""

    def __init__(self, frame: Frame, site, names, n_attackers=2, seed=0, origin_tag=10, att_speed=0.6):
        self.frame, self.site, self.names = frame, np.asarray(site, float), list(names)
        self.rng = np.random.default_rng(seed)
        self.tag = origin_tag
        self.t = 0.0
        D = len(names)
        self.def_pos = self.site + np.array([[0.0, (i - (D - 1) / 2) * 1.2, 0.0] for i in range(D)])
        self.def_pos[:, 2] = 1.0
        self.def_vel = np.zeros((D, 3))
        self.def_yaw = np.zeros(D)
        self.rc = np.zeros((D, 4))
        ang = self.rng.uniform(-0.8, 0.8, n_attackers)
        self.att_pos = self.site + np.stack([4.0 * np.cos(ang), 4.0 * np.sin(ang), 0.3 * self.rng.standard_normal(n_attackers)], -1)
        self.att_alive = np.ones(n_attackers, bool)
        self.att_speed = att_speed
        self.downed = 0

    def command(self, i: int, rc):
        self.rc[i] = np.asarray(rc, float) / 100.0

    def step(self, dt: float):
        self.t += dt
        for i in range(len(self.names)):
            right, fwd, up, yaw_cw = self.rc[i]
            c, s = math.cos(self.def_yaw[i]), math.sin(self.def_yaw[i])
            left = -right
            v_des = np.array([c * fwd - s * left, s * fwd + c * left, up]) * MAX_LIN_SPEED
            self.def_vel[i] += (v_des - self.def_vel[i]) * min(1.0, dt / 0.25)
            self.def_pos[i] += self.def_vel[i] * dt
            self.def_yaw[i] += -yaw_cw * MAX_YAW_RATE * dt
        for k in range(len(self.att_pos)):
            if not self.att_alive[k]:
                self.att_pos[k, 2] = max(0.05, self.att_pos[k, 2] - 1.5 * dt)  # falls
                continue
            to = self.site - self.att_pos[k]
            d = np.linalg.norm(to)
            u = to / max(d, 1e-6)
            lat = np.array([-u[1], u[0], 0.0]) * 0.5 * math.sin(1.5 * self.t + k)
            self.att_pos[k] += (u + lat) / np.linalg.norm(u + lat) * self.att_speed * dt
            if np.min(np.linalg.norm(self.def_pos - self.att_pos[k], axis=-1)) < 0.25:
                self.att_alive[k] = False
                self.downed += 1

    def datagram(self) -> dict:
        drones = []

        def pose(p_sim, yaw_sim):
            p_tag = self.frame.sim_to_map(p_sim)  # origin tag: map frame == tag frame (R = I)
            # sim yaw -> telemetry yaw for the origin tag (inverse of Frame.yaw_from_tag with R_tag = I)
            f_sim = np.array([math.cos(yaw_sim), math.sin(yaw_sim), 0.0])
            f_tag = self.frame.R.T @ f_sim
            return {"tag": self.tag, "xyz": [round(float(v), 4) for v in p_tag],
                    "yaw": round(math.degrees(math.atan2(f_tag[0], f_tag[2])), 2), "age": 0.03}

        for i, n in enumerate(self.names):
            drones.append({"name": n, "slot": i + 1, "team": "defense", "link": True, "fly": True, "mode": "auto",
                           "bat": 80, "h": round(float(self.def_pos[i, 2]), 2), "pose": pose(self.def_pos[i], self.def_yaw[i])})
        for k in range(len(self.att_pos)):
            drones.append({"name": f"ATT-{k}", "slot": 10 + k, "team": "attack", "link": True,
                           "fly": bool(self.att_alive[k]), "mode": "auto", "bat": 80,
                           "h": round(float(self.att_pos[k, 2]), 2),
                           "pose": pose(self.att_pos[k], 0.0)})
        return {"v": 1, "src": "synthetic", "seq": int(self.t / DT), "t": self.t, "hz": 1 / DT, "drones": drones}


# =============================================================================================
# Env-parity world: AS-Defend-Live-v0 curriculum stages, closed loop through the whole pipeline
# =============================================================================================
# numpy copy of tello_defend_env_cfg.CURRICULUM (only what the world needs) + Live-v0 defaults
STAGES = {
    0: dict(active=(1, 1), difficulty=-1.0, speed=(0.0, 0.0), spawn_radius=3.5, capture="radius", capture_radius=0.8),
    1: dict(active=(1, 2), difficulty=0.0, spawn_radius=4.5, capture="radius", capture_radius=0.6),
    2: dict(active=(2, 2), difficulty=0.2, spawn_radius=6.0, capture="radius", capture_radius=0.4),
    3: dict(active=(2, 2), difficulty=0.3, spawn_radius=6.0, capture="ram", capture_radius=0.25),
    4: dict(active=(2, 3), difficulty=0.4, spawn_radius=6.0, capture="ram", capture_radius=0.25, feed=True, own_noise=0.02),
    5: dict(active=(2, 3), difficulty=0.6, spawn_radius=6.0, capture="ram", capture_radius=0.25, feed=True, own_noise=0.02,
            loss_prob=0.2),
    6: dict(active=(2, 4), difficulty=0.9, spawn_radius=6.0, capture="ram", capture_radius=0.25, feed=True, own_noise=0.02,
            loss_prob=0.3),
}
STRAIGHT, WEAVE, WAYPOINT, DASH = 0, 1, 2, 3


def difficulty_params(d: float) -> dict:
    """numpy copy of attackers.difficulty_params."""
    d = float(min(max(d, 0.0), 1.0))
    probs = np.array([max(0.1, 1.0 - 1.2 * d), d, max(0.0, 1.5 * (d - 0.3)), max(0.0, 1.5 * (d - 0.5))])
    return dict(speed=(0.2 + 0.6 * d, 0.4 + 0.8 * d), probs=probs / probs.sum(),
                evade_prob=min(max(1.6 * (d - 0.5), 0.0), 0.8), weave_amp=0.4 + 0.6 * d)


class EnvWorld(SyntheticWorld):
    """AS-Defend-Live-v0 at curriculum stage `stage`, one env, in the sim frame around `site`.

    Same rules as tello_defend_env: defenders spawn on a 2.5 m ring (evenly spaced +-0.3 rad, z 1-2 m, random yaw),
    attackers on the stage's spawn ring (z 1.5-3 m above the floor), scripted by the numpy port of
    attackers.AttackerBrain (velocity command once per policy step, max 1.3 m/s), capture once per policy step
    (radius, or ram = radius AND closing > 0.3 m/s), a breach = a live attacker within 1.5 m (horizontal) of the site,
    waves respawn when cleared, the episode ends on breach / all defenders lost / 20 s. Defenders: crash below 0.15 m,
    above 4 m, beyond arena_radius, or within 0.2 m of each other. Dynamics: TelloVelocityController-like first-order
    command lag + velocity loop (tau 0.15 s, 4 m/s^2 clamps, 120 Hz substeps); stages >= 4 use the FAST_DR means (tau 0.27 / z 0.15, 3.2 m/s^2).
    Stages >= 4 also get the feed model (0-0.3 s lag, 3 cm noise, 10% dropout) and 2 cm own-pose noise.
    Downed attackers fall and are reported `fly: false`. Emits telemetry v1 like SyntheticWorld."""

    SITE_RADIUS, DEF_RADIUS, EP_STEPS = 1.5, 2.5, int(round(20.0 / DT))
    SUBSTEPS = 9  # 120 Hz physics like FAST_PHYSICS

    def __init__(self, frame: Frame, site, names, stage: int = 3, seed=0, origin_tag=10, ram_min_speed=0.3,
                 att_max_speed=1.3, evade_radius=1.5):
        self.stage = int(stage)
        self.p = dict(STAGES[self.stage])
        self.ram_min_speed, self.att_max_speed, self.evade_radius = ram_min_speed, att_max_speed, evade_radius
        self.arena_radius = max(9.0, self.p["spawn_radius"] + 3.0)
        fast_dr = self.stage >= 4
        self.tau_xy, self.tau_z, self.acc = (0.27, 0.15, 3.2) if fast_dr else (0.15, 0.15, 4.0)
        super().__init__(frame, site, names, n_attackers=4, seed=seed, origin_tag=origin_tag)
        self.reset()

    # -- episode --------------------------------------------------------------------------
    def reset(self):
        D, A = len(self.names), 4
        rng = self.rng
        self.t, self.steps, self.downed, self.breached, self.crashes, self.waves = 0.0, 0, 0, False, 0, 0
        base = rng.uniform(0, 2 * math.pi)
        ang = base + np.arange(D) * 2 * math.pi / D + rng.uniform(-0.3, 0.3, D)
        self.def_pos = np.stack([self.site[0] + self.DEF_RADIUS * np.cos(ang), self.site[1] + self.DEF_RADIUS * np.sin(ang),
                                 rng.uniform(1.0, 2.0, D)], -1)
        self.def_vel = np.zeros((D, 3))
        self.def_cmd = np.zeros((D, 3))  # filtered command (TelloVelocityController._lin_vel_cmd)
        self.def_yaw = rng.uniform(-math.pi, math.pi, D)
        self.def_alive = np.ones(D, bool)
        self.rc = np.zeros((D, 4))
        lo, hi = self.p["active"]
        n = int(rng.integers(lo, hi + 1))
        self.att_active = np.arange(A) < n
        self.att_pos = np.zeros((A, 3))
        self.att_vel = np.zeros((A, 3))
        self.att_cmd = np.zeros((A, 3))
        self.att_alive = np.zeros(A, bool)
        self.feed_lag = rng.uniform(0.0, 0.3) if self.p.get("feed") else 0.0
        self.hist = []  # (t, att_pos) for the delayed feed
        self._spawn_wave()

    def _spawn_wave(self):
        rng, A = self.rng, 4
        act = self.att_active
        ang = rng.uniform(0, 2 * math.pi, A)
        r = self.p["spawn_radius"]
        pos = np.stack([self.site[0] + r * np.cos(ang), self.site[1] + r * np.sin(ang), rng.uniform(1.5, 3.0, A)], -1)
        self.att_pos[act] = pos[act]
        self.att_vel[act] = 0.0
        self.att_cmd[act] = 0.0
        self.att_alive = act.copy()
        # brain sample (attackers.AttackerBrain.sample)
        d = self.p["difficulty"]
        if d >= 0:
            q = difficulty_params(d)
            lo, hi = q["speed"]
            self.beh = rng.choice(4, size=A, p=q["probs"])
            self.evasive = rng.random(A) < q["evade_prob"]
            self.amp = np.full(A, q["weave_amp"])
        else:  # legacy straight dive at attacker_speed
            lo, hi = self.p["speed"]
            self.beh = np.zeros(A, int)
            self.evasive = np.zeros(A, bool)
            self.amp = np.zeros(A)
        self.att_speed = rng.uniform(lo, hi, A)
        to_site = self.site - self.att_pos
        lat = np.stack([-to_site[:, 1], to_site[:, 0], np.zeros(A)], -1)
        lat /= np.maximum(np.linalg.norm(lat, axis=-1, keepdims=True), 1e-6)
        off = (rng.random(A) * 1.5 + 1.5) * np.where(rng.random(A) < 0.5, -1.0, 1.0)
        self.waypoint = self.att_pos + 0.5 * to_site + lat * off[:, None]
        self.waypoint[:, 2] = self.site[2] - 1.25 + rng.random(A) * 1.5 + 1.0
        self.reached = np.zeros(A, bool)
        self.hold = rng.random(A) * 3.0 + 1.0
        self.phase = rng.random(A) * 2 * math.pi
        self.bt = np.zeros(A)
        self.waves += 1

    @property
    def done(self) -> bool:
        return self.breached or not self.def_alive.any() or self.steps >= self.EP_STEPS

    # -- attacker brain (attackers.AttackerBrain.velocity) --------------------------------
    def _brain_velocity(self):
        A = 4
        to_site = self.site - self.att_pos
        to_wp = self.waypoint - self.att_pos
        self.reached |= (self.beh == WAYPOINT) & (np.linalg.norm(to_wp, axis=-1) < 0.5)
        go_wp = (self.beh == WAYPOINT) & ~self.reached
        goal = np.where(go_wp[:, None], to_wp, to_site)
        d = goal / np.maximum(np.linalg.norm(goal, axis=-1, keepdims=True), 1e-6)
        lat = np.stack([-d[:, 1], d[:, 0], np.zeros(A)], -1)
        lat /= np.maximum(np.linalg.norm(lat, axis=-1, keepdims=True), 1e-6)
        weave = (self.beh == WEAVE) * self.amp
        s = np.sin(2.0 * self.bt + self.phase)
        up = np.array([0.0, 0.0, 1.0])
        d = d + weave[:, None] * (s[:, None] * lat + 0.3 * np.cos(1.3 * self.bt + self.phase)[:, None] * up)
        rel = self.att_pos[:, None] - self.def_pos[None]  # (A, D, 3)
        dist = np.where(self.def_alive[None], np.linalg.norm(rel, axis=-1), 1e3)
        j = dist.argmin(-1)
        dmin = dist[np.arange(A), j]
        away = rel[np.arange(A), j]
        away = away - (away * d).sum(-1, keepdims=True) * d / np.maximum((d * d).sum(-1, keepdims=True), 1e-6)
        away /= np.maximum(np.linalg.norm(away, axis=-1, keepdims=True), 1e-6)
        w = (self.evasive & (dmin < self.evade_radius)) * 2.0 * np.maximum(1.0 - dmin / self.evade_radius, 0.0)
        d = d + w[:, None] * away
        d /= np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-6)
        factor = np.ones(A)
        dash = self.beh == DASH
        factor[dash & (self.bt < self.hold)] = 0.15
        factor[dash & (self.bt >= self.hold)] = 1.4
        speed = np.minimum(self.att_speed * factor, self.att_max_speed)
        return d * speed[:, None]

    # -- one policy step ------------------------------------------------------------------
    def step(self, dt: float):
        if self.done:
            return
        self.t += dt
        self.steps += 1
        self.bt += dt
        att_target = np.where(self.att_alive[:, None], self._brain_velocity(), 0.0)
        D = len(self.names)
        v_des = np.zeros((D, 3))
        for i in range(D):
            right, fwd, up, yaw_cw = self.rc[i] if self.def_alive[i] else (0.0, 0.0, 0.0, 0.0)
            c, s = math.cos(self.def_yaw[i]), math.sin(self.def_yaw[i])
            v_des[i] = np.array([c * fwd + s * right, s * fwd - c * right, up]) * MAX_LIN_SPEED
        h = dt / self.SUBSTEPS
        tau = np.array([self.tau_xy, self.tau_xy, self.tau_z])
        for _ in range(self.SUBSTEPS):
            # defenders: filtered command, accel-limited tracking; dead ones fall
            # TelloVelocityController.compute: lagged command, then accel = (cmd - v) / tau, both accel-clamped
            self.def_cmd += np.clip((v_des - self.def_cmd) * np.minimum(1.0, h / tau), -self.acc * h, self.acc * h)
            dv = np.clip((self.def_cmd - self.def_vel) / tau, -self.acc, self.acc) * h
            self.def_vel = np.where(self.def_alive[:, None], self.def_vel + dv, self.def_vel + np.array([0, 0, -9.81 * h]))
            self.def_pos += self.def_vel * h
            self.def_pos[:, 2] = np.maximum(self.def_pos[:, 2], 0.0)
            for i in range(D):
                if self.def_alive[i]:
                    self.def_yaw[i] += -self.rc[i, 3] * MAX_YAW_RATE * h
            # attackers (velocity controller at 1.3 m/s, tau 0.15, 4 m/s^2); downed ones fall
            self.att_cmd += np.clip((att_target - self.att_cmd) * min(1.0, h / 0.15), -4.0 * h, 4.0 * h)
            dva = np.clip((self.att_cmd - self.att_vel) / 0.15, -4.0, 4.0) * h
            fall = self.att_active & ~self.att_alive
            self.att_vel = np.where(fall[:, None], self.att_vel + np.array([0, 0, -9.81 * h]), self.att_vel + dva)
            self.att_pos += self.att_vel * h
            ground = self.att_pos[:, 2] <= 0.0
            self.att_pos[ground, 2] = 0.0
            self.att_vel[ground] = 0.0
        self._events()
        self.hist.append((self.t, self.att_pos.copy()))
        if len(self.hist) > 20:
            self.hist.pop(0)

    def _events(self):
        p = self.p
        rel = self.att_pos[None] - self.def_pos[:, None]  # (D, A, 3)
        dist = np.linalg.norm(rel, axis=-1)
        hit = (dist < p["capture_radius"]) & self.att_alive[None] & self.def_alive[:, None]
        if p["capture"] == "ram":
            closing = ((self.def_vel[:, None] - self.att_vel[None]) * rel).sum(-1) / np.maximum(dist, 1e-6)
            hit &= closing > self.ram_min_speed
        captured = hit.any(0)
        self.att_alive &= ~captured
        self.downed += int(captured.sum())
        z = self.def_pos[:, 2]
        crash = (z < 0.15) | (z > 4.0) | (np.linalg.norm(self.def_pos[:, :2] - self.site[:2], axis=-1) > self.arena_radius)
        dd = np.linalg.norm(self.def_pos[:, None] - self.def_pos[None], axis=-1) + np.eye(len(z)) * 99
        crash |= ((dd < 0.2) & self.def_alive[None]).any(-1)
        crash &= self.def_alive
        lost = crash.copy()
        if p.get("loss_prob", 0.0) > 0:
            lost |= hit.any(-1) & (self.rng.random(len(z)) < p["loss_prob"])
        lost &= self.def_alive
        self.crashes += int(crash.sum())
        self.def_alive &= ~lost
        radial = np.linalg.norm(self.att_pos[:, :2] - self.site[:2], axis=-1)
        self.breached = bool(((radial < self.SITE_RADIUS) & self.att_alive).any()) or not self.def_alive.any()
        if not self.att_alive.any() and not self.done:
            self._spawn_wave()

    # -- feed -----------------------------------------------------------------------------
    def datagram(self) -> dict:
        msg = super().datagram()
        att_seen = self.att_pos
        if self.p.get("feed") and self.hist:
            t_want = self.t - self.feed_lag
            att_seen = min(self.hist, key=lambda r: abs(r[0] - t_want))[1]
        drones = []
        for d in msg["drones"]:
            if d["team"] == "attack":
                k = int(d["name"].split("-")[1])
                if not self.att_active[k]:
                    continue  # parked: not in the arena
                if self.p.get("feed"):
                    if self.rng.random() < 0.1:
                        continue  # dropout
                    p_sim = att_seen[k] + self.rng.normal(0, 0.03, 3)
                    d["pose"]["xyz"] = [round(float(v), 4) for v in self.frame.sim_to_map(p_sim)]
                d["fly"] = bool(self.att_alive[k])
            else:
                i = self.names.index(d["name"])
                d["fly"] = bool(self.def_alive[i])
                if self.p.get("own_noise"):
                    p_sim = self.def_pos[i] + self.rng.normal(0, self.p["own_noise"], 3)
                    d["pose"]["xyz"] = [round(float(v), 4) for v in self.frame.sim_to_map(p_sim)]
            drones.append(d)
        msg["drones"] = drones
        return msg


def sim_eval(stage: int, make_policy, episodes: int = 64, seed: int = 0, frame: Frame | None = None,
             site=(3.0, 0.0, SITE_HEIGHT / 2), geofence=None, defenders=("TELLO-1", "TELLO-2", "TELLO-3"),
             safety: SafetyCfg | None = None, kalman: KalmanCfg | None = None, max_speed: float | None = None) -> dict:
    """Closed loop: EnvWorld(stage) -> telemetry -> DefendLive (filters, obs, policy, safety, rc) -> EnvWorld.

    Returns the env's episode metrics (downed/ep, breach rate, crashes/ep) + tick latency. `geofence` None =
    the env's arena (site +-arena_radius, z 0.15-4), i.e. the safety layer only separates defenders."""
    frame = frame or Frame("ydown_zin", 1.5, {10: (np.eye(3), np.zeros(3))})
    site = np.asarray(site, float)
    downed, breach, crashes, steps, lat = [], [], [], [], []
    for ep in range(episodes):
        world = EnvWorld(frame, site, list(defenders), stage=stage, seed=seed * 100003 + ep)
        if geofence is None:
            r = world.arena_radius
            gf = (site[0] - r, site[0] + r, site[1] - r, site[1] + r, 0.15, 4.0)
        else:
            gf = geofence
        sc = safety or SafetyCfg(geofence=gf)
        if safety is None:
            sc.yaw_rate_max = 1.0
            if max_speed is not None:
                sc.max_speed = float(max_speed)
        cfg = Config(defenders=list(defenders), site=tuple(site), safety=sc, kalman=kalman or KalmanCfg())
        links = {n: NullLink(n, world, i) for i, n in enumerate(defenders)}
        runner = DefendLive(cfg, make_policy(), [make_adapter("telemetry", frame)], links)
        t = 0.0
        while not world.done:
            world.step(DT)
            t += DT
            for i, n in enumerate(defenders):  # a crashed defender is down for the rest of the episode
                if not world.def_alive[i]:
                    links[n].flying = False
            t0 = time.perf_counter()
            runner.tick(t, [(t, world.datagram())])
            lat.append(time.perf_counter() - t0)
        downed.append(world.downed)
        breach.append(world.breached)  # incl. all defenders lost, like the env
        crashes.append(world.crashes)
        steps.append(world.steps)
    lat_ms = np.array(lat) * 1e3
    return {"stage": stage, "episodes": episodes, "downed_per_ep": float(np.mean(downed)),
            "downed_se": float(np.std(downed) / math.sqrt(max(episodes, 1))), "breach_rate": float(np.mean(breach)),
            "crashes_per_ep": float(np.mean(crashes)), "ep_len_s": float(np.mean(steps) * DT),
            "tick_ms": {"p50": float(np.percentile(lat_ms, 50)), "p99": float(np.percentile(lat_ms, 99)),
                        "max": float(lat_ms.max())}}


# =============================================================================================
# Drone links
# =============================================================================================
class NullLink:
    """Dry-run / replay link: records what would be sent. Never touches the network."""

    def __init__(self, name: str, world: SyntheticWorld | None = None, index: int = 0):
        self.name, self.world, self.index = name, world, index
        self.sent = []
        self.flying = world is not None
        self.battery = 80 if world is not None else None
        self.last_state = 0.0

    def link_age(self, now):
        return 0.0

    def rc(self, a, b, c, d):
        self.sent.append(("rc", a, b, c, d))
        if self.world is not None:
            self.world.command(self.index, (a, b, c, d))

    def land(self):
        self.sent.append(("land",))
        self.flying = False
        if self.world is not None:
            self.world.command(self.index, (0, 0, -60, 0))

    def takeoff(self):
        self.sent.append(("takeoff",))
        self.flying = True

    def close(self):
        pass


class TelloLink:
    """One Tello over its USB dongle (SO_BINDTODEVICE), like core/tello_dual_video.py. LIVE ONLY.

    The WiFi join itself is core/tello_link.py's job (nmcli); run it with --no_fly first, or pass
    --join-wifi to call its connect_drone() from here."""

    TELLO = ("192.168.10.1", 8889)
    allow = False  # set by main() only for --live --i-know-this-flies

    def __init__(self, name: str, iface: str, local_port: int):
        if not TelloLink.allow:
            raise RuntimeError("TelloLink is disabled outside --live --i-know-this-flies")
        self.name, self.iface = name, iface
        self.cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, iface.encode())
        self.cmd.bind(("0.0.0.0", local_port))
        self.state = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.state.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.state.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, iface.encode())
        self.state.bind(("0.0.0.0", 8890))
        self.state.settimeout(0.5)
        self.last_state = 0.0
        self.battery = None
        self.flying = False
        self._stop = threading.Event()
        threading.Thread(target=self._read_state, daemon=True).start()
        self._send("command")

    def _read_state(self):
        while not self._stop.is_set():
            try:
                data, _ = self.state.recvfrom(2048)
            except (socket.timeout, OSError):
                continue
            self.last_state = time.monotonic()
            for kv in data.decode(errors="ignore").strip().split(";"):
                k, _, v = kv.partition(":")
                if k == "bat":
                    try:
                        self.battery = int(v)
                    except ValueError:
                        pass

    def _send(self, text: str):
        try:
            self.cmd.sendto(text.encode(), self.TELLO)
        except OSError:
            pass

    def link_age(self, now):
        return time.monotonic() - self.last_state if self.last_state else math.inf

    def rc(self, a, b, c, d):
        self._send(f"rc {a} {b} {c} {d}")

    def takeoff(self):
        self._send("takeoff")
        self.flying = True

    def land(self):
        self._send("rc 0 0 0 0")
        self._send("land")
        self.flying = False

    def close(self):
        self._stop.set()


def dongle_map(path=DEFAULT_DONGLES) -> dict:
    """{drone name: iface} from config/dongles.json."""
    reg = json.loads(Path(path).read_text())
    return {d["drone"]: d["iface"] for d in reg.get("dongles", []) if d.get("drone")}


# =============================================================================================
# The runner
# =============================================================================================
@dataclass
class Config:
    defenders: list = field(default_factory=lambda: ["TELLO-1", "TELLO-2", "TELLO-3"])
    site: tuple = (3.0, 0.0, SITE_HEIGHT / 2)
    safety: SafetyCfg = field(default_factory=SafetyCfg)
    attacker_stale_s: float = 1.0  # attacker track older than this -> slot invalid
    attacker_fallen_z: float = 0.25  # attacker below this height -> treated as downed
    lead_latency_s: float = 0.0  # extrapolate attacker estimates by this much (0 = sim parity)
    attacker_reseed_m: float = 1.5  # meas this far from the estimate (or fly False -> True) = respawn: re-seed, vel 0
    kalman: KalmanCfg = field(default_factory=KalmanCfg)


class DefendLive:
    def __init__(self, cfg: Config, policy, adapters: list, links: dict, log=None, verbose=False):
        self.cfg, self.policy, self.adapters, self.links = cfg, policy, adapters, links
        self.D = len(cfg.defenders)
        self.site = np.asarray(cfg.site, float)
        self.kf = KalmanCV(cfg.kalman, (self.D,))
        self.own_t_meas = np.full(self.D, -math.inf)  # last fused measurement time
        self.own_yaw = np.zeros(self.D)
        self.own_meta = [dict() for _ in range(self.D)]
        self.attackers = {}  # name -> dict(est_pos, est_vel, t_meas, new, fly)
        self.prev_target = None
        self.safety = Safety(cfg.safety)
        self.log, self.verbose = log, verbose
        self.killed = False
        self.t_prev = None
        self.ticks = 0
        self.stats = {"datagrams": 0, "own_meas": 0, "att_meas": 0, "reasons": {}}

    # -- ingest -----------------------------------------------------------------------------
    def ingest(self, datagrams: list):
        idx = {n: i for i, n in enumerate(self.cfg.defenders)}
        for t_recv, msg in datagrams:
            self.stats["datagrams"] += 1
            for adapter in self.adapters:
                for tr in adapter.parse(msg, t_recv):
                    if tr.name in idx:
                        i = idx[tr.name]
                        self.own_meta[i] = {"link": tr.link, "fly": tr.fly, "bat": tr.bat, "h": tr.h}
                        if tr.pos is not None and tr.t_meas > self.own_t_meas[i]:
                            self._own_meas(i, tr)
                    elif tr.team == "attack" or (tr.team == "" and tr.pos is not None):
                        self._att_meas(tr)

    def _own_meas(self, i, tr):
        m = np.zeros(self.D, bool)
        m[i] = True
        z = np.zeros((self.D, 3))
        z[i] = tr.pos
        if self.own_t_meas[i] > -math.inf and tr.t_meas - self.own_t_meas[i] > self.kf.cfg.lost_after:
            self.kf.reset(m)
        ok = self.kf.update(z, m)
        if ok[i]:
            self.own_t_meas[i] = tr.t_meas
            self.stats["own_meas"] += 1
        if tr.yaw is not None:
            self.own_yaw[i] = tr.yaw

    def _att_meas(self, tr):
        a = self.attackers.get(tr.name)
        if a is None:
            if tr.pos is None:
                return
            a = self.attackers[tr.name] = {"est_pos": tr.pos.copy(), "est_vel": np.zeros(3), "t_meas": tr.t_meas,
                                           "meas": None, "fly": tr.fly}
        # respawn / teleport (the env re-seeds the feed filter on a new wave: feed.AttackerFeed.reset_mask)
        revived = a["fly"] is False and tr.fly is True
        if tr.pos is not None and (revived or np.linalg.norm(tr.pos - a["est_pos"]) > self.cfg.attacker_reseed_m):
            a["est_pos"], a["est_vel"], a["t_meas"], a["meas"] = tr.pos.copy(), np.zeros(3), tr.t_meas, None
            a["fly"] = tr.fly
            return
        a["fly"] = tr.fly
        if tr.pos is not None and tr.t_meas > a["t_meas"]:
            a["meas"], a["t_meas"] = tr.pos.copy(), tr.t_meas
            self.stats["att_meas"] += 1
        elif tr.pos is not None and a["meas"] is None and tr.t_meas == a["t_meas"]:
            a["meas"] = tr.pos.copy()

    # -- one control tick -------------------------------------------------------------------
    def predict(self, now: float) -> float:
        """Advance own-drone Kalman to `now` BEFORE fusing this tick's poses (else the estimate runs one tick
        ahead of the measurement, ~7 cm at 1 m/s). Returns dt."""
        dt = DT if self.t_prev is None else min(max(now - self.t_prev, 1e-3), 0.5)
        self.t_prev = now
        self.kf.predict(dt, active=(now - self.own_t_meas) <= self.kf.cfg.lost_after)
        return dt

    def state(self, now: float, dt: float = DT):
        # own drones: freeze when lost (predicted in predict(), fused in ingest())
        pose_age = now - self.own_t_meas
        lost = pose_age > self.kf.cfg.lost_after
        self.kf.freeze(lost)
        own_pos, own_vel = self.kf.p.copy(), self.kf.v.copy()
        # attackers: alpha-beta at the tick rate (sim parity), stale/fallen -> invalid
        live = []
        for name, a in self.attackers.items():
            measured = a["meas"] is not None
            meas = a["meas"] if measured else a["est_pos"]
            a["est_pos"], a["est_vel"] = alpha_beta_step(a["est_pos"], a["est_vel"], meas, measured, dt, FEED_ALPHA, FEED_BETA)
            a["meas"] = None
            age = now - a["t_meas"]
            fallen = a["est_pos"][2] < self.cfg.attacker_fallen_z or a["fly"] is False
            if age <= self.cfg.attacker_stale_s and not fallen:
                pos = a["est_pos"] + a["est_vel"] * self.cfg.lead_latency_s
                live.append((float(np.linalg.norm(pos - self.site)), name, pos, a["est_vel"].copy()))
        live.sort(key=lambda r: r[0])
        att_pos, att_vel = np.zeros((NUM_SLOTS, 3)), np.zeros((NUM_SLOTS, 3))
        att_alive = np.zeros(NUM_SLOTS, bool)
        names = []
        for k, (_, name, p, v) in enumerate(live[:NUM_SLOTS]):
            att_pos[k], att_vel[k], att_alive[k] = p, v, True
            names.append(name)
        return own_pos, own_vel, pose_age, att_pos, att_vel, att_alive, names

    def tick(self, now: float, datagrams: list):
        dt = self.predict(now)
        self.ingest(datagrams)
        own_pos, own_vel, pose_age, att_pos, att_vel, att_alive, att_names = self.state(now, dt)
        links = [self.links[n] for n in self.cfg.defenders]
        flying = np.array([l.flying for l in links])
        def_alive = flying & (pose_age <= self.cfg.safety.stale_land_s) & self.kf.ready
        obs, target, ttg, aim = build_obs(own_pos, own_vel, self.own_yaw, def_alive, att_pos, att_vel, att_alive,
                                          self.site, self.prev_target)
        self.prev_target = target if att_alive.any() else None
        ctx = dict(obs=obs, own_pos=own_pos, own_vel=own_vel, def_alive=def_alive, att_alive=att_alive,
                   target=target, aim=aim, site=self.site)
        actions = self.policy.act(ctx)
        battery = [l.battery if l.battery is not None else self.own_meta[i].get("bat") for i, l in enumerate(links)]
        link_age = np.array([l.link_age(now) for l in links])
        cmds, reasons = self.safety.apply(actions, own_pos, pose_age, link_age, battery, flying, self.killed)
        rcs = []
        for i, (link, (kind, a)) in enumerate(zip(links, cmds)):
            if kind == "land":
                if link.flying:
                    link.land()
                rcs.append("land")
            elif kind == "idle":
                rcs.append("idle")
            else:
                rc = world_to_rc(a, float(self.own_yaw[i]))
                link.rc(*rc)
                rcs.append(rc)
            for r in filter(None, reasons[i].split(",")):
                key = r.split(" ")[0]
                self.stats["reasons"][key] = self.stats["reasons"].get(key, 0) + 1
        self.ticks += 1
        rec = {"t": round(now, 4), "own_pos": own_pos.round(3).tolist(), "pose_age": np.minimum(pose_age, 99).round(3).tolist(),
               "att": att_names, "att_pos": att_pos[att_alive].round(3).tolist(), "target": target.tolist(),
               "rc": rcs, "why": reasons}
        if self.log:
            self.log.write(json.dumps(dict(rec, obs=obs.round(4).tolist()), separators=(",", ":")) + "\n")
        if self.verbose:
            print(f"t {now:8.2f}  att {len(att_names)}  target {target.tolist()}  rc {rcs}  {[r for r in reasons if r]}", flush=True)
        return rec, obs

    def land_all(self):
        for link in self.links.values():
            try:
                link.land()
            except Exception:  # noqa: BLE001 -- landing must try every drone
                pass


# =============================================================================================
# Replay report
# =============================================================================================
def replay_report(rows: list, frame: Frame, defenders: list) -> dict:
    """Feed stats + frame sanity from a recording (no control)."""
    tel = TelemetryV1Adapter(frame)
    gaps, ages, lat = [], [], []
    zs, hs, grounded_z = [], [], []
    prev = None
    names = set()
    for t_recv, msg in rows:
        if prev is not None:
            gaps.append(t_recv - prev)
        prev = t_recv
        if isinstance(msg, dict) and "t" in msg:
            lat.append(t_recv - float(msg["t"]))
        for tr in tel.parse(msg, t_recv) if isinstance(msg, dict) else []:
            names.add((tr.name, tr.team))
            if tr.pos is None:
                continue
            ages.append(t_recv - tr.t_meas)
            if tr.h is not None and tr.fly:
                zs.append(tr.pos[2])
                hs.append(tr.h)
            if tr.fly is False:
                grounded_z.append(tr.pos[2])
    rep = {"datagrams": len(rows), "duration_s": round(rows[-1][0] - rows[0][0], 2) if rows else 0.0,
           "drones": sorted(names), "unmapped_tags": sorted(tel.unmapped)}

    def pct(x, q):
        return round(float(np.percentile(x, q)), 3) if len(x) else None

    rep["interval_s"] = {"median": pct(gaps, 50), "p90": pct(gaps, 90), "max": pct(gaps, 100)}
    rep["pose_age_s"] = {"median": pct(ages, 50), "p90": pct(ages, 90)}
    rep["recv_minus_sender_t_s"] = {"median": pct(lat, 50), "p90": pct(lat, 90)}
    if len(zs) >= 5 and np.std(hs) > 0.05 and np.std(zs) > 1e-6:
        corr = float(np.corrcoef(zs, hs)[0, 1])
        rep["frame_check"] = {"corr_sim_z_vs_tello_h": round(corr, 3),
                              "verdict": f"--map-convention {frame.convention} looks right" if corr > 0.5 else
                              ("WRONG convention: try the other --map-convention" if corr < -0.5 else "inconclusive")}
    else:
        rep["frame_check"] = {"verdict": "inconclusive (need airborne drones with varying h)"}
    if grounded_z:
        rep["grounded_sim_z_median"] = pct(grounded_z, 50)
        rep["suggest_origin_height"] = round(-float(np.median(grounded_z)) + 0.05 + 0.0, 3)
    return rep


# =============================================================================================
# CLI
# =============================================================================================
class KeyboardKill:
    def __init__(self, on_kill):
        self.on_kill = on_kill
        if sys.stdin and sys.stdin.isatty():
            threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in sys.stdin:
            if line.strip().lower()[:1] in ("k", "q", "l", ""):
                self.on_kill("keyboard")
                return


def _floats(text, n):
    v = [float(x) for x in text.split(",")]
    if len(v) != n:
        raise argparse.ArgumentTypeError(f"want {n} comma-separated numbers")
    return tuple(v)


def default_geofence(map_info: dict, frame: Frame) -> tuple:
    if map_info.get("geofence"):
        return tuple(float(x) for x in map_info["geofence"])
    ys = [frame.map_to_sim(t)[1] for _, t in map_info["tags"].values()] or [0.0]
    return (0.6, 6.0, min(ys) - 3.0, max(ys) + 3.0, 0.4, 2.3)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="never contact a drone; print rc")
    mode.add_argument("--replay", metavar="FILE", help="run the pipeline on a recording (JSONL)")
    mode.add_argument("--live", action="store_true", help="fly (needs --i-know-this-flies)")
    ap.add_argument("--i-know-this-flies", action="store_true")
    ap.add_argument("--policy", choices=("assign-lead", "assign-lead-safe", "rl", "hover"), default="assign-lead")
    ap.add_argument("--weights", help=".npz MLP for --policy rl")
    ap.add_argument("--feed", default="udp:9100",
                    help="udp:PORT (live telemetry / organizer feed) | synthetic (dry-run world). Repeatable via comma")
    ap.add_argument("--format", choices=("auto", "telemetry", "generic"), default="auto", help="feed adapter")
    ap.add_argument("--generic-frame", choices=("map", "sim", "tag"), default="map")
    ap.add_argument("--defenders", default="TELLO-1,TELLO-2,TELLO-3")
    ap.add_argument("--map", default=str(DEFAULT_MAP))
    ap.add_argument("--map-convention", choices=sorted(CONVENTIONS), default="ydown_zin")
    ap.add_argument("--origin-height", type=float, default=1.5, help="origin tag centre above the floor [m]")
    ap.add_argument("--site", type=lambda s: _floats(s, 3), help="protected site centre, sim frame x,y,z")
    ap.add_argument("--geofence", type=lambda s: _floats(s, 6), help="sim xmin,xmax,ymin,ymax,zmin,zmax")
    ap.add_argument("--dongles", default=str(DEFAULT_DONGLES))
    ap.add_argument("--record", help="append raw feed datagrams here (JSONL, replayable)")
    ap.add_argument("--log", help="per-tick JSONL (obs, rc, safety reasons)")
    ap.add_argument("--ticks", type=int, default=0, help="stop after N ticks (0 = until killed / replay end)")
    ap.add_argument("--attackers", type=int, default=2, help="synthetic world attackers")
    ap.add_argument("--realtime", action="store_true", help="dry-run synthetic: sleep to 13.5 Hz")
    ap.add_argument("--episodes", type=int, default=64, help="--feed stage:K closed-loop episodes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kalman", type=lambda s: _floats(s, 2), metavar="MEAS_SIGMA,ACCEL_SIGMA",
                    help="own-pose Kalman: tag noise std [m], accel noise [m/s^2] (default 0.03,8; noisier tags -> raise the first)")
    ap.add_argument("--max-speed", type=float, default=None,
                    help="safety cap on the commanded xyz velocity norm, fraction of rc 100 (default sqrt(3) = env "
                         "per-axis parity; 1.0 = max 1 m/s in any direction)")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    map_info = load_map(args.map)
    frame = Frame(args.map_convention, args.origin_height, map_info["tags"])
    geofence = args.geofence or default_geofence(map_info, frame)
    if args.site:
        site = args.site
    elif map_info.get("site"):
        site = tuple(float(x) for x in map_info["site"])
    else:
        site = ((geofence[0] + geofence[1]) / 2, (geofence[2] + geofence[3]) / 2, SITE_HEIGHT / 2)
    cfg = Config(defenders=[s.strip() for s in args.defenders.split(",") if s.strip()], site=site,
                 safety=SafetyCfg(geofence=geofence))
    if args.max_speed is not None:
        cfg.safety.max_speed = float(args.max_speed)
    if args.kalman:
        cfg.kalman = KalmanCfg(meas_sigma=args.kalman[0], accel_sigma=args.kalman[1])
    if args.policy == "rl":
        if not args.weights:
            print("--policy rl needs --weights FILE.npz", file=sys.stderr)
            return 2
        policy = MLPPolicy(args.weights)
    else:
        policy = AssignLeadPolicy() if args.policy.startswith("assign-lead") else HoverPolicy()
    kw = {"frame_in": args.generic_frame} if args.format in ("generic", "auto") else {}
    adapters = [make_adapter(args.format, frame, **kw)]
    log = open(args.log, "w", buffering=1) if args.log else None
    recorder = Recorder(args.record) if args.record else None
    print(f"defend_live: policy {policy.name}, defenders {cfg.defenders}, convention {frame.convention}, "
          f"site {np.round(cfg.site, 2).tolist()}, geofence {np.round(geofence, 2).tolist()}", flush=True)

    # ---- replay -----------------------------------------------------------------------------
    if args.replay:
        rows = read_recording(args.replay)
        if not rows:
            print(f"{args.replay}: no datagrams", file=sys.stderr)
            return 2
        links = {n: NullLink(n) for n in cfg.defenders}
        for l in links.values():
            l.flying = True  # pretend airborne so the full pipeline runs
        runner = DefendLive(cfg, policy, adapters, links, log=log, verbose=not args.quiet)
        t, k, n = rows[0][0], 0, 0
        while k < len(rows) and (not args.ticks or n < args.ticks):
            batch = []
            while k < len(rows) and rows[k][0] <= t:
                batch.append(rows[k])
                k += 1
            runner.tick(t, batch)
            t += DT
            n += 1
        rep = replay_report(rows, frame, cfg.defenders)
        rep["ticks"] = runner.ticks
        rep["safety_reasons"] = runner.stats["reasons"]
        rep["obs_dim"] = OBS_DIM
        print(json.dumps(rep, indent=1), flush=True)
        return 0

    # ---- env-parity closed loop (AS-Defend-Live-v0 stage K) ---------------------------------
    if args.dry_run and args.feed.startswith("stage:"):
        mk = (lambda: MLPPolicy(args.weights)) if args.policy == "rl" else type(policy)
        rep = sim_eval(int(args.feed[6:]), mk, episodes=args.episodes, seed=args.seed, frame=frame, site=cfg.site,
                       defenders=cfg.defenders, kalman=cfg.kalman, max_speed=args.max_speed)
        rep["policy"] = policy.name
        print(json.dumps(rep, indent=1), flush=True)
        return 0

    # ---- dry-run / live ---------------------------------------------------------------------
    world = None
    sources = []
    for spec in args.feed.split(","):
        spec = spec.strip()
        if spec == "synthetic":
            world = SyntheticWorld(frame, cfg.site, cfg.defenders, n_attackers=args.attackers)
        elif spec.startswith("udp:"):
            sources.append(UdpSource(int(spec[4:]), recorder=recorder))
        else:
            print(f"unknown --feed {spec!r}", file=sys.stderr)
            return 2
    if args.live:
        if not args.i_know_this_flies or world is not None:
            print("--live needs --i-know-this-flies and a real feed (not synthetic)", file=sys.stderr)
            return 2
        TelloLink.allow = True
        ifaces = dongle_map(args.dongles)
        links = {n: TelloLink(n, ifaces[n], 9000 + i) for i, n in enumerate(cfg.defenders)}
    else:
        links = {n: NullLink(n, world, i) for i, n in enumerate(cfg.defenders)}
    runner = DefendLive(cfg, policy, adapters, links, log=log, verbose=not args.quiet)

    def kill(why):
        if not runner.killed:
            print(f"KILL ({why}): landing all", flush=True)
        runner.killed = True
        runner.land_all()

    signal.signal(signal.SIGINT, lambda *_: kill("ctrl-c"))
    KeyboardKill(kill)
    if args.live:
        for l in links.values():
            l.takeoff()
            time.sleep(0.3)
        time.sleep(4.0)
    n = 0
    sim_t = 0.0
    try:
        while not runner.killed and (not args.ticks or n < args.ticks):
            tick_start = time.monotonic()
            if world is not None:
                world.step(DT)
                sim_t += DT
                now = sim_t
                batch = [(now, world.datagram())]
                if recorder:
                    recorder.write(now, batch[0][1])
            else:
                now = time.monotonic()
                batch = [b for s in sources for b in s.poll(now)]
            runner.tick(now, batch)
            n += 1
            if world is None or args.realtime:
                time.sleep(max(0.0, DT - (time.monotonic() - tick_start)))
    finally:
        if runner.killed or args.live:
            runner.land_all()
            if args.live:
                for _ in range(5):  # land is fire-and-forget UDP: repeat
                    runner.land_all()
                    time.sleep(0.2)
    if world is not None:
        print(f"synthetic: {n} ticks, attackers downed {world.downed}/{len(world.att_pos)}, "
              f"reasons {runner.stats['reasons']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
