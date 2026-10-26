"""Generate defend trajectory .npz files using defend_live.py's pure-numpy EnvWorld sim.

No Isaac Lab required.  Runs the RL MLP (from policy.npz) through the closed-loop
sim_eval harness while recording positions each tick.

Outputs (defend schema expected by ops/traj_io.py):
  defenders  (1, T, D, 3)   – env axis is always 1 (single episode)
  attackers  (1, T, A, 3)
  alive      (1, T, A)      bool
  site_radius float
  spawn_radius float
  dt         float
  label      str
  overrides  str
"""
from __future__ import annotations

import argparse
import json
import sys
import os
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ensure defend_live can be imported (it's a standalone file)
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DLI = REPO / "drone_deployment_infra"
sys.path.insert(0, str(DLI))
sys.path.insert(0, str(REPO))

from defend_live import (  # noqa: E402
    DT, EnvWorld, Frame, SafetyCfg, KalmanCfg, Config, NullLink,
    DefendLive, make_adapter, MLPPolicy,
    SITE_HEIGHT, OBS_DIM,
)

SITE = np.array([3.0, 0.0, SITE_HEIGHT / 2])
DEFENDERS = ("TELLO-1", "TELLO-2", "TELLO-3")


def make_frame():
    return Frame("ydown_zin", 1.5, {10: (np.eye(3), np.zeros(3))})


def record_episode(stage: int, make_policy, seed: int = 0, max_steps: int = 2000,
                   n_attackers: int | None = None):
    """Run one episode and return arrays of shape (T, D/A, 3)."""
    frame = make_frame()
    world = EnvWorld(frame, SITE, list(DEFENDERS), stage=stage, seed=seed)
    if n_attackers is not None:
        # EnvWorld._brain_velocity hardcodes A=4 so we cannot truly reduce the pool.
        # Instead park all but `n_attackers` outside the arena from the start.
        rng = world.rng
        A_total = len(world.att_pos)
        active = min(n_attackers, A_total)
        # park excess attackers very far away (> geofence; alive=False after first ram)
        for k in range(active, A_total):
            world.att_pos[k] = np.array([50.0, 50.0, 0.0])  # outside arena
            world.att_alive[k] = False

    r = world.arena_radius
    gf = (SITE[0] - r, SITE[0] + r, SITE[1] - r, SITE[1] + r, 0.15, 4.0)
    sc = SafetyCfg(geofence=gf)
    sc.yaw_rate_max = 1.0
    cfg = Config(defenders=list(DEFENDERS), site=tuple(SITE), safety=sc, kalman=KalmanCfg())
    links = {n: NullLink(n, world, i) for i, n in enumerate(DEFENDERS)}
    runner = DefendLive(cfg, make_policy(), [make_adapter("telemetry", frame)], links)

    D = len(DEFENDERS)
    A_slots = len(world.att_pos)
    def_hist, att_hist, alive_hist = [], [], []
    yaw_hist, dalive_hist, t_hist = [], [], []

    # (render) event capture without touching the sim: wrap _events/_spawn_wave on this instance only, so a capture
    # that clears a wave (and respawns it inside the same step) is still recorded at the capture point.
    events, _ev = [], {}
    _orig_events, _orig_spawn = world._events, world._spawn_wave

    def _spawn_wrapped():
        _ev["pre_spawn"] = (world.att_pos.copy(), world.att_alive.copy())
        _orig_spawn()

    def _events_wrapped():
        alive0 = world.att_alive.copy()
        _ev.pop("pre_spawn", None)
        _orig_events()
        pos1, alive1 = _ev.get("pre_spawn", (world.att_pos.copy(), world.att_alive.copy()))
        for k in np.flatnonzero(alive0 & ~alive1):
            dd = np.linalg.norm(world.def_pos - pos1[k], axis=-1)
            events.append(dict(step=len(def_hist), t=world.t, kind="capture", attacker=int(k),
                               defender=int(np.argmin(np.where(world.def_alive, dd, 1e9))), pos=pos1[k].tolist()))
        if "pre_spawn" in _ev:
            events.append(dict(step=len(def_hist), t=world.t, kind="wave", attacker=-1, defender=-1, pos=[0, 0, 0]))
        if world.breached:
            events.append(dict(step=len(def_hist), t=world.t, kind="breach", attacker=-1, defender=-1, pos=[0, 0, 0]))

    world._events, world._spawn_wave = _events_wrapped, _spawn_wrapped

    t = 0.0
    step = 0
    while not world.done and step < max_steps:
        world.step(DT)
        t += DT
        for i, n in enumerate(DEFENDERS):
            if not world.def_alive[i]:
                links[n].flying = False
        runner.tick(t, [(t, world.datagram())])

        def_hist.append(world.def_pos.copy())  # (D, 3)
        att_hist.append(world.att_pos.copy())   # (A, 3)
        alive_hist.append(world.att_alive.copy())  # (A,)
        yaw_hist.append(world.def_yaw.copy())       # (D,) simulated defender yaw [rad]
        dalive_hist.append(world.def_alive.copy())  # (D,) False after a defender crash
        t_hist.append(t)
        step += 1

    T = len(def_hist)
    # Pad attacker arrays to 4 slots (traj_io expects fixed A)
    A_pad = max(4, A_slots)
    def_arr = np.stack(def_hist)  # (T, D, 3)
    att_raw = np.stack(att_hist)  # (T, A_slots, 3)
    alv_raw = np.stack(alive_hist).astype(bool)  # (T, A_slots)

    att_arr = np.zeros((T, A_pad, 3), np.float32)
    alv_arr = np.zeros((T, A_pad), bool)
    att_arr[:, :A_slots] = att_raw
    alv_arr[:, :A_slots] = alv_raw

    extra = dict(def_yaw=np.stack(yaw_hist).astype(np.float32), def_alive=np.stack(dalive_hist).astype(bool),
                 t=np.asarray(t_hist, np.float64), site_center=np.asarray(world.site, np.float32),
                 site_radius=np.float32(world.SITE_RADIUS), spawn_radius=np.float32(world.p["spawn_radius"]),
                 capture_radius=np.float32(world.p["capture_radius"]), arena_radius=np.float32(world.arena_radius),
                 downed=np.int32(world.downed), breached=np.bool_(world.breached), crashes=np.int32(world.crashes),
                 stage=np.int32(stage), events=json.dumps(events))
    return def_arr.astype(np.float32), att_arr, alv_arr, world.downed, world.breached, extra


def save_traj(out_path: str, def_arr, att_arr, alv_arr, label: str, overrides: str = "", extra=None):
    """Save in defend schema (single episode → env axis = 1)."""
    # extra (render): true site centre / breach radius / stage spawn + capture radius from EnvWorld
    # (the old hard-coded site_radius 0.5 / spawn 4.0 at the origin did not match the sim: site is at SITE, r 1.5)
    extra = dict(extra or {})
    for k in ("def_yaw", "def_alive"):
        if k in extra:
            extra[k] = extra[k][None]  # env axis
    extra.setdefault("site_radius", np.float32(0.5))
    extra.setdefault("spawn_radius", np.float32(4.0))
    np.savez_compressed(
        out_path,
        defenders=def_arr[None],   # (1, T, D, 3)
        attackers=att_arr[None],   # (1, T, A, 3)
        alive=alv_arr[None],       # (1, T, A)
        frame="EnvWorld sim frame (z-up, m), site at site_center",
        **extra,
        dt=np.float32(DT),
        label=label,
        overrides=overrides,
    )
    print(f"saved {out_path}  (T={def_arr.shape[0]}, D={def_arr.shape[1]}, A={att_arr.shape[1]})")


def main():
    ap = argparse.ArgumentParser(description="Record defend trajectory via pure-numpy EnvWorld.")
    ap.add_argument("--policy-npz", required=True, help="policies/<name>/policy.npz")
    ap.add_argument("--stage", type=int, default=2, help="EnvWorld curriculum stage (0–6)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-attackers", type=int, default=None, help="Override attacker count (e.g. 1 for 1v1)")
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument("--label", default=None)
    ap.add_argument("--max-steps", type=int, default=1500)
    args = ap.parse_args()

    weights_path = args.policy_npz
    label = args.label or f"defend_rl_sparse_s{args.stage}"
    overrides = f"stage={args.stage} seed={args.seed}"
    if args.n_attackers:
        overrides += f" n_att={args.n_attackers}"

    print(f"stage={args.stage} seed={args.seed} n_att={args.n_attackers}  policy={weights_path}")

    def make_policy():
        return MLPPolicy(weights_path)

    def_arr, att_arr, alv_arr, downed, breached, extra = record_episode(
        args.stage, make_policy, seed=args.seed,
        max_steps=args.max_steps, n_attackers=args.n_attackers,
    )
    print(f"  downed={downed}  breached={breached}  T={def_arr.shape[0]}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_traj(args.out, def_arr, att_arr, alv_arr, label, overrides, extra)


if __name__ == "__main__":
    main()
