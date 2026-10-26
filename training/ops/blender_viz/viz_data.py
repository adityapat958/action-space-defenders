"""Simulation trajectory .npz -> one render-agnostic scene description (numpy only, runs inside Blender's Python).

Nothing here simulates, smooths or resamples: every array is the recorded sim state, sliced to one env/episode.
Frame convention of every supported recorder is Isaac/EnvWorld world-z-up, metres, positions relative to the env
origin (Defend EnvWorld: the sim frame around `site_center`). Blender is also z-up metres, so the mapping is identity.

Schemas (auto-detected):
  basic    ops/turing_score_basic.py --traj   T1 Hover / T2 Waypoint: pos, quat (wxyz), goal (T, E, 3), done, dt, meta
  approach ops/turing_play_traj.py --traj     T4 Approach: agents/gates/pads (T, E, N, 3) [+ quats, gate_passed, arrived, landed]
  defend   scripts/evaluate.py --traj (Isaac) or ops/render_final_defend_traj.py (defend_live EnvWorld):
           defenders (E, T, D, 3), attackers (E, T, A, 3), alive (E, T, A) [+ def_yaw, def_alive, site_center, events]

Scene = dict(task, title, source, dt, T, t (T,), agents, targets, regions, events, hud (per-step strings), bounds)
  agent  = dict(name, role 'attacker'|'defender'|'agent', pos (T,3), quat (T,4) wxyz or None, quat_source,
                visible (T,) bool, downed (T,) bool, segments [(i0, i1)] inclusive sample ranges with continuous motion)
  target = dict(name, kind 'goal'|'gate'|'pad', pos (T,3), visible (T,) bool, reached (T,) bool or None)
  region = dict(kind 'site'|'keepout'|'touchdown'|'ring', center (3,), radius, height, label)
  event  = dict(step, kind 'capture'|'breach'|'wave'|'gate'|'landed'|'crash', agent, other, pos (3,))
"""
import json
import math
import os
import re

import numpy as np

JUMP = 0.3  # [m] between consecutive samples => teleport (reset / respawn / parking), never drawn as motion
FALL_JUMP = 1.0  # [m] same for a downed (free-falling) attacker: <= 0.6 m per 74 ms step from 3 m; respawns jump >= 3 m
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def _meta(d):
    try:
        return json.loads(str(d["meta"])) if "meta" in d.files else {}
    except Exception:
        return {}


def _scalar(d, k, default=None):
    return d[k].item() if k in d.files else default


def segments(pos, visible):
    """Split sample indices into runs of continuous motion: visible and no teleport between neighbours."""
    T = len(pos)
    step = np.r_[np.inf, np.linalg.norm(np.diff(pos, axis=0), axis=-1)]
    out, start = [], None
    for i in range(T):
        ok = bool(visible[i]) and np.isfinite(pos[i]).all()
        if ok and start is not None and step[i] > JUMP:
            out.append((start, i - 1)); start = None
        if ok and start is None:
            start = i
        if not ok and start is not None:
            out.append((start, i - 1)); start = None
    if start is not None:
        out.append((start, T - 1))
    return out


def _cfg_value(rel_path, name):
    """Fallback for old recordings without a value in meta: read `name = <float>` from the env cfg source."""
    try:
        src = open(os.path.join(REPO, rel_path)).read()
        m = re.search(rf"^\s*{name}\s*(?::[^=]+)?=\s*([0-9.]+)", src, re.M)
        return float(m.group(1)) if m else None
    except OSError:
        return None


def _episode_end(done, T):
    """Last sample of the first episode. Recorders log the state after env.step(), and Isaac resets inside step(), so
    the sample flagged `done` is already the next episode's spawn (verified: 0.9 m jump into it) -> stop one before."""
    idx = np.flatnonzero(done)
    return int(idx[0]) - 1 if len(idx) and idx[0] > 0 else T - 1


def _yaw_quat(yaw):
    return np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], -1)


# ---------------------------------------------------------------------------------------------------------------
def load_basic(d, env, first_episode=True):
    meta = _meta(d)
    T = d["pos"].shape[0]
    end = _episode_end(d["done"][:, env], T) if first_episode else T - 1
    sl = slice(0, end + 1)
    pos, quat, goal = d["pos"][sl, env].astype(float), d["quat"][sl, env].astype(float), d["goal"][sl, env].astype(float)
    T = len(pos)
    vis = np.ones(T, bool)
    task = meta.get("task", "")
    agent = dict(name="drone", role="agent", pos=pos, quat=quat, quat_source="sim (Isaac root_quat_w)",
                 visible=vis, downed=np.zeros(T, bool), segments=segments(pos, vis))
    moving_goal = np.ptp(goal, axis=0).max() > 1e-4
    targets = [dict(name="goal", kind="goal", pos=goal, visible=vis, reached=None, moving=bool(moving_goal))]
    err = np.linalg.norm(pos - goal, axis=-1)
    hud = [f"distance to goal {e:5.3f} m" for e in err]
    short = {"AS-Tello-Hover-v0": "T1 Hover", "AS-Tello-Waypoint-v0": "T2 Waypoint"}.get(task, task)
    label = meta.get("label", "").replace("logs/skrl/", "")
    return dict(task=short, title=f"{short} - RL policy", source=f"Isaac Lab {task} - {label} - seed {meta.get('seed', '?')} env {env}",
                dt=float(_scalar(d, "dt", meta.get("dt", 1 / 60))), T=T, agents=[agent], targets=targets, regions=[],
                events=[], hud=hud, meta=meta)


def load_approach(d, env, first_episode=True):
    meta = _meta(d)
    P = d["agents"][:, env].astype(float)  # (T, N, 3)
    T0, N = P.shape[:2]
    end = _episode_end(d["done"][:, env], T0) if first_episode else T0 - 1
    T = end + 1
    P = P[:T]
    gates, pads = d["gates"][:T, env].astype(float), d["pads"][:T, env].astype(float)
    quats = d["quats"][:T, env].astype(float) if "quats" in d.files else None
    landed = d["landed"][:T, env].astype(bool) if "landed" in d.files else None
    gpass = d["gate_passed"][:T, env].astype(bool) if "gate_passed" in d.files else None
    agents, targets, events = [], [], []
    for k in range(N):
        vis = np.ones(T, bool)
        agents.append(dict(name=f"drone_{k}", role="agent", pos=P[:, k], quat=None if quats is None else quats[:, k],
                           quat_source="sim (Isaac root_quat_w)" if quats is not None else "none recorded",
                           visible=vis, downed=np.zeros(T, bool), segments=segments(P[:, k], vis)))
        targets.append(dict(name=f"gate_{k}", kind="gate", pos=gates[:, k], visible=vis,
                            reached=None if gpass is None else gpass[:, k], moving=False))
        targets.append(dict(name=f"pad_{k}", kind="pad", pos=pads[:, k], visible=vis,
                            reached=None if landed is None else landed[:, k], moving=False))
        for flag, kind in ((gpass, "gate"), (landed, "landed")):
            if flag is not None:
                for i in np.flatnonzero(flag[1:, k] & ~flag[:-1, k]) + 1:
                    events.append(dict(step=int(i), kind=kind, agent=f"drone_{k}", other=None, pos=P[i, k]))
    cfg = "simulation_infra/source/action_space_kit/action_space_kit/tasks/direct/tello_approach/tello_approach_env_cfg.py"
    r_keep = meta.get("keepout_radius") or _cfg_value(cfg, "keepout_radius")
    r_goal = meta.get("goal_radius") or _cfg_value(cfg, "goal_radius")
    regions = []
    if r_keep:
        regions.append(dict(kind="keepout", center=np.zeros(3), radius=float(r_keep), height=None, label="keep-out sphere"))
    if r_goal:
        regions.append(dict(kind="touchdown", center=np.zeros(3), radius=float(r_goal), height=None, label="touchdown zone"))
    hud = []
    for i in range(T):
        s = []
        if gpass is not None:
            s.append(f"gates {int(gpass[i].sum())}/{N}")
        if landed is not None:
            s.append(f"landed {int(landed[i].sum())}/{N}")
        hud.append("   ".join(s))
    if T < T0:  # the next sample is the reset: the env reported done here (success or abort is not in the recording)
        hud[-1] += "   episode done"
    ck = meta.get("checkpoint", "").split("logs/sb3/")[-1]
    return dict(task="T4 Approach", title="T4 Approach - shared PPO policy, 4 drones",
                source=f"Isaac Lab AS-Tello-Approach-v0 - {ck} - seed {meta.get('seed', '?')} env {env}",
                dt=float(_scalar(d, "dt", meta.get("dt", 0.05))), T=T, agents=agents, targets=targets, regions=regions,
                events=sorted(events, key=lambda e: e["step"]), hud=hud, meta=meta)


def load_defend(d, env, first_episode=True):
    Dp, Ap, alive = d["defenders"][env].astype(float), d["attackers"][env].astype(float), d["alive"][env].astype(bool)
    T, D = Dp.shape[:2]
    A = Ap.shape[1]
    dt = float(_scalar(d, "dt", 0.075))
    site = np.asarray(d["site_center"], float) if "site_center" in d.files else np.zeros(3)
    site_r = float(_scalar(d, "site_radius", 1.5))
    envworld = "site_center" in d.files
    site_h = 2.5  # tello_defend_env_cfg.site_height == defend_live.SITE_HEIGHT
    def_alive = d["def_alive"][env].astype(bool) if "def_alive" in d.files else np.ones((T, D), bool)
    yaw = d["def_yaw"][env].astype(float) if "def_yaw" in d.files else None
    agents = []
    for i in range(D):
        vis = np.ones(T, bool)
        agents.append(dict(name=f"defender_{i}", role="defender", pos=Dp[:, i],
                           quat=None if yaw is None else _yaw_quat(yaw[:, i]),
                           quat_source="sim yaw (EnvWorld def_yaw; velocity model, no tilt)" if yaw is not None else "none recorded",
                           visible=vis, downed=~def_alive[:, i], segments=segments(Dp[:, i], def_alive[:, i])))
    for k in range(A):
        if not alive[:, k].any():
            continue  # slot never active in this episode (stage masks it)
        # a downed attacker stays drawn (falling, dark) until it teleports (respawn / Isaac parking)
        step = np.r_[0.0, np.linalg.norm(np.diff(Ap[:, k], axis=0), axis=-1)]
        vis = alive[:, k].copy()
        for i in range(1, T):
            if not alive[i, k] and vis[i - 1] and step[i] <= FALL_JUMP:
                vis[i] = True
        agents.append(dict(name=f"attacker_{k}", role="attacker", pos=Ap[:, k], quat=None, quat_source="none (position-only)",
                           visible=vis, downed=vis & ~alive[:, k], segments=segments(Ap[:, k], alive[:, k])))
    events = []
    if "events" in d.files:  # exact capture points (recorder wraps EnvWorld._events)
        for e in json.loads(str(d["events"])):
            if e["kind"] == "capture":
                events.append(dict(step=e["step"], kind="capture", agent=f"attacker_{e['attacker']}",
                                   other=f"defender_{e['defender']}", pos=np.asarray(e["pos"], float)))
            elif e["kind"] == "breach":
                events.append(dict(step=e["step"], kind="breach", agent=None, other=None, pos=site))
    else:  # Isaac: alive True -> False; last live sample is the capture point
        for k in range(A):
            for i in np.flatnonzero(alive[:-1, k] & ~alive[1:, k]) + 1:
                p = Ap[i - 1, k]
                j = int(np.argmin(np.linalg.norm(Dp[i - 1] - p, axis=-1)))
                events.append(dict(step=int(i), kind="capture", agent=f"attacker_{k}", other=f"defender_{j}", pos=p))
    for i in range(D):
        for s in np.flatnonzero(def_alive[:-1, i] & ~def_alive[1:, i]) + 1:
            events.append(dict(step=int(s), kind="crash", agent=f"defender_{i}", other=None, pos=Dp[s, i]))
    events.sort(key=lambda e: e["step"])
    regions = [dict(kind="site", center=np.array([site[0], site[1], 0.0]), radius=site_r, height=site_h, label="protected site")]
    if "spawn_radius" in d.files:
        regions.append(dict(kind="ring", center=np.array([site[0], site[1], 0.0]), radius=float(d["spawn_radius"]), height=None,
                            label="attacker spawn ring"))
    caps = np.zeros(T, int)
    for e in events:
        if e["kind"] == "capture":
            caps[e["step"]:] += 1
    n_att = int(alive.any(0).sum())
    hud = [f"intercepted {c}" + ("   BREACH" if any(e['kind'] == 'breach' and e['step'] <= i for e in events) else "")
           for i, c in enumerate(caps)]
    label = str(_scalar(d, "label", "defend"))
    over = str(_scalar(d, "overrides", ""))
    src = "defend_live EnvWorld closed loop (deploy pipeline)" if envworld else "Isaac Lab AS-Defend-Live-v0 (scripts/evaluate.py)"
    return dict(task="T3 Defend", title=f"T3 Defend - {label} - {D} defenders vs {n_att} attacker slots",
                source=f"{src} - {over} - env {env}", dt=dt, T=T, agents=agents, targets=[], regions=regions,
                events=events, hud=hud, meta=dict(label=label, overrides=over, capture_radius=_scalar(d, "capture_radius")))


def load(path, env=0, first_episode=True):
    d = np.load(path, allow_pickle=True)
    if "defenders" in d.files:
        S = load_defend(d, env, first_episode)
    elif "pos" in d.files:
        S = load_basic(d, env, first_episode)
    elif "agents" in d.files:
        S = load_approach(d, env, first_episode)
    else:
        raise ValueError(f"unknown trajectory schema in {path}: {d.files}")
    S["path"] = os.path.abspath(path)
    S["t"] = np.arange(S["T"]) * S["dt"]
    pts = [a["pos"][a["visible"]] for a in S["agents"]] + [t["pos"] for t in S["targets"]]
    pts += [np.array([r["center"] + [r["radius"], r["radius"], 0], r["center"] - [r["radius"], r["radius"], 0]])
            for r in S["regions"] if r["kind"] != "ring"]
    P = np.concatenate([p.reshape(-1, 3) for p in pts])
    P = P[np.isfinite(P).all(-1)]
    lo, hi = P.min(0), P.max(0)
    lo[2] = min(lo[2], 0.0)
    S["bounds"] = (lo, hi)
    return S


def summary(S):
    lines = [f"{S['title']}  [{S['source']}]", f"  dt {S['dt']:.4f} s, {S['T']} samples = {(S['T'] - 1) * S['dt']:.2f} s sim time"]
    for a in S["agents"]:
        lines.append(f"  {a['name']:12s} {a['role']:9s} visible {a['visible'].sum():4d}/{S['T']} segments {len(a['segments'])} "
                     f"orientation: {a['quat_source']}")
    for r in S["regions"]:
        lines.append(f"  region {r['kind']:9s} centre {np.round(r['center'], 3).tolist()} r {r['radius']}")
    kinds = {}
    for e in S["events"]:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    lines.append(f"  events {kinds}")
    lo, hi = S["bounds"]
    lines.append(f"  bounds {np.round(lo, 2).tolist()} .. {np.round(hi, 2).tolist()}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        print(summary(load(p)))
