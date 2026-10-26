"""Load any Action Space trajectory .npz into one scene description (used by render3d.py and blender_points.py).

Supported files:
  - basic   (ops/turing_score_basic.py --traj): pos/goal (T, E, 3), quat, done, dt, meta
  - defend  (scripts/evaluate.py --traj):        defenders (E, T, D, 3), attackers (E, T, A, 3), alive (E, T, A), dt, site_radius ...
  - multi   (generic): agents (T, E, N, 3) [+ targets (T, E, M, 3)], dt, meta

Scene for env e: dict(dt, title, groups=[dict(name, color, pos (T, N, 3), alive (T, N) bool or None, size, trail)],
                     discs=[dict(center (3,), radius, color)], bounds=(lo (3,), hi (3,)))
Numpy only, so it also runs inside Blender's Python.
"""
import json

import numpy as np

BLUE, RED, GREEN, GREY = (0.15, 0.35, 0.95), (0.9, 0.15, 0.15), (0.1, 0.7, 0.2), (0.5, 0.5, 0.5)


def _meta(d):
    if "meta" in d.files:
        try:
            return json.loads(str(d["meta"]))
        except Exception:
            pass
    return {}


def load(path, env=0):
    d = np.load(path, allow_pickle=True)
    meta = _meta(d)
    dt = float(d["dt"]) if "dt" in d.files else float(meta.get("dt", 1 / 60))
    groups, discs = [], []
    if "defenders" in d.files:  # defend
        dfn = d["defenders"][env]  # (T, D, 3)
        att = d["attackers"][env]
        alive = d["alive"][env].astype(bool)
        groups.append(dict(name="defenders", color=BLUE, pos=dfn, alive=None, size=1.0, trail=1.5))
        groups.append(dict(name="attackers", color=RED, pos=att, alive=alive, size=1.0, trail=1.0))
        site_r = float(d["site_radius"]) if "site_radius" in d.files else 0.5
        discs.append(dict(center=np.zeros(3), radius=site_r, color=GREEN))
        if "spawn_radius" in d.files:
            discs.append(dict(center=np.zeros(3), radius=float(d["spawn_radius"]), color=GREY, ring=True))
        label = str(d["label"]) if "label" in d.files else "defend"
        title = f"Defend · {label} {str(d['overrides']) if 'overrides' in d.files else ''}".strip()
    elif "pos" in d.files:  # basic single drone + goal
        groups.append(dict(name="drone", color=BLUE, pos=d["pos"][:, env][:, None], alive=None, size=1.0, trail=2.0))
        groups.append(dict(name="goal", color=RED, pos=d["goal"][:, env][:, None], alive=None, size=0.6, trail=2.0))
        title = f"{meta.get('task', '')} · {meta.get('label', '')}".replace("logs/skrl/", "")
    elif "agents" in d.files:
        groups.append(dict(name="agents", color=BLUE, pos=d["agents"][:, env], alive=None, size=1.0, trail=2.0))
        if "targets" in d.files:
            groups.append(dict(name="targets", color=RED, pos=d["targets"][:, env], alive=None, size=0.8, trail=1.0))
        if "gates" in d.files:  # approach: per-drone gate waypoint + landing pad (static per episode)
            groups.append(dict(name="gates", color=(1.0, 0.55, 0.0), pos=d["gates"][:, env], alive=None, size=0.7, trail=0.0))
        if "pads" in d.files:
            groups.append(dict(name="pads", color=GREEN, pos=d["pads"][:, env], alive=None, size=0.7, trail=0.0))
        title = meta.get("title", path)
    else:
        raise ValueError(f"unknown trajectory schema: {d.files}")
    # bounds from live points only (the defend env parks downed attackers far away, e.g. z = 30 m)
    allp = np.concatenate([(g["pos"] if g["alive"] is None else np.where(g["alive"][..., None], g["pos"], np.nan)).reshape(-1, 3)
                           for g in groups])
    allp = allp[np.isfinite(allp).all(-1)]
    lo, hi = allp.min(0), allp.max(0)
    for c in discs:
        lo[:2] = np.minimum(lo[:2], -c["radius"]); hi[:2] = np.maximum(hi[:2], c["radius"])
    lo[2] = min(lo[2], 0.0)
    return dict(dt=dt, title=title, groups=groups, discs=discs, bounds=(lo, hi), T=groups[0]["pos"].shape[0], meta=meta)
