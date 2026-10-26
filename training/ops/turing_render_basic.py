"""Render T1/T2 trajectory .npz files (from ops/turing_score_basic.py --traj) with matplotlib. CPU only, no Isaac.

    python ops/turing_render_basic.py a.npz [b.npz ...] --out compare.mp4   # columns = files, rows = envs
    python ops/turing_render_basic.py a.npz --png summary.png               # static: top-down paths + distance vs time

Top-down (x-y) view; drone = blue dot with trail, goal = red x with trail, arrow = heading (yaw).
Text: distance to goal and altitude. A grey flash marks episode resets.
"""
import argparse, json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("traj", nargs="+")
p.add_argument("--out", default=None, help="mp4 path")
p.add_argument("--png", default=None, help="static summary png path")
p.add_argument("--envs", type=int, default=2)
p.add_argument("--fps", type=int, default=20)
p.add_argument("--trail", type=float, default=2.0)
a = p.parse_args()

data = [np.load(f, allow_pickle=True) for f in a.traj]
metas = [json.loads(str(d["meta"])) for d in data]
labels = []
for m in metas:
    lab = m["label"]
    if "/" in lab:  # checkpoint path -> run name + checkpoint
        parts = lab.split("/")
        run = parts[-3]; lab = run.split("_ppo_torch_")[-1] + " " + run[5:16] + "/" + parts[-1]
    labels.append(lab)
dt = float(data[0]["dt"])
T = min(d["pos"].shape[0] for d in data)
E = min(a.envs, *(d["pos"].shape[1] for d in data))
yaw = [2 * np.arctan2(d["quat"][..., 3], d["quat"][..., 0]) for d in data]  # wxyz, small roll/pitch
dist = [np.linalg.norm(d["goal"] - d["pos"], axis=-1) for d in data]


def lims(d, e):
    xy = np.concatenate([d["pos"][:T, e, :2], d["goal"][:T, e, :2]])
    xy = xy[np.isfinite(xy).all(-1)]
    c = xy.mean(0) if len(xy) else np.zeros(2)
    r = max(np.abs(xy - c).max() if len(xy) else 1.0, 0.5) + 0.3
    return c, r


if a.png:
    fig, ax = plt.subplots(2, len(data), figsize=(4.5 * len(data), 8), squeeze=False)
    for j, d in enumerate(data):
        for e in range(E):
            ax[0, j].plot(*d["pos"][:T, e, :2].T, lw=1, label=f"env {e}")
            ax[0, j].plot(*d["goal"][:T, e, :2].T, "r--", lw=0.8)
            ax[1, j].plot(np.arange(T) * dt, dist[j][:T, e], lw=1)
        ax[0, j].set_title(labels[j], fontsize=8); ax[0, j].set_aspect("equal"); ax[0, j].set_xlabel("x [m]")
        ax[1, j].set_xlabel("t [s]"); ax[1, j].set_ylabel("dist to goal [m]"); ax[1, j].axhline(0.2, c="g", ls=":")
        ax[1, j].set_ylim(0, min(np.nanmax(dist[j][:T, :E]) * 1.1 if np.isfinite(dist[j][:T, :E]).any() else 1, 5))
    fig.suptitle(metas[0]["task"] + "  (solid = drone, dashed red = goal)")
    fig.tight_layout(); fig.savefig(a.png, dpi=110); print("wrote", a.png)

if a.out:
    from matplotlib.animation import FFMpegWriter

    stride = max(1, round(1.0 / (a.fps * dt)))
    ntrail = int(a.trail / dt)
    fig, ax = plt.subplots(E, len(data), figsize=(4 * len(data), 4 * E), squeeze=False)
    w = FFMpegWriter(fps=a.fps)
    with w.saving(fig, a.out, dpi=90):
        for t in range(0, T, stride):
            for j, d in enumerate(data):
                for e in range(E):
                    x = ax[e, j]; x.cla()
                    c, r = lims(d, e)
                    x.set_xlim(c[0] - r, c[0] + r); x.set_ylim(c[1] - r, c[1] + r); x.set_aspect("equal")
                    s = max(0, t - ntrail)
                    x.plot(*d["goal"][s:t + 1, e, :2].T, "r-", lw=0.8, alpha=0.5)
                    x.plot(*d["pos"][s:t + 1, e, :2].T, "b-", lw=1.2, alpha=0.6)
                    x.plot(*d["goal"][t, e, :2], "rx", ms=10)
                    px, py = d["pos"][t, e, :2]
                    x.plot(px, py, "bo", ms=8)
                    if np.isfinite(yaw[j][t, e]):
                        x.arrow(px, py, 0.25 * np.cos(yaw[j][t, e]), 0.25 * np.sin(yaw[j][t, e]), head_width=0.06, color="b")
                    if d["done"][max(t - stride, 0):t + 1, e].any():
                        x.set_facecolor("0.85")
                    x.set_title(f"{labels[j]} env{e}\nt={t * dt:4.1f}s d={dist[j][t, e]:.2f}m z={d['pos'][t, e, 2]:.2f}m", fontsize=8)
            w.grab_frame()
    print("wrote", a.out)
