"""3D matplotlib render of any trajectory .npz (see ops/traj_io.py): drones as point objects with short trails.

    python ops/render3d.py run.npz --env 0 --out run3d.mp4 --png run3d.png
    python ops/render3d.py a.npz b.npz --out compare3d.mp4      # side by side, same env index

Blue = our drones/defenders, red = goal/attackers (grey x when downed), green disc = protected site.
The camera slowly orbits so depth reads in 2D video. CPU only.
"""
import argparse, os, sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traj_io import load  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("traj", nargs="+")
p.add_argument("--env", type=int, default=0)
p.add_argument("--out", default=None)
p.add_argument("--png", default=None)
p.add_argument("--fps", type=int, default=20)
p.add_argument("--seconds", type=float, default=None, help="Cap the clip length.")
p.add_argument("--orbit", type=float, default=40.0, help="Degrees of camera orbit over the clip.")
p.add_argument("--titles", nargs="*", default=None)
a = p.parse_args()

scenes = [load(f, a.env) for f in a.traj]
titles = a.titles or [s["title"] for s in scenes]
dt = scenes[0]["dt"]
T = min(s["T"] for s in scenes)
if a.seconds:
    T = min(T, int(a.seconds / dt))
fig = plt.figure(figsize=(5.2 * len(scenes), 5.4))
axes = [fig.add_subplot(1, len(scenes), i + 1, projection="3d") for i in range(len(scenes))]


def setup(ax, s, title):
    lo, hi = s["bounds"]
    c = (lo + hi) / 2
    r = max((hi - lo)[:2].max() / 2, 0.6) + 0.2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(0, max(hi[2] + 0.3, 1.0))
    ax.set_box_aspect((1, 1, max(hi[2] + 0.3, 1.0) / (2 * r)))
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(title, fontsize=8, pad=-2)
    th = np.linspace(0, 2 * np.pi, 80)
    for dsc in s["discs"]:
        x = dsc["center"][0] + dsc["radius"] * np.cos(th); y = dsc["center"][1] + dsc["radius"] * np.sin(th)
        if dsc.get("ring"):
            ax.plot(x, y, 0 * th, ls="--", color=dsc["color"], lw=0.8)
        else:
            ax.plot_trisurf(np.r_[dsc["center"][0], x], np.r_[dsc["center"][1], y], np.zeros(81), color=dsc["color"], alpha=0.25)


def draw(ax, s, title, t, full=False):
    ax.cla(); setup(ax, s, title)
    for g in s["groups"]:
        P = g["pos"]; n = P.shape[1]
        k0 = 0 if full else max(0, t - int(g["trail"] / dt))
        for j in range(n):
            seg = P[k0:t + 1, j]
            ok = np.isfinite(seg).all(-1)
            if g["alive"] is not None:
                ok &= g["alive"][k0:t + 1, j]
            seg = np.where(ok[:, None], seg, np.nan)
            jump = np.r_[False, np.linalg.norm(np.diff(seg, axis=0), axis=-1) > 0.3]  # episode reset / respawn
            seg[jump] = np.nan
            if ok.sum() > 1:
                ax.plot(*seg.T, color=g["color"], lw=1.0 if not full else 0.8, alpha=0.6)
            x = P[t, j]
            if not np.isfinite(x).all():
                continue
            if g["alive"] is not None and not g["alive"][t, j]:
                continue  # downed: hidden (the env parks it far away)
            ax.scatter(*x, s=45 * g["size"], color=g["color"], marker="o", depthshade=False)
            ax.plot([x[0], x[0]], [x[1], x[1]], [0, x[2]], color=g["color"], lw=0.4, alpha=0.3)  # drop line = height cue
    txt = f"t = {t * dt:4.1f} s"
    for g in s["groups"]:
        if g["alive"] is not None:
            al = g["alive"][: t + 1]
            txt += f"   {g['name']} downed: {int((al[:-1] & ~al[1:]).sum())}"
    ax.text2D(0.02, 0.95, txt, transform=ax.transAxes, fontsize=8)


if a.png:
    for ax, s, ti in zip(axes, scenes, titles):
        draw(ax, s, ti, T - 1, full=True); ax.view_init(elev=28, azim=-60)
    fig.tight_layout(); fig.savefig(a.png, dpi=110); print("wrote", a.png)

if a.out:
    from matplotlib.animation import FFMpegWriter

    stride = max(1, round(1.0 / (a.fps * dt)))
    frames = list(range(0, T, stride))
    w = FFMpegWriter(fps=a.fps, bitrate=2400)
    with w.saving(fig, a.out, dpi=100):
        for i, t in enumerate(frames):
            for ax, s, ti in zip(axes, scenes, titles):
                draw(ax, s, ti, t); ax.view_init(elev=28, azim=-60 + a.orbit * i / max(len(frames) - 1, 1))
            w.grab_frame()
    print("wrote", a.out, f"({len(frames)} frames)")
