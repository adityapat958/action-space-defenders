# Copyright (c) 2026, Action Space. SPDX-License-Identifier: BSD-3-Clause
"""Render AS-Defend trajectories (from `evaluate.py --traj`) to an mp4 with matplotlib -- CPU only, no Vulkan.

    python scripts/render_defend_traj.py outputs/defend/traj_chase.npz outputs/defend/traj_assign-lead.npz \
        --out outputs/videos/defend_compare.mp4

Several .npz files render side by side (one column each, same env rows), so baselines run with the
same --seed can be compared episode for episode. Top-down view: blue = defenders with trails,
red = live attackers (grey x = downed), shaded disc = protected site, dashed ring = spawn radius.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

p = argparse.ArgumentParser()
p.add_argument("traj", nargs="+")
p.add_argument("--out", default="outputs/videos/defend.mp4")
p.add_argument("--envs", type=int, default=2, help="Rows: how many envs per file.")
p.add_argument("--fps", type=int, default=20)
p.add_argument("--trail", type=float, default=2.0, help="Trail length [s].")
a = p.parse_args()

data = [np.load(f, allow_pickle=True) for f in a.traj]
dt = float(data[0]["dt"])
stride = max(1, round(1.0 / (a.fps * dt)))
T = min(d["alive"].shape[1] for d in data)
rows, cols = min(a.envs, min(d["alive"].shape[0] for d in data)), len(data)
R = float(data[0]["spawn_radius"]) * 1.15

fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)
arts = []
for c, d in enumerate(data):
    for r in range(rows):
        ax = axes[r][c]
        ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.add_patch(plt.Circle((0, 0), float(d["site_radius"]), color="gold", alpha=0.35))
        ax.add_patch(plt.Circle((0, 0), float(d["spawn_radius"]), fill=False, ls="--", color="grey", lw=0.8))
        if r == 0:
            ax.set_title(f"{d['label']}" + (f"\n{d['overrides']}" if str(d["overrides"]) else ""), fontsize=9)
        D, A = d["defenders"].shape[2], d["attackers"].shape[2]
        arts.append(dict(
            d=d, r=r,
            dtrail=[ax.plot([], [], "-", color="tab:blue", alpha=0.4, lw=1)[0] for _ in range(D)],
            atrail=[ax.plot([], [], "-", color="tab:red", alpha=0.3, lw=1)[0] for _ in range(A)],
            dpts=ax.scatter(np.zeros(D), np.zeros(D), c="tab:blue", s=40, zorder=3),
            apts=ax.scatter(np.zeros(A), np.zeros(A), c="tab:red", s=40, marker="^", zorder=3),
            txt=ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=8),
            downed=0,
        ))
fig.tight_layout()
tl = int(a.trail / dt)
writer = FFMpegWriter(fps=a.fps, bitrate=2400)
with writer.saving(fig, a.out, dpi=100):
    for t in range(0, T, stride):
        for s in arts:
            d, r = s["d"], s["r"]
            dp, ap, al = d["defenders"][r], d["attackers"][r], d["alive"][r]
            t0 = max(0, t - tl)
            for i, ln in enumerate(s["dtrail"]):
                ln.set_data(dp[t0:t + 1, i, 0], dp[t0:t + 1, i, 1])
            for j, ln in enumerate(s["atrail"]):
                m = al[t0:t + 1, j]
                ln.set_data(np.where(m, ap[t0:t + 1, j, 0], np.nan), np.where(m, ap[t0:t + 1, j, 1], np.nan))
            s["dpts"].set_offsets(dp[t, :, :2])
            s["apts"].set_offsets(ap[t, :, :2])
            s["apts"].set_color(np.where(al[t], "tab:red", "lightgrey"))
            if t > 0:
                s["downed"] += int((al[t - stride] & ~al[t]).sum()) if t >= stride else 0
            s["txt"].set_text(f"t={t * dt:4.1f}s  downed={s['downed']}")
        writer.grab_frame()
print(f"[render] wrote {a.out} ({T * dt:.0f} s, {len(data)} columns x {rows} rows)")
