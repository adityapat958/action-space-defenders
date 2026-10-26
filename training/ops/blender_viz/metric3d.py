"""Metric 3D plots (matplotlib) of the same recordings the Blender renderer uses (shared loader viz_data.py).

    python ops/blender_viz/metric3d.py --config ops/blender_viz/configs/t3_defend.json [--mp4] [--png] [--fps 20]
    python ops/blender_viz/metric3d.py --all --png --mp4          # every config in configs/

Equal-aspect axes in metres (sim frame, z up), red attackers / green defenders + RL drones, amber goals / gates /
interceptions. PNG = whole episode (full paths, start o, end ., capture x) with an x/y/z panel and a top view.
MP4 = time-synchronised playback with history trails and a slow orbit. Numbers on screen come from the recording.
"""
import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import viz_data  # noqa: E402

RGB = {"attacker": "#e3120b", "defender": "#0fb526", "agent": "#0fb526"}
TGT = {"goal": "#ffa000", "gate": "#ff8c00", "pad": "#9a9a9a"}
CAP = "#f2b51c"


def nan_breaks(P, segs):
    """Concatenate segments with NaN rows between them so teleports are never drawn as lines."""
    out = []
    for i0, i1 in segs:
        out.append(P[i0:i1 + 1]); out.append(np.full((1, 3), np.nan))
    return np.concatenate(out) if out else np.empty((0, 3))


def set_equal(ax, lo, hi):
    c, r = (lo + hi) / 2, max((hi - lo).max() / 2, 0.25)
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(0, max(hi[2] * 1.1, 2 * r * 0.5))
    ax.set_box_aspect((1, 1, max(hi[2] * 1.1, r) / (2 * r)))
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")


def draw_regions(ax, S, top=False):
    a = np.linspace(0, 2 * np.pi, 120)
    for r in S["regions"]:
        c, R = np.asarray(r["center"]), r["radius"]
        x, y = c[0] + R * np.cos(a), c[1] + R * np.sin(a)
        if top:
            ls = "--" if r["kind"] == "ring" else "-"
            col = RGB["defender"] if r["kind"] == "site" else "#8aa0c8" if r["kind"] == "keepout" else "#888"
            ax.plot(x, y, ls, color=col, lw=1.2, alpha=0.8)
            if r["kind"] == "site":
                ax.fill(x, y, color=col, alpha=0.12)
            continue
        if r["kind"] == "site":
            for z in (0, r["height"] or 0):
                ax.plot(x, y, z, color=RGB["defender"], lw=1.2, alpha=0.7)
            for k in range(0, 120, 15):
                ax.plot([x[k]] * 2, [y[k]] * 2, [0, r["height"] or 0], color=RGB["defender"], lw=0.6, alpha=0.35)
        elif r["kind"] == "keepout":
            for el in np.radians([0, 30, 60]):
                ax.plot(c[0] + R * np.cos(el) * np.cos(a), c[1] + R * np.cos(el) * np.sin(a), R * np.sin(el) + 0 * a,
                        color="#8aa0c8", lw=0.8, alpha=0.6)
            for az in np.radians(np.arange(0, 180, 30)):
                e = np.linspace(0, np.pi, 60)
                ax.plot(c[0] + R * np.cos(e) * np.cos(az), c[1] + R * np.cos(e) * np.sin(az), R * np.sin(e),
                        color="#8aa0c8", lw=0.6, alpha=0.4)
        elif r["kind"] == "touchdown":
            ax.plot(x, y, 0 * a, color="#bbbbbb", lw=1.0)
        elif r["kind"] == "ring":
            ax.plot(x, y, 0 * a, "--", color="#888888", lw=0.8, alpha=0.7)


def static(S, out_png):
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    top = fig.add_subplot(2, 2, 2)
    ts = fig.add_subplot(2, 2, 4)
    lo, hi = S["bounds"]
    draw_regions(ax, S); draw_regions(top, S, top=True)
    t = S["t"]
    for g in S["targets"]:
        P = g["pos"]
        if g.get("moving"):
            ax.plot(*P.T, color=TGT[g["kind"]], lw=1.0, alpha=0.8)
            top.plot(P[:, 0], P[:, 1], color=TGT[g["kind"]], lw=1.0, alpha=0.8)
        else:
            m = "o" if g["kind"] == "gate" else "s"
            ax.scatter(*P[0], color=TGT[g["kind"]], marker=m, s=30)
            top.scatter(P[0, 0], P[0, 1], color=TGT[g["kind"]], marker=m, s=25)
    labelled = set()
    for a in S["agents"]:
        col = RGB[a["role"]]
        L = nan_breaks(a["pos"], a["segments"])
        lab = None if a["role"] in labelled else {"attacker": "attackers", "defender": "defenders", "agent": "RL drone(s)"}[a["role"]]
        labelled.add(a["role"])
        ax.plot(*L.T, color=col, lw=1.4, label=lab)
        top.plot(L[:, 0], L[:, 1], color=col, lw=1.2)
        for i0, i1 in a["segments"]:
            ax.scatter(*a["pos"][i0], color=col, marker="o", s=22, facecolors="none")
            top.scatter(*a["pos"][i0, :2], color=col, marker="o", s=18, facecolors="none")
        ax.scatter(*a["pos"][a["segments"][-1][1]], color=col, s=12)
        z = np.where(a["visible"] & ~a["downed"], a["pos"][:, 2], np.nan)
        ts.plot(t, z, color=col, lw=1.0, alpha=0.9)
    caps = [e for e in S["events"] if e["kind"] == "capture"]
    if caps:
        C = np.array([e["pos"] for e in caps])
        ax.scatter(*C.T, color=CAP, marker="X", s=60, edgecolors="k", linewidths=0.4, label=f"interceptions ({len(caps)})", zorder=5)
        top.scatter(C[:, 0], C[:, 1], color=CAP, marker="X", s=50, edgecolors="k", linewidths=0.4, zorder=5)
        for e in caps:
            ts.axvline(e["step"] * S["dt"], color=CAP, lw=0.8, alpha=0.7)
    for e in S["events"]:
        if e["kind"] in ("gate", "landed"):
            ts.axvline(e["step"] * S["dt"], color=TGT["gate"] if e["kind"] == "gate" else "#555", lw=0.6, alpha=0.6, ls=":")
    set_equal(ax, np.asarray(lo, float), np.asarray(hi, float))
    ax.view_init(elev=28, azim=-60)
    ax.legend(loc="upper left", fontsize=9)
    top.set_aspect("equal"); top.grid(alpha=0.3); top.set_xlabel("x [m]"); top.set_ylabel("y [m]"); top.set_title("top view")
    ts.set_xlabel("sim time [s]"); ts.set_ylabel("altitude z [m]"); ts.grid(alpha=0.3)
    ts.set_title("altitude (amber = interception, dotted = gate / landing)" if S["events"] else "altitude")
    if S["targets"] and S["targets"][0]["kind"] == "goal":
        d = np.linalg.norm(S["agents"][0]["pos"] - S["targets"][0]["pos"], axis=-1)
        tw = ts.twinx(); tw.plot(t, d, color=TGT["goal"], lw=1.0); tw.set_ylabel("distance to goal [m]", color=TGT["goal"])
    fig.suptitle(f"{S['title']}\n{S['source']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png); plt.close(fig)
    print(f"[png] {out_png}", flush=True)


def anim(S, out_mp4, fps=20, speed=1.0, orbit=40.0):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_subplot(projection="3d")
    lo, hi = np.asarray(S["bounds"][0], float), np.asarray(S["bounds"][1], float)
    T, dt = S["T"], S["dt"]
    n = int((T - 1) * dt / speed * fps) + 1 + fps  # +1 s hold
    idx = np.minimum((np.arange(n) * speed / fps / dt).round().astype(int), T - 1)
    draw_regions(ax, S)
    set_equal(ax, lo, hi)
    lines, dots, drops = [], [], []
    for a in S["agents"]:
        col = RGB[a["role"]]
        lines.append(ax.plot([], [], [], color=col, lw=1.6)[0])
        dots.append(ax.plot([], [], [], "o", color=col, ms=7, mec="k", mew=0.5)[0])
        drops.append(ax.plot([], [], [], color=col, lw=0.6, alpha=0.4)[0])
    tg = []
    for g in S["targets"]:
        m = {"goal": "*", "gate": "o", "pad": "s"}[g["kind"]]
        tg.append((g, ax.plot([], [], [], m, color=TGT[g["kind"]], ms=9 if m == "*" else 6, mfc="none" if m == "o" else TGT[g["kind"]])[0]))
    gt = [g for g in S["targets"] if g.get("moving")]
    gl = [ax.plot([], [], [], color=TGT["goal"], lw=0.9, alpha=0.8)[0] for _ in gt]
    caps = [e for e in S["events"] if e["kind"] == "capture"]
    cap_sc = ax.plot([], [], [], "X", color=CAP, ms=9, mec="k", mew=0.4)[0]
    txt = fig.text(0.02, 0.95, "", fontsize=12, family="monospace")
    fig.text(0.02, 0.98, S["title"], fontsize=12, weight="bold", va="top")
    fig.text(0.02, 0.02, S["source"] + "   |   axes in metres, sim frame (z up)", fontsize=8, color="#555")

    def upd(k):
        i = idx[k]
        for a, L, D, Z in zip(S["agents"], lines, dots, drops):
            segs = [(i0, min(i1, i)) for i0, i1 in a["segments"] if i0 <= i]
            P = nan_breaks(a["pos"], segs)
            L.set_data(P[:, 0], P[:, 1]); L.set_3d_properties(P[:, 2])
            if a["visible"][i]:
                p = a["pos"][i]
                D.set_data([p[0]], [p[1]]); D.set_3d_properties([p[2]])
                D.set_color("#555555" if a["downed"][i] else RGB[a["role"]])
                Z.set_data([p[0], p[0]], [p[1], p[1]]); Z.set_3d_properties([0, p[2]])
            else:
                D.set_data([], []); D.set_3d_properties([]); Z.set_data([], []); Z.set_3d_properties([])
        for g, M in tg:
            p = g["pos"][i]
            M.set_data([p[0]], [p[1]]); M.set_3d_properties([p[2]])
            if g.get("reached") is not None and g["reached"][i]:
                M.set_color(RGB["agent"])
        for g, L in zip(gt, gl):
            P = g["pos"][: i + 1]
            L.set_data(P[:, 0], P[:, 1]); L.set_3d_properties(P[:, 2])
        C = np.array([e["pos"] for e in caps if e["step"] <= i]).reshape(-1, 3)
        cap_sc.set_data(C[:, 0], C[:, 1]); cap_sc.set_3d_properties(C[:, 2])
        txt.set_text(f"t = {i * dt:6.2f} s   {S['hud'][i]}")
        ax.view_init(elev=26, azim=-60 + orbit * k / max(1, n - 1))
        return []

    ani = animation.FuncAnimation(fig, upd, frames=n, blit=False)
    ani.save(out_mp4, writer=animation.FFMpegWriter(fps=fps, bitrate=4000, extra_args=["-pix_fmt", "yuv420p"]))
    plt.close(fig)
    print(f"[mp4] {out_mp4} ({n} frames)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="*", default=[])
    p.add_argument("--all", action="store_true")
    p.add_argument("--png", action="store_true")
    p.add_argument("--mp4", action="store_true")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--out", default="docs/results/metric3d")
    a = p.parse_args()
    cfgs = sorted(glob.glob(os.path.join(HERE, "configs", "*.json"))) if a.all else a.config
    out = a.out if os.path.isabs(a.out) else os.path.join(viz_data.REPO, a.out)
    os.makedirs(out, exist_ok=True)
    for c in cfgs:
        cfg = json.load(open(c))
        npz = cfg["npz"] if os.path.isabs(cfg["npz"]) else os.path.join(viz_data.REPO, cfg["npz"])
        S = viz_data.load(npz, env=cfg.get("env", 0))
        if cfg.get("title"):
            S["title"] = cfg["title"]
        if a.png:
            static(S, os.path.join(out, f"{cfg['name']}_metric3d.png"))
        if a.mp4:
            anim(S, os.path.join(out, f"{cfg['name']}_metric3d.mp4"), fps=a.fps, speed=float(cfg.get("speed", 1.0)))


if __name__ == "__main__":
    main()
