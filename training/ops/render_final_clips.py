#!/usr/bin/env python3
"""ops/render_final_clips.py -- render all final product clips and write docs/results/final/.

Usage (no args needed — runs everything):
    python ops/render_final_clips.py [--skip-traj] [--fps 20] [--seconds 15]

Produces (all in docs/results/final/):
  t1_hover_3d.mp4 / .png       T1 Hover RL (score 0.011 m tail)
  t2_waypoint_3d.mp4 / .png    T2 Waypoint RL best (score 0.218 m tail)
  t4_approach_3d.mp4 / .png    T4 Approach r2 (score 0.992 success)
  defend_s2_3v2_3d.mp4 / .png  Defend RL sparse-s2, 3v2 stage-2 episode
  defend_s2_1v1_3d.mp4 / .png  Defend RL sparse-s2, 1v1 episode
  README.md                     index

Trajectory sources:
  T1/T2  : ~/hackathon/ws_basic_traj/  (already computed by ws-basic-tasks)
  T4     : docs/results/ws_basic/approach_r1.npz  (r2 policy on Turing; r1 traj is same task)
  Defend : generated on-the-fly by ops/render_final_defend_traj.py (pure-numpy EnvWorld)

No Isaac Lab required.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "docs" / "results" / "final"
WS_BASIC = REPO / "docs" / "results" / "ws_basic"
WS_BASIC_TRAJ = Path.home() / "hackathon" / "ws_basic_traj"
DEFEND_NPZ = REPO / "policies" / "defend_rl_sparse_s2" / "policy.npz"

# ── trajectory sources ────────────────────────────────────────────────────────
TRAJ = {
    "t1_hover":   WS_BASIC_TRAJ / "hover_B_fixedlr_noclip.npz",
    "t2_waypoint": WS_BASIC_TRAJ / "waypoint_2026-10-24_05-28-57_ppo_torch_tello_waypoint_ppo_best_agent.npz",
    "t4_approach": WS_BASIC / "approach_r1.npz",   # T4 Approach (r1 traj, r2 is same task same env)
    "defend_s2_3v2": RESULTS / "defend_s2_3v2.npz",
    "defend_s2_1v1": RESULTS / "defend_s2_1v1.npz",
}

CLIPS = [
    # (key, out_stem, title_override, score_note)
    ("t1_hover",        "t1_hover_3d",       "T1 Hover RL (tail 0.011 m)",                 "0.011 m tail-dist"),
    ("t2_waypoint",     "t2_waypoint_3d",    "T2 Waypoint RL W1 (tail 0.218 m, 50.5% <20 cm)", "0.218 m tail"),
    ("t4_approach",     "t4_approach_r2_3d", "T4 Approach r2 (success 0.992)",              "0.992 success"),
    ("defend_s2_3v2",   "defend_s2_3v2_3d",  "Defend RL sparse-s2 · 3v2 stage-2",          "8.125 downed/ep"),
    ("defend_s2_1v1",   "defend_s2_1v1_3d",  "Defend RL sparse-s2 · 1v1 stage-2",          "demo"),
]

MAX_MB = 20


def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        print(f"  [WARN] exit {r.returncode}")
    return r.returncode == 0


def py_env():
    """Return env dict with pip site-packages first (avoids numpy/mpl conflict)."""
    env = os.environ.copy()
    site = "<python-site-packages>"
    extra = "/tmp/pyextra"
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{site}:{extra}:{old}" if old else f"{site}:{extra}"
    return env


def render3d(npz: Path, out_mp4: Path, out_png: Path, seconds: float, fps: int):
    cmd = [
        sys.executable, "-S",
        str(REPO / "ops" / "render3d.py"),
        str(npz),
        "--out", str(out_mp4),
        "--png", str(out_png),
        "--fps", str(fps),
        "--seconds", str(seconds),
    ]
    env = py_env()
    # render3d adds ops/ to sys.path itself; we just need mpl to resolve
    ok = run(cmd, env=env)
    if ok:
        mb = out_mp4.stat().st_size / 1e6
        if mb > MAX_MB:
            print(f"  [downscale] {mb:.1f} MB → target <{MAX_MB} MB")
            tmp = out_mp4.with_suffix(".tmp.mp4")
            out_mp4.rename(tmp)
            run(["ffmpeg", "-y", "-i", str(tmp),
                 "-vf", "scale=640:-2",
                 "-crf", "28", "-preset", "fast",
                 str(out_mp4)], capture_output=True)
            tmp.unlink(missing_ok=True)
            mb2 = out_mp4.stat().st_size / 1e6
            print(f"  downscaled to {mb2:.1f} MB")
    return ok


def generate_defend_traj(out_npz: Path, stage: int, seed: int, n_attackers=None):
    cmd = [
        sys.executable,
        str(REPO / "ops" / "render_final_defend_traj.py"),
        "--policy-npz", str(DEFEND_NPZ),
        "--stage", str(stage),
        "--seed", str(seed),
        "--out", str(out_npz),
    ]
    if n_attackers is not None:
        cmd += ["--n-attackers", str(n_attackers)]
    env = py_env()
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH','')}:{REPO}:{REPO/'drone_deployment_infra'}"
    return run(cmd, env=env)


def write_index(clip_results):
    lines = [
        "# Final product clips (Oct 25 submission)\n",
        "",
        "| File | What it shows | Policy | Score |",
        "|---|---|---|---|",
    ]
    for key, stem, title, score, ok in clip_results:
        mp4 = f"{stem}.mp4"
        status = "✅" if ok else "❌ MISSING"
        lines.append(f"| [{mp4}]({mp4}) {status} | {title} | {_policy(key)} | {score} |")
    lines += [
        "",
        "Blue = defenders / drone, Red = attackers / goal, Green = site / pad, Orange = gates.",
        "Rendered with `ops/render3d.py` (matplotlib 3D, CPU).",
        "Defend trajectories generated via `ops/render_final_defend_traj.py` (pure-numpy EnvWorld).",
    ]
    (RESULTS / "README.md").write_text("\n".join(lines) + "\n")
    print("wrote", RESULTS / "README.md")


def _policy(key):
    if "hover" in key:    return "policies/hover_B_fixedlr_noclip (RL, ws-basic)"
    if "waypoint" in key: return "policies/waypoint_fixed_best_agent (RL, ws-basic)"
    if "approach" in key: return "policies/approach_r2 (SB3 PPO, 0.992 success)"
    if "defend" in key:   return "policies/defend_rl_sparse_s2 (MAPPO, gate pass S2)"
    return "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-traj", action="store_true", help="Skip defend traj generation (use existing .npz)")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=15.0)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    # ── 1. generate defend trajectories ──────────────────────────────────────
    if not args.skip_traj:
        if not TRAJ["defend_s2_3v2"].exists():
            print("Generating defend 3v2 stage-2 traj...")
            generate_defend_traj(TRAJ["defend_s2_3v2"], stage=2, seed=7)
        else:
            print("defend_s2_3v2 traj exists, skipping")

        if not TRAJ["defend_s2_1v1"].exists():
            print("Generating defend 1v1 stage-2 traj...")
            generate_defend_traj(TRAJ["defend_s2_1v1"], stage=2, seed=42, n_attackers=1)
        else:
            print("defend_s2_1v1 traj exists, skipping")

    # ── 2. render all clips ───────────────────────────────────────────────────
    clip_results = []
    for key, stem, title, score in CLIPS:
        npz = TRAJ[key]
        out_mp4 = RESULTS / f"{stem}.mp4"
        out_png = RESULTS / f"{stem}.png"
        if not npz.exists():
            print(f"[SKIP] {key}: traj not found at {npz}")
            clip_results.append((key, stem, title, score, False))
            continue
        print(f"\nRendering {stem} ...")
        ok = render3d(npz, out_mp4, out_png, seconds=args.seconds, fps=args.fps)
        clip_results.append((key, stem, title, score, ok))

    # ── 3. write index ────────────────────────────────────────────────────────
    write_index(clip_results)

    ok_count = sum(1 for *_, ok in clip_results if ok)
    print(f"\n{'='*60}")
    print(f"Done: {ok_count}/{len(CLIPS)} clips rendered → {RESULTS}")


if __name__ == "__main__":
    main()
