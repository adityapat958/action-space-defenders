#!/usr/bin/env python3
"""replay_log.py — summarise and analyse defend_rl / defend_lead JSONL logs.

Usage:
  python3 replay_log.py <logfile.jsonl>               # summary to stdout
  python3 replay_log.py <logfile.jsonl> --csv out.csv # also dump trajectory CSV

Summary output:
  - Run metadata (weights, git commit, t_start)
  - Ticks per drone, total duration
  - Stale % (hover_stale ticks)
  - RC saturation % (rc_capped ticks)
  - Min distance to each attacker slot (ever seen)
  - Attacker pose source breakdown

CSV columns (one row per tick):
  t, drone, own_x, own_y, own_z, own_age, own_src,
  atk0_x, atk0_y, atk0_z, atk0_age, ... (up to 4 slots),
  rc_lr, rc_fb, rc_ud, rc_yaw, hover_stale, rc_capped
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def load(path: str):
    meta = None
    ticks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("meta"):
                meta = obj
            else:
                ticks.append(obj)
    return meta, ticks


def summarise(meta, ticks, file=sys.stdout):
    def p(*a, **kw):
        print(*a, **kw, file=file)

    p("=" * 60)
    if meta:
        p(f"weights   : {meta.get('weights', '?')}")
        p(f"sha256    : {meta.get('weights_sha256', '?')[:16]}...")
        p(f"git       : {meta.get('git_commit', '?')}")
        p(f"t_start   : {meta.get('t_start', '?')}")
    p(f"total ticks: {len(ticks)}")
    if not ticks:
        p("(empty log)")
        return

    by_drone: dict[str, list] = defaultdict(list)
    for tk in ticks:
        by_drone[tk.get("drone", "?")].append(tk)

    for drone, dts in sorted(by_drone.items()):
        ts = [t.get("t", 0) for t in dts]
        dur = max(ts) - min(ts) if len(ts) > 1 else 0.0
        n_stale = sum(1 for t in dts if t.get("flags", {}).get("hover_stale"))
        n_capped = sum(1 for t in dts if t.get("flags", {}).get("rc_capped"))
        pct_stale = 100.0 * n_stale / len(dts)
        pct_capped = 100.0 * n_capped / len(dts)
        p(f"\n--- {drone} ({len(dts)} ticks, {dur:.1f} s) ---")
        p(f"  hover_stale : {pct_stale:.1f}%")
        p(f"  rc_capped   : {pct_capped:.1f}%")

        src_counts: dict[str, int] = defaultdict(int)
        for t in dts:
            src_counts[t.get("own", {}).get("src", "?")] += 1
        p(f"  pose sources: {dict(src_counts)}")

        # min distance to attackers
        min_dists: dict[int, float] = {}
        for t in dts:
            own = t.get("own", {}).get("pos")
            if own is None:
                continue
            for i, atk in enumerate(t.get("attackers", [])):
                ap = atk.get("pos")
                if ap is None:
                    continue
                d = math.sqrt(sum((a - b) ** 2 for a, b in zip(own, ap)))
                if i not in min_dists or d < min_dists[i]:
                    min_dists[i] = d
        for i, md in sorted(min_dists.items()):
            p(f"  min dist to attacker[{i}]: {md:.3f} m")

    p("=" * 60)


def dump_csv(ticks, out_path: str):
    ATK_SLOTS = 4
    fieldnames = [
        "t", "drone", "own_x", "own_y", "own_z", "own_age", "own_src",
    ]
    for i in range(ATK_SLOTS):
        fieldnames += [f"atk{i}_x", f"atk{i}_y", f"atk{i}_z", f"atk{i}_age"]
    fieldnames += ["rc_lr", "rc_fb", "rc_ud", "rc_yaw", "hover_stale", "rc_capped"]

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for tk in ticks:
            own = tk.get("own", {})
            pos = own.get("pos") or [None, None, None]
            rc = tk.get("rc", [0, 0, 0, 0])
            flags = tk.get("flags", {})
            row: dict = {
                "t": tk.get("t"),
                "drone": tk.get("drone"),
                "own_x": pos[0] if len(pos) > 0 else None,
                "own_y": pos[1] if len(pos) > 1 else None,
                "own_z": pos[2] if len(pos) > 2 else None,
                "own_age": own.get("age"),
                "own_src": own.get("src"),
                "rc_lr": rc[0] if len(rc) > 0 else None,
                "rc_fb": rc[1] if len(rc) > 1 else None,
                "rc_ud": rc[2] if len(rc) > 2 else None,
                "rc_yaw": rc[3] if len(rc) > 3 else None,
                "hover_stale": flags.get("hover_stale"),
                "rc_capped": flags.get("rc_capped"),
            }
            atks = tk.get("attackers", [])
            for i in range(ATK_SLOTS):
                if i < len(atks):
                    ap = atks[i].get("pos") or [None, None, None]
                    row[f"atk{i}_x"] = ap[0] if len(ap) > 0 else None
                    row[f"atk{i}_y"] = ap[1] if len(ap) > 1 else None
                    row[f"atk{i}_z"] = ap[2] if len(ap) > 2 else None
                    row[f"atk{i}_age"] = atks[i].get("age")
                else:
                    row[f"atk{i}_x"] = row[f"atk{i}_y"] = row[f"atk{i}_z"] = row[f"atk{i}_age"] = None
            w.writerow(row)
    print(f"CSV written → {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="JSONL log file from defend_rl / defend_lead")
    ap.add_argument("--csv", metavar="OUT.csv", help="also write trajectory CSV")
    args = ap.parse_args()

    meta, ticks = load(args.log)
    summarise(meta, ticks)

    if args.csv:
        dump_csv(ticks, args.csv)


if __name__ == "__main__":
    main()
