"""Summarise a defend_rl JSONL run log (team4 defend_log.py format).

    python3 team4/tools/analyze_defend_log.py path/to/defend_rl_YYYYMMDD_HHMMSS.jsonl [--timeline 6]

Per drone, it prints:
- pose coverage, i.e. the share of ticks with src != none (no pose means the hover fallback rc 0,0,z,0)
- pose windows
- frozen poses (repeated x/y) and position jumps (implied x/y speed > 2.5 m/s)
- attacker z range: z < 0 means the map z and the Tello height z are different frames
- action saturation (|a| > 0.95)
- an optional timeline: every N-th tick shows t, src, pos, yaw (from obs sin/cos), attacker 0, action and rc
"""
import argparse
import collections
import json
import math

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("log")
p.add_argument("--timeline", type=int, default=0, help="print every N-th tick per drone (0 = off)")
a = p.parse_args()
rows = [json.loads(l.replace("Infinity", "1e9")) for l in open(a.log) if l.strip()]
meta = [r for r in rows if r.get("meta")]
R = [r for r in rows if not r.get("meta")]
print("meta:", meta[0] if meta else None)
print(f"ticks: {len(R)}")
by = collections.defaultdict(list)
for r in R:
    by[r["drone"]].append(r)
for d, rs in sorted(by.items()):
    t = np.array([r["t"] for r in rs])
    has = np.array([bool(r["own"]["pos"]) for r in rs])
    A = np.array([r["action"] for r in rs])
    print(f"\n== {d}: {len(rs)} ticks, t {t[0]:.1f}..{t[-1]:.1f} s, src {dict(collections.Counter(r['own']['src'] for r in rs))}")
    print(f"   no-pose (hover fallback): {100 * (~has).mean():.0f}% of ticks")
    win, s = [], None
    for ti, ok in zip(t, has):
        if ok and s is None:
            s = ti
        if not ok and s is not None:
            win.append((round(float(s), 1), round(float(ti), 1))); s = None
    if s is not None:
        win.append((round(float(s), 1), round(float(t[-1]), 1)))
    print(f"   pose windows: {win}")
    if has.sum() > 1:
        P = np.array([r["own"]["pos"] for r in rs if r["own"]["pos"]])
        tp = t[has]
        step = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)
        v = step / np.maximum(np.diff(tp), 1e-3)
        ages = np.array([r["own"]["age"] for r in rs if r["own"]["pos"]])
        print(f"   pose age median {np.median(ages):.3f} s, max {ages.max():.2f} s; repeated x/y {100 * (step < 1e-6).mean():.0f}%; "
              f"jumps > 2.5 m/s at t = {tp[1:][v > 2.5].round(1).tolist()}")
        print(f"   pos first {P[0].round(2)} last {P[-1].round(2)} min {P.min(0).round(2)} max {P.max(0).round(2)}")
    print(f"   action mean {A.mean(0).round(2)}  |a|>0.95 share {(np.abs(A) > 0.95).mean(0).round(2)}  (vx, vy, vz, yaw_rate)")
att = [x["pos"] for r in R for x in r["attackers"]]
if att:
    att = np.array(att)
    print(f"\nattackers: x/y/z min {att.min(0).round(2)} max {att.max(0).round(2)}  (z < 0 = below the floor -> frame mismatch with own Tello height)")
    print(f"ticks with no attackers: {sum(1 for r in R if not r['attackers'])}")
if a.timeline:
    for d, rs in sorted(by.items()):
        print(f"\n-- {d} timeline: t src pos yaw_deg att0 action rc")
        for k, r in enumerate(rs):
            if k % a.timeline and k != len(rs) - 1:
                continue
            o = r["obs"]
            pos = np.round(r["own"]["pos"], 2) if r["own"]["pos"] else "---"
            att0 = np.round(r["attackers"][0]["pos"], 2) if r["attackers"] else "-"
            print(f"{r['t']:5.1f} {r['own']['src']:7s} {pos} yaw {math.degrees(math.atan2(o[9], o[10])):+6.0f} att0 {att0} "
                  f"a {np.round(r['action'], 2)} rc {r['rc']}")
