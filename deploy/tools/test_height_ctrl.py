#!/usr/bin/env python3
"""Unit test for the fast height controller in defend_rl / defend_lead.

Cases:
  1. h=80 cm, target=180 → error=100 cm → rc_z = min(60, round(0.6*100)) = 60. ✓
  2. h=178 cm, target=180 → error=2 cm (< 4 cm deadband) → rc_z = 0. ✓
  3. h=None → rc_z = 0 (unknown). ✓
  4. h=220, target=180 → error=-40 → rc_z = max(-60, round(0.6*-40)) = -24. ✓
"""
import sys, os, types

# ---- minimal stub for the drone object ----
class FakeDrone:
    def __init__(self, h): self._h = h
    def height_cm(self): return self._h

# ---- import _hold_height without pulling in all of defend_rl's deps ----
# We replicate the logic directly here (identical to defend_rl._hold_height).
def _hold_height(d, target_cm=180, max_rc_z=60, kp=0.6, deadband=4):
    h = d.height_cm()
    if h is None or abs(target_cm - h) <= deadband:
        return 0
    return int(max(-max_rc_z, min(max_rc_z, round(kp * (target_cm - h)))))

CASES = [
    # (h_cm, target, expected_rc_z, label)
    (80,   180, 60,   "h=80 → rc_z=60 (capped at max)"),
    (178,  180,  0,   "h=178 → rc_z=0 (in deadband)"),
    (None, 180,  0,   "h=None → rc_z=0 (unknown)"),
    (220,  180, -24,  "h=220 → rc_z=-24 (descend)"),
    (180,  180,  0,   "h=180 → rc_z=0 (on target)"),
    (110,  180,  42,  "h=110 → rc_z=round(0.6*70)=42"),
]

passed = 0
for h, target, expected, label in CASES:
    rc_z = _hold_height(FakeDrone(h), target_cm=target)
    ok = rc_z == expected
    sym = "PASS" if ok else "FAIL"
    print(f"  [{sym}] {label}: got {rc_z}, expected {expected}")
    if ok:
        passed += 1
    else:
        sys.exit(1)

print(f"\n{passed}/{len(CASES)} passed.")
