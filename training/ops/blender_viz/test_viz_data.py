"""Numpy-only checks of the trajectory adapters (no Blender): python3 ops/blender_viz/test_viz_data.py"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import viz_data  # noqa: E402

REPO = viz_data.REPO
CFG = os.path.join(HERE, "configs")


def check(name, ok, msg=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {msg}")
    return ok


def main():
    ok = True
    for f in sorted(os.listdir(CFG)):
        cfg = json.load(open(os.path.join(CFG, f)))
        path = os.path.join(REPO, cfg["npz"])
        if not os.path.exists(path):
            ok &= check(f, False, f"missing {path}")
            continue
        S = viz_data.load(path, env=cfg.get("env", 0))
        raw = np.load(path, allow_pickle=True)
        # 1. positions are the recording, untouched (identity frame mapping)
        for a in S["agents"]:
            if "defenders" in raw.files:
                arr = raw["defenders"] if a["role"] == "defender" else raw["attackers"]
                k = int(a["name"].split("_")[1])
                ref = arr[cfg.get("env", 0), : S["T"], k]
            elif "pos" in raw.files:
                ref = raw["pos"][: S["T"], cfg.get("env", 0)]
            else:
                ref = raw["agents"][: S["T"], cfg.get("env", 0), int(a["name"].split("_")[1])]
            ok &= check(f"{f} {a['name']} positions identical", np.array_equal(a["pos"], ref.astype(float)))
            # 2. no drawn trail segment contains a teleport
            for i0, i1 in a["segments"]:
                if i1 > i0:
                    jump = np.linalg.norm(np.diff(a["pos"][i0:i1 + 1], axis=0), axis=-1).max()
                    if jump > viz_data.JUMP:
                        ok &= check(f"{f} {a['name']} segment {i0}-{i1}", False, f"jump {jump:.2f} m")
            if a["quat"] is not None:
                ok &= check(f"{f} {a['name']} unit quaternions", np.allclose(np.linalg.norm(a["quat"], axis=-1), 1, atol=1e-4))
        # 3. first episode only: no reset jump at the end
        # 4. defend: one capture event per downed attacker, site centre from the recording
        if "downed" in raw.files:
            n = sum(e["kind"] == "capture" for e in S["events"])
            ok &= check(f"{f} captures == EnvWorld.downed", n == int(raw["downed"]), f"{n} vs {int(raw['downed'])}")
            site = [r for r in S["regions"] if r["kind"] == "site"][0]
            ok &= check(f"{f} site centre", np.allclose(site["center"][:2], raw["site_center"][:2]), str(site["center"]))
        roles = {a["role"] for a in S["agents"]}
        print(f"       {S['title']} roles={sorted(roles)} T={S['T']} dt={S['dt']:.4f}")
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
