"""Copy a skrl MAPPO checkpoint with every agent's policy log_std reset to log(STD).

Why: warm-starting stage k+1 from stage k carries a collapsed std (0.07 by S2), so the policy can't explore the
new stage. Usage: python ops/dsp_reset_std.py IN.pt OUT.pt [STD=0.4]
"""
import math
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
std = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
ck = torch.load(src, map_location="cpu", weights_only=False)
for agent, parts in ck.items():
    p = parts["policy"]
    old = p["log_std_parameter"].exp().tolist()
    p["log_std_parameter"] = torch.full_like(p["log_std_parameter"], math.log(std))
    print(f"{agent}: std {[round(x, 3) for x in old]} -> {std}")
torch.save(ck, dst)
