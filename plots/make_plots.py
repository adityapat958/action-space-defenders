"""Regenerate the summary figures for the report. Numbers come from STATUS.md section 4 and policies/*/README.md.
Run: python3 submission/plots/make_plots.py   (writes into submission/plots/ and submission/report/figures/)"""
import os, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "report", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREY, GREEN = "#1f77b4", "#d62728", "#7f7f7f", "#2ca02c"


def save(fig, name):
    fig.tight_layout()
    for d in (HERE, FIG):
        fig.savefig(os.path.join(d, name), dpi=200)
    plt.close(fig)


# 1. Defend: closed-loop stage check (defend_live --dry-run --feed stage:K, 64 episodes)
st = np.arange(4)
rl = [8.86, 9.08, 8.20, 4.77]; rl_e = [0.24, 0.33, 0.19, 0.36]
lead = [6.94, 8.45, 8.00, 7.92]
s1 = [8.42, 7.69, 5.14, 0.80]
breach_rl = [0, 0, 0, 48]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [2.2, 1]})
w = 0.27
ax[0].bar(st - w, rl, w, yerr=rl_e, color=BLUE, label="RL sparse_s2 (ours, primary)", capsize=3)
ax[0].bar(st, lead, w, color=GREY, label="assign-lead-safe (scripted fallback)")
ax[0].bar(st + w, s1, w, color="#9ecae1", label="RL sparse_s1")
ax[0].set_xticks(st, [f"S{k}" for k in st]); ax[0].set_ylabel("attackers downed / episode")
ax[0].set_title("Defend: deploy pipeline, per curriculum stage"); ax[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
ax[1].bar(st, breach_rl, color=RED); ax[1].set_xticks(st, [f"S{k}" for k in st])
ax[1].set_ylabel("breach rate (%)"); ax[1].set_title("RL sparse_s2 breaches")
save(fig, "fig_defend_stages.png")

# 2. Defend curriculum gate results (Isaac, 256 episodes, vs assign-lead-safe at the same stage)
labels = ["S0 dense", "S1 sparse", "S2 sparse", "S3 try1", "S3 try2"]
pol = [7.36, 8.23, 8.125, 8.31, 8.58]
ref = [6.91, 8.31, 8.066, 7.98, 7.98]
passed = [True, True, True, False, False]
fig, ax = plt.subplots(figsize=(6.5, 3.2))
x = np.arange(len(labels))
ax.bar(x - 0.2, pol, 0.4, color=[BLUE if p else "#aec7e8" for p in passed], label="RL policy")
ax.bar(x + 0.2, ref, 0.4, color=GREY, label="assign-lead-safe")
for i, p in enumerate(passed):
    ax.text(x[i] - 0.2, pol[i] + 0.15, "PASS" if p else "FAIL\n(crash)", ha="center", fontsize=7,
            color=GREEN if p else RED)
ax.set_xticks(x, labels); ax.set_ylim(0, 10.5); ax.set_ylabel("downed / episode")
ax.set_title("Defend curriculum gates (Isaac Lab, 256 ep)"); ax.legend(fontsize=8, loc="lower right")
save(fig, "fig_defend_gates.png")

# 3. Basic tasks vs scripted baselines
fig, ax = plt.subplots(1, 3, figsize=(9, 3))
ax[0].bar(["zero", "PD", "RL (ours)"], [13, 96, 91.6], color=[GREY, GREY, BLUE]); ax[0].set_title("T1 Hover: % time < 20 cm")
ax[1].bar(["zero", "PD", "RL r1", "RL W1"], [1, 22.7, 49.6, 50.5], color=[GREY, GREY, BLUE, BLUE]); ax[1].set_title("T2 Waypoint: % time < 20 cm")
ax[2].bar(["shipped it5", "RL r1", "RL r2 (ours)"], [89.1, 96.9, 99.2], color=[GREY, "#9ecae1", BLUE]); ax[2].set_title("T4 Approach: success %")
ax[2].set_ylim(80, 100)
for a in ax: a.tick_params(axis="x", labelsize=8)
save(fig, "fig_basic_tasks.png")

# 4. The training-pipeline fixes: before / after
fig, ax = plt.subplots(1, 3, figsize=(9, 3))
ax[0].bar(["stock cfg", "fixed"], [2.14, 0.0105], color=[RED, BLUE]); ax[0].set_yscale("log"); ax[0].set_title("T1 tail distance (m)")
ax[1].bar(["stock cfg", "fixed"], [2.40, 0.232], color=[RED, BLUE]); ax[1].set_title("T2 tail distance (m)")
ax[2].bar(["6 early runs", "sparse_s2"], [0.03, 8.125], color=[RED, BLUE]); ax[2].set_title("Defend downed / ep")
fig.suptitle("Effect of the frame + clip_actions fixes", fontsize=10)
save(fig, "fig_fixes.png")

# copy the existing qualitative stills into the report figure folder (downscaled jpg keeps the PDF small)
try:
    from PIL import Image
    stills = {"t1_hover_3d.png": "still_t1.jpg", "t2_waypoint_3d.png": "still_t2.jpg",
              "t4_approach_r2_3d.png": "still_t4.jpg", "defend_s2_3v2_3d.png": "still_defend.jpg",
              "blender/t3_defend_Cinematic.png": "blender_defend.jpg", "blender/t4_approach_Cinematic.png": "blender_t4.jpg",
              "blender/t1_hover_Cinematic.png": "blender_t1.jpg", "blender/t2_waypoint_Cinematic.png": "blender_t2.jpg",
              "waypoint_compare.png": "waypoint_compare.jpg", "hover_compare.png": "hover_compare.jpg"}
    for src, dst in stills.items():
        p = os.path.join(HERE, src)
        if os.path.exists(p):
            im = Image.open(p).convert("RGB"); im.thumbnail((1400, 1400)); im.save(os.path.join(FIG, dst), quality=85)
except ImportError:
    print("PIL missing: stills not copied")
print("ok")
