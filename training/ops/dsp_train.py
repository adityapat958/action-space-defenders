"""ws-defend-sparse: run scripts/skrl/train.py with per-stage task ids AS-Defend-Live-S{k}-v0.

Why: `env.curriculum_stage=k` via Hydra fails, because TelloDefendEnvCfg.curriculum_stage defaults to None and Isaac
Lab's from_dict rejects int-for-None. These ids use a cfg subclass with curriculum_stage=k baked in; everything else
(env class, agent yamls) is AS-Defend-Live-v0's. Usage (from simulation_infra/):
    python ops/dsp_train.py --task AS-Defend-Live-S0-v0 <train.py args...>
"""
import os
import runpy
import sys

import gymnasium as gym

PKG = "action_space_kit.tasks.direct.tello_defend"
for k in range(7):
    gym.register(
        id=f"AS-Defend-Live-S{k}-v0",
        entry_point=f"{PKG}.tello_defend_env:TelloDefendEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"dsp_tasks:LiveS{k}",
            "skrl_cfg_entry_point": f"{PKG}.agents:skrl_ppo_cfg.yaml",
            "skrl_ippo_cfg_entry_point": f"{PKG}.agents:skrl_ippo_cfg.yaml",
            "skrl_mappo_cfg_entry_point": f"{PKG}.agents:skrl_mappo_cfg.yaml",
        },
    )

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for dsp_tasks
script = os.path.join(os.getcwd(), "scripts", "skrl", "train.py")
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name="__main__")
