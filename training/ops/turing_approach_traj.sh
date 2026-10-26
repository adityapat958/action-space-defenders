#!/bin/bash
# ws-basic: record Approach trajectories (4 envs x 4 drones, ~25 s) for a checkpoint: CK=... OUT=...
cd ~/Action_Space/simulation_infra; export ISAAC_V=5; source ~/isaac/tools/isaac_env.sh; unset CONDA_PREFIX CONDA_DEFAULT_ENV
export TELLO_USD_PATH=$HOME/isaac/assets/as_kit/tello.usd OMNI_KIT_ACCEPT_EULA=YES TERM=xterm
cp ../ops/turing_play_traj.py scripts/sb3/play_traj.py
python -u scripts/sb3/play_traj.py --task AS-Tello-Approach-v0 --headless --num_envs 16 --seed 7 --steps ${STEPS:-340} --checkpoint $CK --traj $OUT \
  env.spawn_azimuth_range=0.0 env.rew_scale_separation=-20.0 env.rew_violation=-100.0 2>&1 | grep -E "traj|Error|Traceback|step_dt" | tail -5
