#!/bin/bash
# ws-basic-tasks: T2 Waypoint improvement round 2. Variants train IN PARALLEL on one GPU, then each is scored (+ trajectory).
# Baseline to beat (round 1, 800 iters, fixed cfg): tail 0.232 m, 49.6% <20 cm, return 127.
#   W1 longer          : 3000 iters, same obs/reward
#   W2 + goal vel obs  : obs 15 (goal velocity), lets the policy lead a moving goal
#   W3 + precision     : W2 + rew_scale_precision=4 * exp(-d/0.1)
# Run from repo root on a GPU host:  bash ops/turing_waypoint_v2.sh   (env: ISAAC_ENV, ITERS, VARIANTS)
#SBATCH -A cs549
#SBATCH -p academic
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 03:00:00
REPO=${REPO:-$HOME/Action_Space}; cd $REPO/simulation_infra
if [ -n "${ISAAC_ENV:-}" ] || [ -f $HOME/isaac/tools/isaac_env.sh ]; then
  export ISAAC_V=5; source ${ISAAC_ENV:-$HOME/isaac/tools/isaac_env.sh}; unset CONDA_PREFIX CONDA_DEFAULT_ENV
fi
export OMNI_KIT_ACCEPT_EULA=YES TERM=xterm TELLO_USD_PATH=${TELLO_USD_PATH:-$HOME/isaac/assets/as_kit/tello.usd}
mkdir -p outputs outputs/traj; cp $REPO/ops/turing_score_basic.py scripts/score_basic.py
ITERS=${ITERS:-3000}; TAG=${TAG:-v2}
AG="agent.models.policy.clip_actions=False agent.agent.learning_rate_scheduler=null agent.agent.learning_rate=3e-4"
declare -A ENV_OV=( [W1]="" [W2]="env.obs_goal_vel=True" [W3]="env.obs_goal_vel=True env.rew_scale_precision=4.0" )
declare -A SET=(    [W1]="" [W2]="obs_goal_vel=True"     [W3]="obs_goal_vel=True rew_scale_precision=4.0" )
VARIANTS=${VARIANTS:-"W1 W2 W3"}
for v in $VARIANTS; do
  ( python -u scripts/skrl/train.py --task AS-Tello-Waypoint-v0 --headless --seed 42 --max_iterations $ITERS $AG \
      agent.agent.experiment.experiment_name=waypoint_${TAG}_$v ${ENV_OV[$v]} > outputs/ws-basic_wp_${TAG}_$v.log 2>&1
    R=$(ls -td logs/skrl/tello_waypoint/*waypoint_${TAG}_$v | head -1)
    for ck in best_agent.pt $(ls $R/checkpoints | grep -v best | sort -V | tail -1); do
      python -u scripts/score_basic.py --task AS-Tello-Waypoint-v0 --checkpoint $R/checkpoints/$ck --set ${SET[$v]} \
        --out outputs/ws_basic_scores.jsonl --traj outputs/traj/waypoint_${TAG}_${v}_${ck%.pt}.npz >> outputs/ws-basic_wp_${TAG}_$v.log 2>&1
    done
    echo "== $v done $(date +%T)" ) &
  sleep 20  # stagger Kit startup
done
wait; echo "== ALLDONE $(date +%T)"
