#!/bin/bash
# ws-basic-tasks (session A): train + score one basic task on Turing. Physics only, headless.
#   sbatch --export=ALL,T=hover    ops/turing_basic.sbatch   # T1 AS-Tello-Hover-v0     (skrl PPO)
#   sbatch --export=ALL,T=waypoint ops/turing_basic.sbatch   # T2 AS-Tello-Waypoint-v0  (skrl PPO)
#   sbatch --export=ALL,T=approach ops/turing_basic.sbatch   # T4 AS-Tello-Approach-v0  (SB3, recipe of policies/approach_it5_room_fixed)
# Default: academic A30 (starts immediately). `long` queues for days; override with sbatch -A/-p/--gres.
# Already on a GPU node (srun/salloc, e.g. RTX PRO 6000)? Just: T=hover bash ops/turing_basic.sbatch
#   Needs Isaac Sim 5.1 + Lab 2.3.2 env (source $ISAAC_ENV) and the repo at $REPO. Physics only, so Blackwell is fine.
#SBATCH -A cs549
#SBATCH -p academic
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o $REPO/simulation_infra/outputs/ws_basic_%x_%j.log
set -u
REPO=${REPO:-$HOME/Action_Space}
export ISAAC_V=5; source ${ISAAC_ENV:-$HOME/isaac/tools/isaac_env.sh}
unset CONDA_PREFIX CONDA_DEFAULT_ENV
export TELLO_USD_PATH=${TELLO_USD_PATH:-$HOME/isaac/assets/as_kit/tello.usd} OMNI_KIT_ACCEPT_EULA=YES TERM=xterm
cd $REPO/simulation_infra && mkdir -p outputs
cp $REPO/ops/turing_score_basic.py scripts/score_basic.py
# guard: T1/T2 must run with the body-frame force fix 2026-10-24 or results are invalid
grep -q quat_apply_inverse source/action_space_kit/action_space_kit/tasks/direct/tello_hover/tello_hover_env.py \
  || { echo "tello_hover_env.py lacks the world->body force fix; git pull first"; [ "$T" != approach ] && exit 3; }
OUT=outputs/ws_basic_scores.jsonl
echo "node $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) T=$T start $(date +%T)"
run() { echo "== $1 $(date +%T)"; shift; "$@"; echo "== rc=$? $(date +%T)"; }

case "$T" in
hover|waypoint)
  TASK=AS-Tello-${T^}-v0; ITERS=${ITERS:-$([ $T = hover ] && echo 400 || echo 800)}
  # SCORE_ONLY=1: skip baselines + training, (re)score the latest run (e.g. to add trajectories)
  if [ "${SCORE_ONLY:-0}" != 1 ]; then
    for b in zero pd; do
      run baseline_$b python -u scripts/score_basic.py --task $TASK --baseline $b --out $OUT --traj outputs/traj/${T}_baseline_$b.npz
    done
    # clip_actions=True in the stock skrl cfg breaks PPO (LR collapse / NaN, see STATUS §4); the env clamps anyway
    AGENT_OV=${AGENT_OV:-"agent.models.policy.clip_actions=False agent.agent.learning_rate_scheduler=null agent.agent.learning_rate=3e-4"}
    run train python -u scripts/skrl/train.py --task $TASK --headless --seed 42 --max_iterations $ITERS $AGENT_OV
  fi
  RUN=$(ls -td logs/skrl/tello_$T/*_ppo_torch* | head -1); echo "run dir $RUN"
  for ck in best_agent.pt $(ls $RUN/checkpoints | grep -v best | sort -V | tail -1); do
    run score_$ck python -u scripts/score_basic.py --task $TASK --checkpoint $RUN/checkpoints/$ck --out $OUT \
      --traj outputs/traj/${T}_$(basename $RUN)_${ck%.pt}.npz
  done ;;
approach)
  TASK=AS-Tello-Approach-v0; ITERS=${ITERS:-800}
  OV="env.spawn_azimuth_range=0.0 env.rew_scale_separation=-20.0 env.rew_violation=-100.0"
  run train python -u scripts/sb3/train.py --task $TASK --headless --num_envs 512 --seed 42 --max_iterations $ITERS agent.learning_rate=1e-3 $OV
  RUN=$(ls -td logs/sb3/$TASK/*/ | head -1); echo "run dir $RUN"
  run score python -u scripts/sb3/play.py --task $TASK --headless --num_envs 64 --episodes 256 --checkpoint ${RUN}model.zip --report outputs/ws_basic_approach_play.json $OV
  python - <<EOF >> $OUT
import json; r=json.load(open("outputs/ws_basic_approach_play.json")); s=r.get("summary", r)
print(json.dumps(dict(task="$TASK", label="${RUN}model.zip", summary=s)))
EOF
  ;;
approach2)
  # round 2: continue from a round-1 checkpoint (CK, default = latest run) at lower LR (LR, default 3e-4)
  TASK=AS-Tello-Approach-v0; ITERS=${ITERS:-800}
  OV="env.spawn_azimuth_range=0.0 env.rew_scale_separation=-20.0 env.rew_violation=-100.0"
  CK=${CK:-$(ls -td logs/sb3/$TASK/*/model.zip | head -1)}; echo "continue from $CK"
  run train python -u scripts/sb3/train.py --task $TASK --headless --num_envs 512 --seed 43 --max_iterations $ITERS \
      --checkpoint $CK agent.learning_rate=${LR:-3e-4} $OV
  RUN=$(ls -td logs/sb3/$TASK/*/ | head -1); echo "run dir $RUN"
  run score python -u scripts/sb3/play.py --task $TASK --headless --num_envs 64 --episodes 256 --checkpoint ${RUN}model.zip --report outputs/ws_basic_approach2_play.json $OV
  python - <<EOF2 >> $OUT
import json; r=json.load(open("outputs/ws_basic_approach2_play.json")); print(json.dumps(dict(task="$TASK", label="${RUN}model.zip", round=2, summary=r.get("summary"))))
EOF2
  ;;
*) echo "unknown T=$T"; exit 2 ;;
esac
echo "== ALLDONE $(date +%T)"
