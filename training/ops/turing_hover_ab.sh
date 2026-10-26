#!/bin/bash
# ws-basic-tasks A/B: does the KL-adaptive LR collapse break PPO? Hover with fixed LR, +/- clip_actions.
cd ~/Action_Space/simulation_infra; export ISAAC_V=5; source ~/isaac/tools/isaac_env.sh; unset CONDA_PREFIX CONDA_DEFAULT_ENV
export TELLO_USD_PATH=$HOME/isaac/assets/as_kit/tello.usd OMNI_KIT_ACCEPT_EULA=YES TERM=xterm
for v in A_fixedlr B_fixedlr_noclip; do
  X="agent.agent.learning_rate_scheduler=null agent.agent.learning_rate=3e-4 agent.agent.experiment.experiment_name=hover_$v"
  [ $v = B_fixedlr_noclip ] && X="$X agent.models.policy.clip_actions=False"
  echo "== train $v $(date +%T)"; python -u scripts/skrl/train.py --task AS-Tello-Hover-v0 --headless --seed 42 --max_iterations 400 $X 2>&1 | grep -E "Error|Traceback|Training time" 
  R=$(ls -td logs/skrl/tello_hover/*hover_$v | head -1)
  python -u scripts/score_basic.py --task AS-Tello-Hover-v0 --checkpoint $R/checkpoints/agent_9600.pt --out outputs/ws_basic_scores.jsonl --traj outputs/traj/hover_$v.npz 2>&1 | sed -n '/===/,$p'
done
echo ALLDONE
