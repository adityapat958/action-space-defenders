#!/bin/bash
# ws-assign-reward: MAPPO on AS-Defend-Assign-v0 (assigned-target closing + slot-0 obs + half-shared capture).
# Same budget/seed as ops/defend_train.sh so runs are directly comparable with the stock-reward RL baseline.
#   ~/defend_assign_train.sh > ~/ws-assign-reward_train.log 2>&1   (on the H100, detached)
source ~/isaac/env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES TELLO_USD_PATH=$HOME/isaac/assets/tello.usd TERM=xterm
cd ~/Action_Space/simulation_infra && mkdir -p outputs/train
HARD="env.attacker_speed=[0.8,1.2] env.attacker_evasion=1.0"
ITERS=${ITERS:-1500}
run() { local name=$1; shift; nohup python -u scripts/skrl/train.py --task AS-Defend-Assign-v0 --headless --num_envs 4096 --seed 1 \
          --max_iterations $ITERS "$@" agent.agent.experiment.experiment_name=$name > outputs/train/$name.log 2>&1 & echo "started $name pid $!"; }
# Optimisation fix (seen in every stock run: KL-adaptive LR collapses 5e-4 -> 2e-5 in ~30 iters, return drifts down):
# gamma 0.99 at 60 Hz = 1.7 s horizon < 4-8 s time-to-capture; timeouts are not deaths (waves respawn); looser KL.
TUNED="agent.agent.discount_factor=0.995 agent.agent.time_limit_bootstrap=True agent.agent.learning_rate=3.0e-4 agent.agent.learning_rate_scheduler_kwargs.kl_threshold=0.05"
case "${VARIANT:-stock}" in
  stock) run assign_mappo_hard --algorithm MAPPO $HARD ;;          # same agent cfg as ws-rl mappo_hard: isolates reward+obs
  tuned) run assign_mappo_hard_tuned --algorithm MAPPO $HARD $TUNED ;;
  tuned_stockreward)  # control: same optimiser fix on the stock task, to split reward effect from optimiser effect
         nohup python -u scripts/skrl/train.py --task AS-Defend-v0 --headless --num_envs 4096 --seed 1 --max_iterations $ITERS \
           --algorithm MAPPO $HARD $TUNED agent.agent.experiment.experiment_name=stock_mappo_hard_tuned > outputs/train/stock_mappo_hard_tuned.log 2>&1 &
         echo "started stock_mappo_hard_tuned pid $!" ;;
esac
wait; echo "== ASSIGN TRAIN ALLDONE $(date +%T)"
