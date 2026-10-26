#!/bin/bash
# ws-assign-reward: score finished runs on hard attackers (256 ep, seed 7) -> ~/ws-assign-reward_scores.txt
source ~/isaac/env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES TELLO_USD_PATH=$HOME/isaac/assets/tello.usd TERM=xterm
cd ~/Action_Space/simulation_infra
OUT=~/ws-assign-reward_scores.txt; : > $OUT
H="attacker_speed=[0.8,1.2] attacker_evasion=1.0"
score() { local task=$1 pat=$2; local d=$(ls -d logs/skrl/tello_defend/*$pat 2>/dev/null | tail -1); [ -z "$d" ] && { echo "$pat: no run" >> $OUT; return; }
  local ck=$d/checkpoints/best_agent.pt; [ -f $ck ] || ck=$(ls -t $d/checkpoints/*.pt | head -1)
  echo "== $pat ($task) $ck" >> $OUT
  python -u scripts/evaluate.py --task $task --checkpoint $ck --algorithm MAPPO --num_envs 256 --episodes 256 --seed 7 --set $H 2>&1 \
    | grep -E "interception rate|breach rate|time to intercept|downed per|closest to" >> $OUT; }
score AS-Defend-v0        _mappo_torch_mappo_hard
score AS-Defend-Assign-v0 _assign_mappo_hard
score AS-Defend-Assign-v0 _assign_mappo_hard_tuned
score AS-Defend-v0        _stock_mappo_hard_tuned
echo "ref (scripted, hard): chase 0.95 downed/ep breach 100% | assign-chase 1.41 / 100%" >> $OUT
echo SCORE_DONE >> $OUT
