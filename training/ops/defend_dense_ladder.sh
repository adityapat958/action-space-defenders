#!/bin/bash
# ws-defend-dense (session C, drift): AS-Defend-Live-v0, reward_mode=dense, curriculum ladder on the H100.
# Per stage: gate = max(assign-lead, assign-lead-safe) downed/ep on the same stage (dense, EVAL_EP eps, seed 7)
#   -> train NSEED seeds in parallel (warm start from previous stage's winner) -> eval best + last of each
#   -> pass if winner >= gate; else one retry (RETRY_ITERS more from the stage winner); 2nd fail -> exit 42.
# Same training knobs as session D (sparse) so C vs D is a clean reward-mode comparison:
#   MAPPO, fixed LR 3e-4, gamma 0.99, timeout bootstrap, clip_actions False (yaml).
# Results: outputs/defend_dense_scores.jsonl (one line per eval + one gate line per try), logs outputs/dd/.
set -u
MODE=${MODE:-ladder}; START=${START:-0}; END=${END:-6}; CKPT=${CKPT:-}
NSEED=${NSEED:-2}; SEED=${SEED:-42}
EVAL_EP=${EVAL_EP:-256}; EVAL_NE=${EVAL_NE:-256}
ITERS_FAST=${ITERS_FAST:-600}; ITERS_REAL=${ITERS_REAL:-400}; RETRY_ITERS=${RETRY_ITERS:-}
NE_FAST=${NE_FAST:-4096}; NE_REAL=${NE_REAL:-2048}
# GATE_FRAC: pass if policy >= GATE_FRAC * scripted on stages < STRICT_FROM (easy stages where scripted is near-optimal);
# stages >= STRICT_FROM always need policy >= scripted. MB = mini_batches (raise with num_envs).
GATE_FRAC=${GATE_FRAC:-1.0}; STRICT_FROM=${STRICT_FROM:-4}; MB=${MB:-4}

# node wakes wipe apt libs and the apt index (libxcb.so.1 ImportError); reinstall only if missing
if [ ! -e /usr/lib/x86_64-linux-gnu/libxcb.so.1 ]; then
  sudo apt-get update -qq >/dev/null 2>&1
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libxcb1 libgl1 libx11-6 libglib2.0-0 libsm6 libxext6 libxrender1 libxt6 libglu1-mesa >/dev/null 2>&1
fi
source ~/isaac/env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES TELLO_USD_PATH=$HOME/isaac/assets/tello.usd TERM=xterm
cd ~/Action_Space/simulation_infra && mkdir -p outputs/dd
TASK=AS-Defend-Live-v0; OUT=outputs/defend_dense_scores.jsonl; L=outputs/dd
AG="agent.agent.learning_rate_scheduler=null agent.agent.learning_rate=3e-4 agent.agent.discount_factor=0.99 agent.agent.time_limit_bootstrap=True agent.agent.mini_batches=$MB"
echo "dense ladder GATE_FRAC=$GATE_FRAC STRICT_FROM=$STRICT_FROM MB=$MB NE_FAST=$NE_FAST MODE=$MODE START=$START END=$END NSEED=$NSEED CKPT=$CKPT start $(date +%T)"

downed() { grep -a "downed per episode:" "$1" | tail -1 | awk '{print $4}'; }
breach() { grep -a "breach rate:" "$1" | tail -1 | awk '{print $3}'; }
bnum() { local b=$(breach "$1"); b=${b%\%}; echo ${b:-100}; }                      # breach % as a number
crashes() { local c=$(grep -a "Episode/crashes" "$1" | tail -1 | awk '{print $2}'); echo ${c:-99}; }

ev() {  # ev S TAG LOG [evaluate.py args...]  (foreground; call with & for parallel)
  local s=$1 tag=$2 lf=$3; shift 3
  timeout 2400 python -u scripts/evaluate.py --headless --task $TASK --num_envs $EVAL_NE --episodes $EVAL_EP --seed 7 \
    --set curriculum_stage=$s "reward_mode='dense'" "$@" > $lf 2>&1
  local d=$(downed $lf); d=${d:-null}
  echo "{\"ws\":\"defend-dense\",\"stage\":$s,\"tag\":\"$tag\",\"downed\":$d,\"breach\":\"$(breach $lf)\",\"log\":\"$lf\"}" >> $OUT
  echo "eval S$s $tag -> downed $d breach $(breach $lf)"
}

trn() {  # trn S NAME SEED ITERS NE [CKPT]
  local s=$1 name=$2 seed=$3 it=$4 ne=$5 ck=${6:-}
  local ra=""; [ -n "$ck" ] && ra="--checkpoint $ck"
  local t0=$(date +%s)
  python -u scripts/skrl/train.py --task $TASK --algorithm MAPPO --headless --num_envs $ne --seed $seed \
    --max_iterations $it $ra env.curriculum_stage=$s env.reward_mode=dense $AG \
    agent.agent.experiment.experiment_name=$name > $L/train_$name.log 2>&1
  echo "train $name rc=$? $(( $(date +%s)-t0 ))s"
}

case "$MODE" in
tput)  # 20 iters at S0 (velocity) and S4 (cascade); read it/s at the tail of each log
  trn 0 dense_tput_s0 $SEED 20 $NE_FAST &
  trn 4 dense_tput_s4 $SEED 20 $NE_REAL &
  wait
  for f in $L/train_dense_tput_s0.log $L/train_dense_tput_s4.log; do
    echo "$f: $(tr '\r' '\n' < $f | grep -aoE '[0-9.]+(it/s|s/it)' | tail -1)"; done ;;
ladder)
  for S in $(seq $START $END); do
    if [ $S -le 3 ]; then NE=$NE_FAST; IT=$ITERS_FAST; else NE=$NE_REAL; IT=$ITERS_REAL; fi
    echo "== S$S NE=$NE ITERS=$IT $(date +%T)"
    ev $S base:assign-lead $L/s${S}_base_lead.log --baseline assign-lead &
    ev $S base:assign-lead-safe $L/s${S}_base_safe.log --baseline assign-lead-safe &
    wait
    BASE=$(python3 -c "print(max(float('$(downed $L/s${S}_base_lead.log)' or 0), float('$(downed $L/s${S}_base_safe.log)' or 0)))")
    echo "GATE BAR S$S = $BASE"
    for try in 1 2; do
      [ $try = 2 ] && IT=${RETRY_ITERS:-$IT}
      NAMES=()
      for k in $(seq 1 $NSEED); do
        NAME=dense_s${S}_t${try}_k$k; NAMES+=($NAME)
        trn $S $NAME $((SEED+100*try+k)) $IT $NE "$CKPT" &
        sleep 20   # stagger Kit startups
      done
      wait
      CANDS=()
      for NAME in "${NAMES[@]}"; do
        RUN=$(ls -td logs/skrl/tello_defend/*_$NAME 2>/dev/null | head -1)
        [ -z "$RUN" ] && { echo "no run dir for $NAME"; tail -30 $L/train_$NAME.log; continue; }
        for ck in best_agent.pt $(ls $RUN/checkpoints | grep -v best | sort -V | tail -1); do CANDS+=("$RUN/checkpoints/$ck"); done
      done
      [ ${#CANDS[@]} = 0 ] && { echo "== STOP: no checkpoints at S$S try$try"; exit 3; }
      i=0
      for c in "${CANDS[@]}"; do
        i=$((i+1)); ev $S "policy:$c" $L/s${S}_t${try}_ev$i.log --algorithm MAPPO --checkpoint $c &
        (( i % 4 == 0 )) && wait
      done
      wait
      BEST=""; BESTD=-1; BESTLOG=/dev/null; i=0
      for c in "${CANDS[@]}"; do
        i=$((i+1)); d=$(downed $L/s${S}_t${try}_ev$i.log); d=${d:--1}
        python3 -c "exit(0 if $d>$BESTD else 1)" && { BEST=$c; BESTD=$d; BESTLOG=$L/s${S}_t${try}_ev$i.log; }
      done
      F=$GATE_FRAC; [ $S -ge $STRICT_FROM ] && F=1.0
      # 2026-10-24 10:20: S<STRICT_FROM: downed >= F x lead-safe AND breach <= lead-safe + 2 pts AND crashes <= lead-safe
      BB=$(bnum $L/s${S}_base_safe.log); BC=$(crashes $L/s${S}_base_safe.log)
      PB=$(bnum $BESTLOG); PC=$(crashes $BESTLOG)
      if [ $S -ge $STRICT_FROM ]; then PASS=$(python3 -c "print(int($BESTD>=$BASE))")
      else PASS=$(python3 -c "print(int($BESTD>=$F*$BASE and $PB<=$BB+2 and $PC<=$BC+1e-9))"); fi
      echo "  breach $PB vs $BB, crashes $PC vs $BC"
      echo "GATE S$S try$try: policy $BESTD vs scripted $BASE x$F -> pass=$PASS ($BEST)"
      echo "{\"ws\":\"defend-dense\",\"gate\":1,\"stage\":$S,\"try\":$try,\"policy\":$BESTD,\"scripted\":$BASE,\"pass\":$PASS,\"frac\":$F,\"breach\":$PB,\"crashes\":$PC,\"ckpt\":\"$PWD/$BEST\",\"ne\":$NE,\"iters\":$IT}" >> $OUT
      CKPT=$PWD/$BEST
      if [ $PASS = 1 ]; then  # trajectories for rendering
        timeout 1200 python -u scripts/evaluate.py --headless --task $TASK --num_envs 16 --episodes 8 --seed 7 --algorithm MAPPO \
          --checkpoint $BEST --set curriculum_stage=$S "reward_mode='dense'" --traj $PWD/$L/traj_dense_s$S.npz --traj_envs 4 --traj_seconds 30 \
          > $L/traj_s$S.log 2>&1 && echo "TRAJ S$S $PWD/$L/traj_dense_s$S.npz"
        break
      fi
    done
    [ $PASS = 1 ] || { echo "== STOP: S$S failed the gate twice; last ckpt $CKPT"; exit 42; }
  done ;;
*) echo "unknown MODE=$MODE"; exit 2 ;;
esac
echo "== ALLDONE $(date +%T)"
