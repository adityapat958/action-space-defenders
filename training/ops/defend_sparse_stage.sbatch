#!/bin/bash
# ws-defend-sparse (session D): AS-Defend-Live-v0, reward_mode=sparse, curriculum restart-from-checkpoint.
#   Throughput:  sbatch -J dsp_tput --export=ALL,MODE=tput,NE=2048,DYN=cascade ops/defend_sparse_stage.sbatch
#   Ladder:      sbatch -J dsp_ladder --export=ALL,MODE=ladder[,START=0,END=6,CKPT=...,ITERS=300] ops/defend_sparse_stage.sbatch
#                probes cascade@2048 (+1024) first, picks NE/DYN by rule (see pick()), then per stage: gate -> train -> eval,
#                advance on pass, one retry (ITERS more from the stage's own best) on fail, stop on 2nd fail.
#   One stage:   sbatch -J dsp_s0 --export=ALL,MODE=stage,S=0,ITERS=300[,CKPT=<prev best_agent.pt>][,NE=2048][,DYN=velocity] ops/defend_sparse_stage.sbatch
# Stage = gate baseline (assign-lead, same stage) -> train (fixed LR 3e-4, gamma 0.99, timeout bootstrap) -> eval best + last.
# Results: outputs/defend_sparse_scores.jsonl (one line per eval, tagged) + outputs/evaluate_results.jsonl (evaluate.py's own).
# Gate: policy downed/ep >= assign-lead downed/ep on the same stage. Evals always use cascade dynamics (deploy model).
#SBATCH -A cs549
#SBATCH -p academic
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o $REPO/simulation_infra/outputs/dsp_%x_%j.log
set -u
REPO=${REPO:-$HOME/Action_Space}
export ISAAC_V=5; source ${ISAAC_ENV:-$HOME/isaac/tools/isaac_env.sh}
unset CONDA_PREFIX CONDA_DEFAULT_ENV
export TELLO_USD_PATH=${TELLO_USD_PATH:-$HOME/isaac/assets/as_kit/tello.usd} OMNI_KIT_ACCEPT_EULA=YES TERM=xterm
cd $REPO/simulation_infra && mkdir -p outputs
grep -q '"AS-Defend-Live-v0"' source/action_space_kit/action_space_kit/tasks/direct/tello_defend/__init__.py \
  || { echo "tello_defend lacks AS-Defend-Live-v0; sync it first"; exit 3; }
TASK=AS-Defend-Live-v0
MODE=${MODE:-stage}; NE=${NE:-2048}; MB=${MB:-}; DYN=${DYN:-cascade}; S=${S:-0}; ITERS=${ITERS:-300}; CKPT=${CKPT:-}
EVAL_EP=${EVAL_EP:-256}; EVAL_NE=${EVAL_NE:-256}; SEED=${SEED:-42}
OUT=outputs/defend_sparse_scores.jsonl
echo "node $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) MODE=$MODE S=$S NE=$NE DYN=$DYN ITERS=$ITERS CKPT=$CKPT start $(date +%T)"
run() { echo "== $1 $(date +%T)"; local t0=$(date +%s); shift; "$@"; local rc=$?; echo "== rc=$rc $(( $(date +%s)-t0 ))s $(date +%T)"; return $rc; }
AG="${MB:+agent.agent.mini_batches=$MB }agent.models.policy.clip_actions=False agent.agent.learning_rate_scheduler=null agent.agent.learning_rate=3e-4 agent.agent.discount_factor=0.99 agent.agent.time_limit_bootstrap=True"

ev() {  # ev TAG [evaluate.py args...]  -> runs evaluate, appends tagged summary line to $OUT
  local tag=$1; shift
  local before=$(wc -l < outputs/evaluate_results.jsonl 2>/dev/null || echo 0)
  run eval_$tag python -u scripts/evaluate.py --task $TASK --num_envs $EVAL_NE --episodes $EVAL_EP --seed 7 \
      --set curriculum_stage=$S "reward_mode='sparse'" "$@" || { echo "== STOP: eval $tag failed"; exit 4; }
  tail -n +$((before+1)) outputs/evaluate_results.jsonl 2>/dev/null | tail -1 | \
    python -c "import sys,json; [print(json.dumps(dict(ws='defend-sparse', stage=$S, tag='$tag', job='${SLURM_JOB_ID:-}', rec=json.loads(l)))) for l in sys.stdin]" >> $OUT
}

case "$MODE" in
tput)
  # 20 iterations x 16 rollouts = 320 policy steps per env; read the tqdm it/s near the end of the log
  run train_tput python -u $REPO/ops/dsp_train.py --task AS-Defend-Live-S${S}-v0 --algorithm MAPPO --headless --num_envs $NE --seed $SEED \
    --max_iterations 20 env.reward_mode=sparse env.dynamics_model=$DYN $AG \
    agent.agent.experiment.experiment_name=sparse_tput_${DYN}_${NE} ;;
stage)
  [ "${SKIP_BASE:-0}" = 1 ] || ev assign-lead --baseline assign-lead
  RA=""; [ -n "$CKPT" ] && RA="--checkpoint $CKPT"
  NAME=sparse_s${S}_${DYN}
  run train python -u $REPO/ops/dsp_train.py --task AS-Defend-Live-S${S}-v0 --algorithm MAPPO --headless --num_envs $NE --seed $SEED \
    --max_iterations $ITERS $RA env.reward_mode=sparse env.dynamics_model=$DYN $AG \
    agent.agent.experiment.experiment_name=$NAME
  RUN=$(ls -td logs/skrl/tello_defend/*_$NAME | head -1); echo "run dir $RUN"
  for ck in best_agent.pt $(ls $RUN/checkpoints | grep -v best | sort -V | tail -1); do
    ev "policy:$RUN/checkpoints/$ck" --algorithm MAPPO --checkpoint $RUN/checkpoints/$ck
  done ;;
ladder)
  # first ladder job to start wins; duplicates queued on other partitions exit (rm -r outputs/dsp_ladder.lock to reset)
  if ! mkdir outputs/dsp_ladder.lock 2>/dev/null; then echo "ladder already running: $(cat outputs/dsp_ladder.lock/job)"; exit 0; fi
  echo ${SLURM_JOB_ID:-$$} > outputs/dsp_ladder.lock/job; trap 'rm -rf outputs/dsp_ladder.lock' EXIT
  START=${START:-0}; END=${END:-6}; MIN_IPS=${MIN_IPS:-2.7}   # 2.7 env-steps/s = 300 iters x 16 rollouts in <= 30 min
  ips() { python - "$1" <<'PY'
import re,sys
t=open(sys.argv[1],errors="ignore").read().replace("\r","\n")
v=[float(a) if u=="it/s" else 1/float(a) for a,u in re.findall(r"([0-9.]+)(it/s|s/it)",t)]
print(v[-1] if v else 0)
PY
  }
  probe() {  # probe NE DYN -> prints env-steps/s
    local lf=outputs/dsp_probe_${1}_${2}_${SLURM_JOB_ID:-x}.log
    python -u $REPO/ops/dsp_train.py --task AS-Defend-Live-S${START}-v0 --algorithm MAPPO --headless --num_envs $1 --seed $SEED --max_iterations 12 \
      env.reward_mode=sparse env.dynamics_model=$2 $AG agent.agent.experiment.experiment_name=sparse_tput_${2}_$1 > $lf 2>&1
    ips $lf
  }
  if [ -z "${PICK:-}" ]; then
    C2=$(probe 2048 cascade); echo "PROBE cascade 2048: $C2 steps/s ($(python -c "print(round(2048*$C2))") samples/s/agent)"
    [ "$C2" = 0 ] && { echo "== STOP: probe failed, see outputs/dsp_probe_2048_cascade_${SLURM_JOB_ID:-x}.log"; tail -30 outputs/dsp_probe_2048_cascade_${SLURM_JOB_ID:-x}.log; exit 3; }
    if python -c "exit(0 if $C2>=$MIN_IPS else 1)"; then PICK=c2048
    else C1=$(probe 1024 cascade); echo "PROBE cascade 1024: $C1 steps/s"
      # 1024 only helps if it is really faster; on A30 it isn't (1.95 vs 1.89 steps/s: cost = 37 substeps/step, not env count)
      if python -c "exit(0 if $C1>=$MIN_IPS else 1)"; then PICK=c1024; elif python -c "exit(0 if $C1>=1.3*$C2 else 1)"; then PICK=v012; else PICK=v012w; fi
    fi
  fi
  echo "PICK=$PICK (c2048: cascade 2048 all stages | c1024: cascade 1024 all | v012: velocity 2048 for S0-2, cascade 1024 S3+ | v012w: same but cascade 2048 S3+)"
  for S in $(seq $START $END); do
    case $PICK in c2048) NE=2048; DYN=cascade;; c4096) NE=4096; DYN=cascade;; c16k) NE=16384; DYN=cascade;; c1024) NE=1024; DYN=cascade;;
                  v012) if [ $S -le 2 ]; then NE=2048; DYN=velocity; else NE=1024; DYN=cascade; fi;;
                  v012w) if [ $S -le 2 ]; then NE=2048; DYN=velocity; else NE=2048; DYN=cascade; fi;; esac
    # gate = max(assign-lead, assign-lead-safe): -safe avoids teammate collisions commit 2938b74, the stricter bar at S5+
    # baseline = the scripted policy with the most downed/ep; keep its breach rate + crashes/ep for the S1-S3 gate
    BASE=0; BBR=0; BCR=0; BL="assign-lead assign-lead-safe"
    [ $S = $START ] && [ -n "${BASE0:-}" ] && { BASE=$BASE0; BBR=${BBR0:-0}; BCR=${BCR0:-0}; BL=""; echo "gate baseline S$S reused: $BASE0 breach $BBR crashes $BCR"; }
    for b in $BL; do
      ev $b --baseline $b
      read B BR CR < <(tail -1 $OUT | python -c "import sys,json; r=json.loads(sys.stdin.read())['rec']; print(r['downed_per_episode'], r['breach_rate'], r['stats'].get('Episode/crashes',0))")
      if python -c "exit(0 if $B>$BASE else 1)"; then BASE=$B; BBR=$BR; BCR=$CR; fi
    done
    for try in 1 2; do
      NAME=sparse_s${S}_${DYN}_t$try
      RA=""; [ -n "$CKPT" ] && RA="--checkpoint $CKPT"
      run train_s${S}_t$try python -u $REPO/ops/dsp_train.py --task AS-Defend-Live-S${S}-v0 --algorithm MAPPO --headless --num_envs $NE --seed $((SEED+try)) \
        --max_iterations $ITERS $RA env.reward_mode=sparse env.dynamics_model=$DYN $AG \
        agent.agent.experiment.experiment_name=$NAME
      RUN=$(ls -td logs/skrl/tello_defend/*_$NAME | head -1); echo "run dir $RUN"
      BEST=""; BESTD=-1
      # EVAL_CKS=last (default) evaluates only the final checkpoint; =both also evaluates best_agent.pt (+75 s)
      CKS=$(ls $RUN/checkpoints | grep -v best | sort -V | tail -1); [ "${EVAL_CKS:-last}" = both ] && CKS="best_agent.pt $CKS"
      for ck in $CKS; do
        ev "policy:$RUN/checkpoints/$ck" --algorithm MAPPO --checkpoint $RUN/checkpoints/$ck
        read D DBR DCR < <(tail -1 $OUT | python -c "import sys,json; r=json.loads(sys.stdin.read())['rec']; print(r['downed_per_episode'], r['breach_rate'], r['stats'].get('Episode/crashes',0))")
        if python -c "exit(0 if $D>$BESTD else 1)"; then BEST=$RUN/checkpoints/$ck; BESTD=$D; PBR=$DBR; PCR=$DCR; fi
      done
      # gate (2026-10-24): S1-S3 relaxed = downed >= 0.95*scripted AND breach <= scripted+0.02 AND crashes <= scripted
      # (+CRASH_TOL, default 0.1/ep: assign-lead-safe has 0 crashes); S0 and S4+ strict downed >= scripted
      if [ $S -ge 1 ] && [ $S -le 3 ]; then
        PASS=$(python -c "print(int($BESTD>=0.95*$BASE and $PBR<=$BBR+0.02 and $PCR<=$BCR+${CRASH_TOL:-0.1}))")
      else PASS=$(python -c "print(int($BESTD>=$BASE))"); fi
      echo "GATE S$S try$try: policy $BESTD (breach $PBR, crashes $PCR) vs scripted $BASE (breach $BBR, crashes $BCR) -> pass=$PASS ($BEST)"
      echo "{\"ws\":\"defend-sparse\",\"gate\":1,\"stage\":$S,\"try\":$try,\"policy\":$BESTD,\"assign_lead\":$BASE,\"pass\":$PASS,\"ckpt\":\"$PWD/$BEST\",\"ne\":$NE,\"dyn\":\"$DYN\",\"job\":\"${SLURM_JOB_ID:-}\"}" >> $OUT
      CKPT=$PWD/$BEST
      [ $PASS = 1 ] && break
    done
    [ $PASS = 1 ] || { echo "== STOP: S$S failed the gate twice; last ckpt $CKPT"; exit 42; }
  done ;;
*) echo "unknown MODE=$MODE"; exit 2 ;;
esac
echo "== ALLDONE $(date +%T)"
