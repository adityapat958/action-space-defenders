#!/bin/bash
# ws-defend-sparse: ADAPTIVE curriculum ladder (dsp_adaptive.py)
# On gate fail: ease difficulty (alpha interpolation), bump crash penalty, retrain, step back up.
# Launch:
#   sbatch -J dsp_adap_s3 --export=ALL,START=3,END=6,ITERS=600,NE=16384,MB=16 ops/defend_adaptive.sbatch
#   (CKPT=path, ALPHA_STEP=0.33, CRASH_BUMP=5, MAX_RETRIES=4, SEED=42 also accepted)
#SBATCH -A cs549
#SBATCH -p academic
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 18:00:00
#SBATCH -o $REPO/simulation_infra/outputs/dsp_adap_%x_%j.log
set -u
REPO=${REPO:-$HOME/Action_Space}
export ISAAC_V=5; source ${ISAAC_ENV:-$HOME/isaac/tools/isaac_env.sh}
unset CONDA_PREFIX CONDA_DEFAULT_ENV
export TELLO_USD_PATH=${TELLO_USD_PATH:-$HOME/isaac/assets/as_kit/tello.usd} OMNI_KIT_ACCEPT_EULA=YES TERM=xterm
cd $REPO/simulation_infra && mkdir -p outputs

START=${START:-3}
END=${END:-6}
ITERS=${ITERS:-600}
NE=${NE:-16384}
MB=${MB:-16}
CKPT=${CKPT:-$HOME/hackathon/ckpt_backup/sparse_s2/agent_9600.pt}
ALPHA_STEP=${ALPHA_STEP:-0.33}
CRASH_BUMP=${CRASH_BUMP:-5.0}
MAX_RETRIES=${MAX_RETRIES:-4}
SEED=${SEED:-42}
DYNAMICS=${DYNAMICS:-auto}

echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) START=$START END=$END ITERS=$ITERS NE=$NE CKPT=$CKPT start=$(date +%T)"

python -u $REPO/ops/dsp_adaptive.py \
  --start_stage $START \
  --end_stage   $END \
  --ckpt        $CKPT \
  --iters       $ITERS \
  --num_envs    $NE \
  --mini_batches $MB \
  --seed        $SEED \
  --max_retries $MAX_RETRIES \
  --alpha_step  $ALPHA_STEP \
  --crash_bump  $CRASH_BUMP \
  --dynamics    $DYNAMICS \
  --tsv         outputs/dsp_adaptive_log.tsv \
  --job_id      ${SLURM_JOB_ID:-local}

echo "== ALLDONE $(date +%T)"
