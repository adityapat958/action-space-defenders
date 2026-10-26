#!/bin/bash
if [ ! -e /usr/lib/x86_64-linux-gnu/libxcb.so.1 ]; then
  sudo apt-get update -qq >/dev/null 2>&1
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libxcb1 libgl1 libx11-6 libglib2.0-0 libsm6 libxext6 libxrender1 libxt6 libglu1-mesa >/dev/null 2>&1
fi
source ~/isaac/env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES TELLO_USD_PATH=$HOME/isaac/assets/tello.usd TERM=xterm
cd ~/Action_Space/simulation_infra && mkdir -p outputs/fastfit
for m in ${MODELS:-cascade velocity}; do
  t0=$(date +%s)
  timeout -s KILL 3600 python -u scripts/log_step_response.py --model $m --num_envs 256 --seconds ${SECS:-20} --out outputs/fastfit/$m.npz > outputs/fastfit/$m.log 2>&1
  echo "$m rc=$? $(( $(date +%s)-t0 ))s"
done
echo done
