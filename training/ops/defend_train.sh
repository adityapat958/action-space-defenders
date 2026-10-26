#!/bin/bash
# First Defend RL runs (stock reward = RL baseline): MAPPO default, MAPPO hard, IPPO hard. Runs in parallel on one H100.
# apt libs reset on every node wake
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libxcb1 libgl1 libx11-6 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg vulkan-tools libvulkan1 libglu1-mesa >/dev/null 2>&1 || (sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libxcb1 libgl1 libx11-6 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg vulkan-tools libvulkan1 libglu1-mesa >/dev/null)
source ~/isaac/env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES TELLO_USD_PATH=$HOME/isaac/assets/tello.usd TERM=xterm
cd ~/Action_Space/simulation_infra && mkdir -p outputs/train
HARD="env.attacker_speed=[0.8,1.2] env.attacker_evasion=1.0"
ITERS=${ITERS:-1500}
run() { local name=$1; shift; nohup python -u scripts/skrl/train.py --task AS-Defend-v0 --headless --num_envs 4096 --seed 1 \
          --max_iterations $ITERS "$@" agent.agent.experiment.experiment_name=$name > outputs/train/$name.log 2>&1 & echo "started $name pid $!"; }
run mappo_default --algorithm MAPPO
run mappo_hard    --algorithm MAPPO $HARD
run ippo_hard     --algorithm IPPO  $HARD
wait; echo "== TRAIN ALLDONE $(date +%T)"
