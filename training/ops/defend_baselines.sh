#!/bin/bash
# Defend scripted baselines (default, fast, fast+evasive attackers) + trajectories + matplotlib comparison videos.
# apt libs reset on every node wake
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libxcb1 libgl1 libx11-6 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg vulkan-tools libvulkan1 libglu1-mesa >/dev/null 2>&1 || (sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libxcb1 libgl1 libx11-6 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg vulkan-tools libvulkan1 libglu1-mesa >/dev/null)
source ~/isaac/env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES TELLO_USD_PATH=$HOME/isaac/assets/tello.usd TERM=xterm
cd ~/Action_Space/simulation_infra && mkdir -p outputs/defend outputs/videos
python -c "import matplotlib" 2>/dev/null || uv pip install -q matplotlib
command -v ffmpeg >/dev/null || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null
i=0
for SET in "" "attacker_speed=[0.8,1.2]" "attacker_speed=[0.8,1.2] attacker_evasion=1.0"; do
  i=$((i+1)); NAMES=()
  for B in zero chase lead assign-chase assign-lead; do
    TAG=s${i}_$B
    echo "== $B | ${SET:-default} $(date +%T)"
    python -u scripts/evaluate.py --task AS-Defend-v0 --baseline $B --num_envs 256 --episodes 256 --seed 7 \
      --traj outputs/defend/traj_$TAG.npz ${SET:+--set $SET} > outputs/defend/$TAG.log 2>&1
    grep -E "interception rate|breach rate|time to intercept|downed per|closest to" outputs/defend/$TAG.log | sed 's/^/   /'
  done
  python scripts/render_defend_traj.py outputs/defend/traj_s${i}_{chase,assign-chase,assign-lead}.npz \
    --out outputs/videos/defend_baselines_s$i.mp4 --envs 2 2>&1 | grep render
done
echo "== ALLDONE $(date +%T)"
