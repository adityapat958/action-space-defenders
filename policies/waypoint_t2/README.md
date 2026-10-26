# T2 Waypoint (waypoint_t2)

Task `AS-Tello-Waypoint-v0`, skrl PPO, 1024 envs, 800 iters (19200 timesteps).

| | |
|---|---|
| task | `AS-Tello-Waypoint-v0` |
| algorithm | skrl PPO, [256,128,64] ELU, RunningStandardScaler |
| run | `tello_waypoint_ppo` retry (2026-10-24_05-28-57), post body-frame force fix |
| score | **within_20cm 49.6%**, tail_dist 0.232 m, died_rate 0.0, return 127.07 (512 ep, seed 7) |
| source path | `turing:<cluster-log-path>/skrl/tello_waypoint/2026-10-24_05-28-57_ppo_torch_tello_waypoint_ppo/checkpoints/best_agent.pt` |
| checkpoint | `best_agent.pt` (sha256 `42720443...2bf4fa0dc`) |

## Play command
```bash
python simulation_infra/scripts/skrl/play.py \
  --task AS-Tello-Waypoint-v0 \
  --checkpoint policies/waypoint_t2/best_agent.pt \
  --num_envs 16
```
