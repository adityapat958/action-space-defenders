# T1 Hover (hover_t1)

Task `AS-Tello-Hover-v0`, skrl PPO, 1024 envs, 400 iters (9600 timesteps).

| | |
|---|---|
| task | `AS-Tello-Hover-v0` |
| algorithm | skrl PPO, [256,128,64] ELU, RunningStandardScaler |
| run | `hover_B_fixedlr_noclip` (2026-10-24_05-22-51), fixed LR, no obs clip |
| score | **within_20cm 91.6%**, tail_dist 0.0105 m, died_rate 0.0, return 79.98 (512 ep, seed 7) |
| source path | `turing:<cluster-log-path>/skrl/tello_hover/2026-10-24_05-22-51_ppo_torch_hover_B_fixedlr_noclip/checkpoints/agent_9600.pt` |
| checkpoint | `agent_9600.pt` (sha256 `8dec92e9...38af282a`) |

## Play command
```bash
python simulation_infra/scripts/skrl/play.py \
  --task AS-Tello-Hover-v0 \
  --checkpoint policies/hover_t1/agent_9600.pt \
  --num_envs 16
```
