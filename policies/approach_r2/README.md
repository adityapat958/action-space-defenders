# T4 Approach, round 2 (ws-basic-tasks)
AS-Tello-Approach-v0, SB3 PPO. Continued from round 1 (`2026-10-24_05-16-21`, 800 iters at LR 1e-3) for another 800 iters at LR 3e-4, seed 43.
Source: `turing:~/Action_Space/simulation_infra/logs/sb3/AS-Tello-Approach-v0/2026-10-24_08-19-12/` (job 2404970).
Eval (play.py, 256 ep, overrides `env.spawn_azimuth_range=0.0 env.rew_scale_separation=-20.0 env.rew_violation=-100.0`):
**success 0.992, collision 0.0039, crash 0.0039, illegal entry 0**. Round 1: 0.969 / 0.020 / 0. Shipped approach_it5: 0.891 / 0.008.
Run: `python scripts/sb3/play.py --task AS-Tello-Approach-v0 --checkpoint policies/approach_r2/model.zip <overrides>`.
