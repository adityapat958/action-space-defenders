# ws-basic-tasks: run T1 Hover / T2 Waypoint / T4 Approach on a Turing GPU

For a teammate (or their agent) who already holds a Turing GPU (e.g. RTX PRO 6000 via `srun`/`salloc`).
Owner: team4. Results → STATUS.md §4 + `outputs/ws_basic_scores.jsonl`.

## What it does
- **T1 Hover / T2 Waypoint** (skrl PPO, 1024 envs): scores two scripted baselines (`zero`, `pd` = fly straight at the goal),
  trains 400 / 800 iterations, scores `best_agent.pt` + the last checkpoint with `ops/turing_score_basic.py`
  (mean / last-third distance to goal, % steps < 20 cm, early-termination rate, return; 512 episodes).
- **T4 Approach** (SB3, 512 envs × 4 drones): retrains with the shipped `policies/approach_it5_room_fixed/command.txt` recipe
  (800 iterations) and scores with `scripts/sb3/play.py` (256 ep). Shipped policy reference: success **0.891**.

## Prerequisites
1. Repo: `git clone <your-fork> ~/Action_Space` (or `git pull`).
   **Must include the body-frame force fix** in `tello_hover_env.py` (`quat_apply_inverse`); the script refuses T1/T2 without it.
   Without the fix, any yaw rotates the thrust with the body and the drone runs away (the env passed a world-frame force to
   `set_external_force_and_torque`, which Isaac Lab 2.3.2 applies in the body frame).
2. Isaac Sim 5.1 + Isaac Lab 2.3.2 env with `action_space_kit` installed (`simulation_infra/setup.sh`). Physics only, so Blackwell is fine
   (only RTX rendering crashes there).
3. Tello USD at `$TELLO_USD_PATH` (convert from `simulation_infra/assets/tello/tello.urdf` if missing, see `as_run.sh`).

## Run (inside a GPU allocation)
```bash
cd ~/Action_Space
export REPO=$PWD ISAAC_ENV=/path/to/your/isaac_env.sh TELLO_USD_PATH=/path/to/tello.usd   # defaults: ~/Action_Space, ~/isaac/tools/isaac_env.sh, ~/isaac/assets/as_kit/tello.usd
T=waypoint bash ops/turing_basic.sbatch > simulation_infra/outputs/ws_basic_waypoint.log 2>&1   # or hover / approach
```
Optional: `ITERS=1500` to train longer. One task per GPU process; three fit on one 96 GB card in parallel.

Or submit as a batch job: `sbatch -J wsb_hover --export=ALL,T=hover ops/turing_basic.sbatch` (defaults to `-A cs549 -p academic`;
override with `-A/-p/--gres`; `long` queues for days, `quick` / `academic` start within minutes).

## Outputs
- `simulation_infra/outputs/ws_basic_scores.jsonl`: one JSON line per scored policy/baseline.
- Checkpoints: `simulation_infra/logs/skrl/tello_{hover,waypoint}/<date>_ppo_torch/checkpoints/`, `simulation_infra/logs/sb3/AS-Tello-Approach-v0/<date>/model.zip`.
- Pass bar: trained policy has lower `tail_dist` and `died_rate` than the `pd` baseline (T1/T2); success ≥ ~0.85 (T4).
