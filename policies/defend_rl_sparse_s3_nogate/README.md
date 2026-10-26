# defend_rl_sparse_s3_nogate: MAPPO sparse, curriculum stage 3, FAILED the Isaac gate (breach rule)

**Not a gate pass.** Exported because it is the first RL policy that beats assign-lead-safe at S3 in the deploy
pipeline. Whether to fly it is an orchestrator / ws-deploy decision (see the trade-off below). Default deploy policy
stays scripted `assign-lead-safe`.

| | |
|---|---|
| workstream | ws-defend-sparse, exported by ws-export (2026-10-24 13:45 UTC) |
| task | `AS-Defend-Live-S3-v0` = AS-Defend-Live-v0 + `curriculum_stage=3`, `reward_mode=sparse` |
| algorithm | skrl MAPPO, one policy per defender (`defender_0..2`), [256,128,64] ELU, RunningStandardScaler obs, obs dim 53, act dim 4 |
| training | Turing job **2405221** ladder (STD_RESET 0.4), S3 try 2: 16384 envs, MB 16, 600 iters (9600 timesteps), warm start from S3 t1 |
| runtime physics | FAST_PHYSICS for stages 0–3. `env.yaml` says cascade: dumped before `apply_curriculum` |
| source ckpt | `turing:<cluster-log-path>/skrl/tello_defend/2026-10-24_12-33-27_mappo_torch_sparse_s3_cascade_t2/checkpoints/agent_9600.pt` (sha256 `818369d9…e71c60932`) |
| **gate (S3 try2, 256 ep)** | 8.58 downed/ep vs assign-lead-safe 7.98 (1.07×) but **breach 5.9% vs 1.2%** (+4.7 pts > +2 allowed) → **pass=0**. Crashes 0.152 vs 0.055 (+0.097, inside +0.1). Ladder 2405221 STOPPED after S3 failed twice (t1: 8.31, breach 5.1%) |

## Checks
- Parity (`export_policy --check`): torch fp64 vs numpy **8.9e-15**, deployed (clamp ±1) 4.0e-15, fp32 2.8e-6.
- defend_live `--dry-run --feed synthetic --ticks 400 --seed 0`: n=1 **1/1**, n=2 **2/2**, n=3 **3/3**, n=4 **4/4**
  (first RL export to take all 4; sparse_s2 got 1/4, assign-lead 4/4). Fence interventions ~2400 / 1200 drone-ticks, as before.

## Deploy pipeline stage check (ws-deploy v1.2, `--dry-run --feed stage:K --episodes 64`)

| stage | this (seed 0) | breach | assign-lead-safe | breach |
|---|---|---|---|---|
| 0 | 9.31 ± 0.29 | 0% | 6.94 | 0% |
| 1 | 9.50 ± 0.31 | 0% | 8.45 | 0% |
| 2 | 8.64 ± 0.17 | 0% | 8.00 | 0% |
| **3** | **8.33 ± 0.21** | 3.1% | 7.92 ± 0.14 | 0% |
| 3 (seed 1) | **8.44 ± 0.16** | 3.1% | 8.23 ± 0.13 | 0% |

Crashes 0 everywhere (the pipeline's separation layer removes Isaac's 0.15/ep). S3 pooled: **8.38 vs 8.08 (+0.30, ≈1.5 SE)**
but breach 3.1% (2/64 episodes per seed) vs 0%. Trade-off: +4% take-downs for ~3% of episodes where an attacker reaches the site.

Re-export: `python3 drone_deployment_infra/export_policy.py agent_9600.pt policies/defend_rl_sparse_s3_nogate/policy.npz
--agent-yaml agent.yaml --env-yaml env.yaml --check`

checkpoint: agent_9600.pt (sha256 `818369d9…71c60932`)
