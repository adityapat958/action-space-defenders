# defend_rl_sparse_s2: MAPPO sparse, curriculum stage 2 (gate pass, std reset)

**Stage-2 policy, NOT a deploy policy yet.** At S3 (venue-like) it loses to assign-lead-safe (4.77 vs 7.92 downed/ep,
breach 48%). Deploy fallback stays scripted `assign-lead-safe` until an RL export beats 7.92 at S3 in the pipeline.

| | |
|---|---|
| workstream | ws-defend-sparse, exported by ws-export |
| task | `AS-Defend-Live-S2-v0` = AS-Defend-Live-v0 + `curriculum_stage=2`, `reward_mode=sparse` |
| algorithm | skrl MAPPO, one policy per defender (`defender_0..2`), [256,128,64] ELU, RunningStandardScaler obs, obs dim 53, act dim 4 |
| training | Turing job **2405221** (A30) ladder S2→S6, 16384 envs, MB 16, 600 iters/try (9600 timesteps), warm start from S2 t2 (`…09-29-56_mappo_torch_sparse_s2_cascade_t2/agent_9600.pt`) with **STD_RESET=0.4** (action std re-inflated; earlier S2 tries collapsed to std 0.03–0.06 and failed 6.39 / 6.67) |
| runtime physics | FAST_PHYSICS for stages 0–3 (velocity model). `env.yaml` says cascade: dumped before `apply_curriculum` |
| source ckpt | `turing:<cluster-log-path>/skrl/tello_defend/2026-10-24_10-28-56_mappo_torch_sparse_s2_cascade_t1/checkpoints/agent_9600.pt` (sha256 `8fb24361…ddbe4e05`, also in `policy.npz` meta) |
| **gate (S2 try1, 256 ep)** | **8.125 downed/ep** vs assign-lead-safe 8.066 → PASS (relaxed S1–S3 gate: ≥0.95×, breach ≤ +2 pts, crashes ≤ +0.1). Breach 1.95% (scripted 0.78%), crashes 0.078/ep (scripted 0.016) |

## Checks (2026-10-24 ~11:50 UTC)
- Parity (`export_policy --check`): torch fp64 vs numpy **7.1e-15**, deployed path (clamp ±1) **2.7e-15**, fp32 3.3e-6.
- defend_live `--dry-run --feed synthetic --ticks 400 --seed 0`: n=1 **1/1**, n=2 **2/2**, n=3 **3/3**, n=4 1/4 (S1/S0 exports: 0/4; assign-lead 4/4). Fence interventions ~2200/1200 drone-ticks, like the earlier exports.

## Closed-loop stage check (defend_live `--dry-run --feed stage:K --episodes 64`, v1.2 pipeline with max_speed √3)

| stage | downed/ep (this) | breach | crashes | assign-lead-safe (same harness, v1.2) | sparse_s1 (same) | Isaac gate |
|---|---|---|---|---|---|---|
| 0 | **8.86 ± 0.24** | 0% | 0 | 6.94 | 8.42 | — |
| 1 | **9.08 ± 0.33** | 0% | 0 | 8.45 | 7.69 | — |
| 2 | **8.20 ± 0.19** | 0% | 0 | 8.00 | 5.14 | 8.125 |
| 3 | 4.77 ± 0.36 | 48% | 0 | **7.92** | 0.80 | — |

Own-stage pipeline score matches Isaac (8.20 vs 8.125): the 17–20% gap seen at 10:40 is closed by ws-deploy v1.2.
Beats lead-safe at S0–S2 on this harness, loses badly at S3 (more/faster attackers). p99 tick 1.9 ms.

Re-export: `python3 drone_deployment_infra/export_policy.py agent_9600.pt policies/defend_rl_sparse_s2/policy.npz
--agent-yaml agent.yaml --env-yaml env.yaml --check`

checkpoint: agent_9600.pt (sha256 `8fb24361…ddbe4e05`)
