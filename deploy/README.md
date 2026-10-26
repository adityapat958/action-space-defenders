# team4: defender package (3 def v 2 att, organizers' infra)


The organizers run the attackers, telemetry and AprilTag poses. We supply only the defender policy.

## Frame check (do this first)

The frame conversion was broken (`ydown_zin` / 1.5 m origin). Fixed to `yup_zout` / 1.22 m (matches
`map.yaml` frame string and `map_tags.py`'s `C_FLIP`). Always verify before flying for real.

### Step 1 — dry run (hand-carry the drone)

```bash
cd /path/to/organizer/Action_Space/drone_deployment_infra/swarm_ops
# No --fly: prints map-frame xyz every tick. Walk the drone around by hand.
python3 swarm_policies.py --run frame_check:1
```

**What to check:**
- Carry drone to **tag 10** → printed `map≈(0.0, 0.0, 0.0)` (tag 10 is the origin).
- Carry drone to **tag 6** face → printed `map≈(+3.16, -0.62, +4.21)` (from map.yaml).
- Floor level near tag 2 → `sim z ≈ 0`.

If the signs are wrong, the convention fix did not apply — re-run `deploy/install.sh (from this repo)`.

### Step 2 — live flight (1 drone, ~60 s)

```bash
python3 swarm_policies.py --run frame_check:1 --fly --duration 60
```

Expected sequence:
1. Climbs to 1.0 m.
2. Flies to WP1 = 1 m in front of tag 10 → prints `sim≈(1.0, 0.0, 1.22)`.
3. Yaws 180° (back to tag 10, face into room).
4. Flies to WP2 = 1 m off tag 6 along its normal → hovers 3 s.
5. Lands.

## Install (once, ~10 s)
```bash
deploy/install.sh (from this repo) /path/to/organizer/Action_Space   # copies 4 .py + weights, registers defend_rl / defend_lead
cd /path/to/organizer/Action_Space/drone_deployment_infra/swarm_ops
```

## 15-min slot: back to back (defenders = drones 1,2,3; change the numbers to match their fleet)
```bash
# A) Adi: sparse_s2 (S2 gate 8.125 vs scripted 8.066)
DEFEND_RL_WEIGHTS=../../deploy/policies/adi_sparse_s2/policy.npz python3 swarm_policies.py --run defend_rl:1,2,3            # dry-run, 5 s
DEFEND_RL_WEIGHTS=../../deploy/policies/adi_sparse_s2/policy.npz python3 swarm_policies.py --run defend_rl:1,2,3 --fly --duration 120

# B) Aryan: s7_c3 (stage 7 clean feed: 5.67 downed/ep, 90.8% stop rate; best of b3/c3/e3/f3)
DEFEND_RL_WEIGHTS=../../deploy/policies/aryan_s7_c3/policy.npz python3 swarm_policies.py --run defend_rl:1,2,3 --fly --duration 120

# Fallback (classical assign-lead-safe), if RL misbehaves
python3 swarm_policies.py --run defend_lead:1,2,3 --fly --duration 120
```
Abort: Ctrl+C in that terminal (lands all drones). Telemetry port 9100 (`DEFEND_TELEM_PORT` to change).
Pose older than 1 s or missing → the defender hovers. Aryan's README notes: with a delayed or noisy feed, s7 collapses.

## Contents
`policies/adi_sparse_s2/` (policy.npz + agent_9600.pt), `policies/aryan_s7_c3/` (policy.npz + best_agent.pt),
`drone_deployment_infra/{defend_live.py, swarm_ops/policies/defend_rl.py, defend_lead.py, attack_feed.py}`. Numpy only.
Check: both npz load in DefendRLPolicy with 2 and 3 defenders (MLP 53→256→128→64, 3 heads). With no telemetry → hover (0,0,z,0).

## Altitude controller

`defend_rl` and `defend_lead` use their **own** height controller (not `base.hold_height`),
so we can climb faster than the organizers' 30 rc-unit cap.

| Env var | Default | Notes |
|---|---|---|
| `DEFEND_HEIGHT_CM` | `180` | target hover altitude (cm above takeoff) |
| `DEFEND_MAX_RC_Z`  | `60`  | rc climb/descend cap (was 30 in base.py) |

Same P-gain (0.6) and deadband (4 cm) as `base.py`. At 60 units the drone
climbs ~1 m/s, so 80 cm → 180 cm takes ≈2 s (vs ≈4–5 s at cap 30).

Unit test (no real drone needed):
```bash
python3 deploy/tools/test_height_ctrl.py
# h=80  → rc_z=60  (capped at DEFEND_MAX_RC_Z)
# h=178 → rc_z=0   (in 4 cm deadband)
```

To lower the target (e.g. for a low-ceiling venue):
```bash
DEFEND_HEIGHT_CM=120 DEFEND_MAX_RC_Z=60 python3 swarm_policies.py --run defend_rl:1,2,3 --fly
```

## UPDATE (run 2): team-aware defenders, S3 policy added

**Policies (all 3 in the zip):**
| dir | what | note |
|---|---|---|
| `policies/adi_sparse_s3/` | **try first.** sparse S3 (t2, no gate pass) | beats scripted at all 4 deploy-check stages (s2 only 1/4) |
| `policies/adi_sparse_s2/` | sparse S2, gate pass 8.125 | flew run 1, took down 1 attacker |
| `policies/aryan_s7_c3/` | Aryan stage 7, 90.8% stop rate (clean feed) | weak with a delayed or noisy feed |

```bash
cd <org repo>/drone_deployment_infra/swarm_ops
DEFEND_RL_WEIGHTS=../../deploy/policies/adi_sparse_s3/policy.npz python3 swarm_policies.py --run defend_rl:1,2,3 --fly --duration 120
DEFEND_RL_WEIGHTS=../../deploy/policies/adi_sparse_s2/policy.npz python3 swarm_policies.py --run defend_rl:1,2,3 --fly --duration 120
DEFEND_RL_WEIGHTS=../../deploy/policies/aryan_s7_c3/policy.npz  python3 swarm_policies.py --run defend_rl:1,2,3 --fly --duration 120
python3 swarm_policies.py --run defend_lead:1,2,3 --fly --duration 120   # classical fallback
```

**What changed since run 1:**
- Defenders now see their TEAMMATES, as in training. All 3 are computed together each tick (obs D=3, per-agent MLP heads).
  In run 1 each drone saw itself alone (teammates zero = "dead").
- Safety layer active: 0.5 m hard separation (back off + vertical split), 0.8 m soft (closing motion removed), geofence.
- Own and attacker velocities are estimated (finite difference); they were 0 before. Separate assignment memory per drone.
- Altitude target 1.8 m, fast climb: `DEFEND_HEIGHT_CM=180`, `DEFEND_MAX_RC_Z=60`.
- Logs: every tick → `logs/` (`DEFEND_LOG_DIR`). Summarise with `python3 deploy/tools/replay_log.py <file.jsonl>`.
- Optional takeoff fix (drones landing one by one): `deploy/install.sh (from this repo) <org repo> --patch-runner` (keepalive rc 0 during takeoff).
- install.sh now copies ALL our policy .py files (the run-1 list missed new modules).
