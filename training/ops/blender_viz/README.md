# ops/blender_viz: Blender renders of the recorded episodes (T1 Hover, T2 Waypoint, T3 Defend, T4 Approach)

The renderer reads the trajectory `.npz` files the sim recorders already write and keys them into Blender.
It does not simulate, smooth or resample anything, and it does not touch physics or policies.

| File | Role |
|---|---|
| `viz_data.py` | numpy only: `.npz` → scene description (agents + role, positions, orientations, visibility, targets, regions, events, HUD strings). One adapter per recorder schema |
| `drone_model.py` | procedural Tello-class quadrotor (true size 98 × 92 mm, x forward / z up = Isaac body frame) |
| `scene_build.py` | bpy: collections, materials, keyframes, trails, markers, events, 3 cameras, HUD |
| `render.py` | CLI entry: config → build → validate → `.blend` / stills / mp4 |
| `configs/t{1..4}_*.json` | one per task: data file, title, layer toggles |
| `test_viz_data.py` | numpy checks: positions identical to the recording, no teleports inside a trail, captures = `downed`, site centre |
| `../../<render-script>` | `vizjob run render` renders all four tasks on <render-host> and posts stills to Discord |

## Data → Blender mapping (what was checked)

| Task | Recording (recorder) | Frame / units | Orientation | Roles |
|---|---|---|---|---|
| T1 Hover | `docs/results/ws_basic/hover_B_fixedlr_noclip.npz` (`ops/turing_score_basic.py --traj`, Isaac) | z-up m, relative to env origin, 60 Hz | Isaac `root_quat_w` (wxyz) | 1 RL drone (green) + recorded goal (amber) |
| T2 Waypoint | `docs/results/ws_basic/waypoint_fixed_best_agent.npz` (same recorder) | same | same | 1 RL drone + moving goal |
| T3 Defend | `docs/results/blender_viz/data/t3_defend_rl_sparse_s2_stage2_seed7.npz` (`ops/render_final_defend_traj.py`: `policies/defend_rl_sparse_s2` MLP through the `defend_live.py` deploy pipeline in its `EnvWorld` sim, stage 2, seed 7) | EnvWorld sim frame, z-up m, site at (3, 0), 13.5 Hz | defenders: simulated yaw (`EnvWorld.def_yaw`; the velocity model has no tilt). attackers: none recorded → drawn level | 3 defenders green, 2 attacker slots red |
| T4 Approach | `docs/results/blender_viz/data/t4_approach_r2_seed7.npz` (`ops/turing_play_traj.py --traj`, Isaac, `policies/approach_r2` = same md5 as the Turing checkpoint) | z-up m, relative to base (env origin), 20 Hz | Isaac `root_quat_w` | 4 drones of one shared policy (green), gates, pads |

Blender is z-up and metres too, so positions go in unchanged (identity transform; grey x / y arrows mark the sim origin).
Isaac quaternions are wxyz, which is Blender's `rotation_quaternion` order. The model's +x is the camera nose.

Timing: sample *i* sits on frame `1 + i·dt·fps/speed` (sub-frame keys), interpolated linearly between samples, constant
after the end. All drones share the same sample clock, so they stay in sync. Scrub, pause and replay are ordinary
timeline actions on the saved `.blend`. Recordings end at the first `done`, because Isaac resets inside `step()` and the
sample flagged `done` is already the next spawn.

Visual-only choices, labelled in the frame:
- Drone models are scaled up (×5.4 Defend, ×3 Approach, ×1 Hover/Waypoint) so they stay readable. Only the child mesh
  is scaled. The keyed root empty is the recorded position.
- Drones with no recorded orientation are drawn level. `--set visual_heading=true` adds a yaw taken from velocity on a
  separate child empty named `VisualHeading (not sim state)`.
- Props are static blur discs, because rotor speed is not recorded.

Indicators all come from the data:
- Defend: protected site, from the recorded `site_center` and `site_radius` 1.5 m. Its height is 2.5 m (`site_height`).
- Defend: attacker spawn ring, from the stage's `spawn_radius`.
- Defend: interceptions use the exact capture point and capturing defender. The recorder wraps `EnvWorld._events`, so a
  capture that clears a wave and respawns it in the same step is still caught. A capture shows as a flash, then a
  persistent amber ×.
- Defend: a downed attacker turns dark grey and falls along its recorded path until it respawns.
- Approach: keep-out hemisphere (3 m), touchdown zone (1 m) and gate rings (0.4 m, facing the base) come from the
  recording's meta or, for old files, from `tello_approach_env_cfg.py`.
- Approach: gates and pads turn green when the recorded `gate_passed` / `landed` flag sets.
- Optional defender–attacker proximity lines (`proximity_lines`, `proximity_radius`) are computed from recorded positions.

## Run

```bash
# whole pipeline on the lab laptop (GPU, Blender 4.5): validate + .blend + still + mp4 for 3 cameras x 4 tasks
cd ~/hackathon/Action_Space && vizjob run render              # Q=final SAMPLES=32 CAMS=all by default
Q=draft CAMS=Overview vizjob run render -- t3_defend          # quick look at one task
rsync -a <render-host>:as_render/docs/results/blender_viz/ docs/results/blender_viz/   # fetch results

# by hand (any machine with Blender >= 4.2; the home server's 3.0 renders black)
blender -b -P ops/blender_viz/render.py -- --config ops/blender_viz/configs/t3_defend.json \
    --quality final --camera all --validate --save-blend --still 0.6 --anim
    # --quality draft|preview|final|cycles   --res W H   --samples N   --engine EEVEE|CYCLES
    # --camera Overview|Cinematic|TopDown|all   --frames F0 F1   --npz other.npz   --out DIR
    # --set trails=\"both\" speed=0.5 proximity_lines=true drone_scale=1 hud=false ...
python3 ops/blender_viz/test_viz_data.py                      # numpy checks, no Blender needed
```

Config / `--set` keys are listed in `scene_build.DEFAULTS`:
- timing: `fps`, `speed`, `hold_seconds`
- drones: `drone_scale` (0 = auto), `visual_heading`, `roles` (override per name or per role)
- trails and markers: `trails` (`history` | `full` | `both` | `none`), `trail_width`, `start_markers`, `end_markers`,
  `drop_lines`
- events: `capture_bursts`, `capture_marks`, `proximity_lines`, `proximity_radius`
- scene layers: `regions`, `targets`, `target_trails`, `grid`, `axes`, `hud`, `title`
- cameras: `fit_ground`, `overview_elev`, `overview_azim`, `cinematic_elev`, `cinematic_orbit`

Re-running rebuilds only the `DroneViz` collection, so it never duplicates objects. Open the `.blend` with
*Auto Run Python Scripts* on and the HUD clock and counters follow the timeline. The handler is the
`viz_hud_driver.py` text block.

New recordings plug in without code changes:
- Defend: Isaac `scripts/evaluate.py --traj X.npz --checkpoint ... --algorithm MAPPO`, or
  `ops/render_final_defend_traj.py --policy-npz policies/<p>/policy.npz --stage S --seed N --out X.npz`.
- T4 Approach: `ops/turing_approach_traj.sh` with `CK`, `OUT` and `STEPS` set.
- T1 / T2: `ops/turing_score_basic.py --traj`.

Then point a config's `npz` at the new file. Drone counts and durations come from the file.

## Validation (`--validate` → `<name>_validate.json`, the vizjob fails the task if it does not pass)
At every recorded sample the built scene is evaluated at that sample's frame and compared with the file:
- world position error of every drone, which must be < 0.1 mm (measured 0.0)
- orientation error, which must be < 0.1° (measured ≤ 0.04°, float32)
- visibility against the recording
- attackers use the red team material and defenders the green one

The report also lists sim seconds against video seconds, the regions and the event counts.

## Issues found in the earlier `docs/results/final` clips (matplotlib)
- `defend_s2_*`: the protected site was drawn at the origin with r = 0.5 m. In the EnvWorld sim it is at (3, 0) with
  r = 1.5 m. `render_final_defend_traj.py` now records `site_center`, the true radii, `def_yaw`, `def_alive` and capture
  events. The trajectories are bit-identical (same seed).
- `defend_s2_1v1`: only the first wave is 1v1. `EnvWorld._spawn_wave` re-activates the stage's 2 attackers, so it is
  not rendered here.
- `t4_approach_r2_3d`: was made from the round-1 trajectory. Round 2 is now recorded (`t4_approach_r2_seed7.npz`).
