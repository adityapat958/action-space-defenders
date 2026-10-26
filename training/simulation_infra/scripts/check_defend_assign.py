# Copyright (c) 2026, Action Space. SPDX-License-Identifier: BSD-3-Clause
"""Smoke check for target assignment in the defend task (ws-assign-reward).

    python scripts/check_defend_assign.py --task AS-Defend-Assign-v0 --steps 1200
    python scripts/check_defend_assign.py --task AS-Defend-v0 --steps 1200   # stock, for contrast

Drives every defender straight at whatever sits in its observation slot 0. Under
AS-Defend-Assign-v0 that is the assigned attacker, so this obs-only policy should match the
assign-chase baseline; under AS-Defend-v0 slot 0 is always attacker 0. Checks:
  1. obs slot 0 == assigned attacker (rel pos matches simulator state)
  2. no target duplicated while enough attackers are alive
  3. the dense closing term never spikes (respawn / reassignment handled)
and prints downed-per-episode, breach rate and per-step reward stats.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="AS-Defend-Assign-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import ast

import gymnasium as gym
import torch

import action_space_kit.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
for kv in args.set:
    k, v = kv.split("=", 1)
    setattr(cfg, k, ast.literal_eval(v))
env = gym.make(args.task, cfg=cfg)
raw = env.unwrapped
obs, _ = env.reset()
E, D, A = raw.num_envs, raw.cfg.num_defenders, raw.cfg.num_attackers
assigned_mode = raw._uses_assignment()
print(f"[check] task={args.task} assignment={raw.cfg.assignment} obs_assigned_first={raw.cfg.obs_assigned_first} "
      f"team_share={raw.cfg.rew_capture_team_share} D={D} A={A}")

slot_err_max, dup_steps, multi_steps = 0.0, 0, 0
rew_all, ep_done, ep_breach, ep_captures = [], 0, 0, 0.0
shaping_max = 0.0
print("[check] env ready, stepping", flush=True)
for step in range(args.steps):
    if step % 200 == 0:
        print(f"[check] step {step}", flush=True)
    # check 1/2 on the observation the policy is about to act on
    if raw.cfg.obs_assigned_first:
        att_pos, _ = raw._attacker_states()
        for i, d in enumerate(raw._defenders):
            tgt = raw.assigned_targets[:, i]
            truth = (att_pos[torch.arange(E), tgt] - d.data.root_pos_w).clamp(-raw.cfg.obs_clip_pos, raw.cfg.obs_clip_pos)
            # a wave cleared this step leaves every attacker down until the next step's respawn
            live = raw._attacker_alive[torch.arange(E), tgt]
            if live.any():
                err = (obs[f"defender_{i}"][:, 9:12] - truth).abs().max(-1).values[live].max().item()
                slot_err_max = max(slot_err_max, err)
    if assigned_mode:
        enough = raw._attacker_alive.sum(-1) >= D
        tg = raw.assigned_targets
        dup = torch.zeros(E, dtype=torch.bool, device=raw.device)
        for a in range(D):
            for b in range(a + 1, D):
                dup |= tg[:, a] == tg[:, b]
        dup_steps += (dup & enough).sum().item()
        multi_steps += enough.sum().item()

    actions = {}
    for i, agent in enumerate(raw.cfg.possible_agents):
        rel = obs[agent][:, 9:12]
        direction = rel / torch.clamp(torch.linalg.norm(rel, dim=-1, keepdim=True), min=1e-6)
        actions[agent] = torch.cat([direction, torch.zeros(E, 1, device=raw.device)], dim=-1)
    obs, rew, term, trunc, _ = env.step(actions)
    r = torch.stack([rew[a] for a in raw.cfg.possible_agents], dim=-1)
    rew_all.append(r)
    # steps without any event (no capture, breach or clear) isolate the dense terms
    quiet = (r.abs() < 9.0).all(-1)
    if quiet.any():
        shaping_max = max(shaping_max, r[quiet].abs().max().item())
    fin = raw.outcome_valid
    ep_done += fin.sum().item()
    ep_breach += raw.outcome_breached[fin].sum().item()
    ep_captures += raw.outcome_captures[fin].sum().item()

R = torch.cat(rew_all)
print(f"[check] slot0 max abs error vs sim state: {slot_err_max:.4f} m  -> {'OK' if slot_err_max < 1e-3 else 'FAIL'}"
      if raw.cfg.obs_assigned_first else "[check] slot0 check skipped (stock obs)")
if assigned_mode:
    frac = dup_steps / max(multi_steps, 1)
    print(f"[check] duplicate-target fraction (>=D alive): {frac:.4f} -> {'OK' if frac < 1e-6 else 'FAIL'}")
    print(f"[check] assignment switches per env over run: {raw.assignment_switches.mean().item():.2f}")
print(f"[check] dense-only step reward max |r|: {shaping_max:.3f} (closing clamp*scale = {raw.cfg.rew_scale_closing:.1f})"
      f" -> {'OK' if shaping_max <= raw.cfg.rew_scale_closing + 0.5 else 'FAIL'}")
print(f"[check] reward per step mean {R.mean().item():.4f} std {R.std().item():.4f} min {R.min().item():.2f} max {R.max().item():.2f}")
print(f"[check] episodes finished {ep_done}; breach rate {ep_breach / max(ep_done, 1):.3f}; "
      f"downed per episode {ep_captures / max(ep_done, 1):.2f}")
print(f"[check] finite rewards: {bool(torch.isfinite(R).all())}")
env.close()
app.close()
