# Copyright (c) 2026, Action Space. SPDX-License-Identifier: BSD-3-Clause
"""Log defender velocity responses to random step commands, for fitting the fast physics model (ws-fast-env).

Each defender flies a random sequence of held world-frame commands [vx, vy, vz, yaw_rate] (segments 0.7-2.0 s,
sign-alternating so drones stay near their start). Attackers are parked far away and all crash bounds are off,
so nothing resets during the log.

    python scripts/log_step_response.py --model cascade --out outputs/fastfit/cascade.npz
    python scripts/log_step_response.py --model velocity --out outputs/fastfit/velocity.npz
    python scripts/log_step_response.py --model fast_dr --out outputs/fastfit/fast_dr.npz   # velocity + FAST_DR
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["cascade", "velocity", "fast_dr"], required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seconds", type=float, default=40.0)
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import numpy as np
import torch

import action_space_kit.tasks  # noqa: F401
from action_space_kit.tasks.direct.tello_defend import tello_defend_env_cfg as C
from isaaclab_tasks.utils import parse_env_cfg

TASK = "AS-Defend-Live-v0"


def main():
    cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.seed = args_cli.seed
    cfg.curriculum_stage = -1
    cfg.episode_length_s = args_cli.seconds + 10.0
    cfg.active_attackers = [1, 1]
    cfg.attacker_difficulty = -1.0
    cfg.attacker_speed = [0.0, 0.0]
    cfg.spawn_radius = 25.0
    cfg.feed_model = False
    cfg.crash_z_min = None
    cfg.crash_z_max = None
    cfg.arena_radius = None
    cfg.collide_radius = None
    cfg.defender_loss_prob = 0.0
    phys = C.REAL_PHYSICS if args_cli.model == "cascade" else C.FAST_PHYSICS
    cfg.dynamics_model = phys["dynamics_model"]
    cfg.sim.dt = phys["dt"]
    cfg.decimation = phys["decimation"]
    cfg.sim.render_interval = phys["decimation"]
    if args_cli.model == "fast_dr":
        for k, v in C.FAST_DR.items():
            setattr(cfg, k, list(v))

    env = gym.make(TASK, cfg=cfg).unwrapped
    env.reset()
    E, D, dev = env.num_envs, env.cfg.num_defenders, env.device
    dt = env.step_dt
    T = int(args_cli.seconds / dt)
    g = torch.Generator(device=dev).manual_seed(args_cli.seed)

    cmd = torch.zeros(E, D, 4, device=dev)
    left = torch.zeros(E, D, device=dev)
    sign = torch.ones(E, D, 1, device=dev)
    A, V, W, Z = (np.zeros((T, E, D, k), np.float32) for k in (4, 3, 1, 1))
    for t in range(T):
        new = left <= 0
        if new.any():
            # half the segments are single-axis steps (clean for fitting), half full random 4-D commands
            mag = torch.rand(E, D, 4, device=dev, generator=g) * 2 - 1
            axis = torch.randint(0, 4, (E, D), device=dev, generator=g)
            single = torch.rand(E, D, device=dev, generator=g) < 0.5
            onehot = torch.nn.functional.one_hot(axis, 4).float()
            seg = torch.where(single.unsqueeze(-1), onehot * mag.abs() * sign, mag)
            seg[..., 3] *= 0.5  # keep yaw moderate
            # 15% of segments: hover (step down to zero)
            seg = torch.where((torch.rand(E, D, 1, device=dev, generator=g) < 0.15), torch.zeros_like(seg), seg)
            cmd = torch.where(new.unsqueeze(-1), seg, cmd)
            sign = torch.where(new.unsqueeze(-1), -sign, sign)
            left = torch.where(new, 0.7 + 1.3 * torch.rand(E, D, device=dev, generator=g), left)
        actions = {a: cmd[:, i] for i, a in enumerate(env.cfg.possible_agents)}
        env.step(actions)
        left -= dt
        A[t] = cmd.cpu().numpy()
        V[t] = torch.stack([d.data.root_lin_vel_w for d in env._defenders], 1).cpu().numpy()
        W[t] = torch.stack([d.data.root_ang_vel_w[:, 2:3] for d in env._defenders], 1).cpu().numpy()
        Z[t] = torch.stack([d.data.root_pos_w[:, 2:3] for d in env._defenders], 1).cpu().numpy()
        if t % 100 == 0:
            print(f"[log] {args_cli.model} step {t}/{T}", flush=True)
            if t > 0:
                save(env, A[:t + 1], V[:t + 1], W[:t + 1], Z[:t + 1], dt)
    save(env, A, V, W, Z, dt)
    env.close()


def save(env, A, V, W, Z, dt):
    E, D = env.num_envs, env.cfg.num_defenders
    extra = {}
    if env._cascade is not None:
        extra["latency"] = env._cascade.latency.view(E, D).cpu().numpy()
    elif getattr(env, "_fast_dr", False):
        extra["latency"] = torch.stack([c.latency for c in env._defender_ctl], 1).cpu().numpy()
        extra["tau"] = torch.stack([c.tau for c in env._defender_ctl], 1).cpu().numpy()
        extra["acc_xy"] = torch.stack([c.acc_xy for c in env._defender_ctl], 1).cpu().numpy()
    resets = int(env.episode_length_buf.min().item())
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    np.savez_compressed(args_cli.out, actions=A, vel=V, yaw_rate=W, z=Z, dt=dt, physics_dt=env.physics_dt,
                        model=args_cli.model, min_episode_len=resets, **extra)
    print(f"[log] saved {args_cli.out} T={len(A)} E={E} D={D} dt={dt:.4f} min_ep_len={resets}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
