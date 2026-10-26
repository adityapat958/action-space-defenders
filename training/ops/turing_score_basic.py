# Copyright (c) 2026, Action Space. SPDX-License-Identifier: BSD-3-Clause
"""Score T1 Hover / T2 Waypoint: a trained skrl checkpoint or a scripted baseline (ws-basic-tasks).

Run from simulation_infra/ (copied there by ops/turing_basic.sbatch):
    python scripts/score_basic.py --task AS-Tello-Hover-v0 --baseline zero
    python scripts/score_basic.py --task AS-Tello-Waypoint-v0 --checkpoint logs/skrl/.../best_agent.pt

Metrics per finished episode (deterministic policy):
  mean_dist      mean distance to goal over the episode [m]
  tail_dist      mean distance over the last third of the episode [m]
  within_20cm    fraction of steps with distance < 0.2 m
  died           terminated early (too far from goal / hit ground)
  return         undiscounted env reward sum
With --traj, the first --traj_envs envs are recorded for --traj_seconds into an .npz for later
matplotlib / Blender rendering (pos/vel relative to env origin, quat wxyz, action, goal, reward, done, meta json).
Baselines: zero (no command), pd (velocity command = clip(kp * goal_error), the obvious scripted controller).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=512)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--baseline", default=None, choices=["zero", "pd"])
parser.add_argument("--kp", type=float, default=1.5, help="pd baseline gain [1/s]")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--out", default="outputs/score_basic.jsonl")
parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="Env cfg overrides (python literals), must match training, e.g. --set obs_goal_vel=True")
parser.add_argument("--traj", default=None, help="Write a trajectory .npz here (e.g. outputs/traj/hover_best.npz).")
parser.add_argument("--traj_envs", type=int, default=4)
parser.add_argument("--traj_seconds", type=float, default=None, help="Default: one episode length.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import hashlib, json, os, subprocess  # noqa: E402

import numpy as np  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import action_space_kit.tasks  # noqa: E402,F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402


def write_traj(traj, raw):
    """Arrays are (T, envs, ...) float32; agents axis omitted (single drone). quat is wxyz (Isaac convention)."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        commit = None
    ck_sha = hashlib.sha256(open(args.checkpoint, "rb").read()).hexdigest()[:16] if args.checkpoint else None
    meta = dict(task=args.task, label=args.checkpoint or f"baseline:{args.baseline or 'zero'}", checkpoint_sha16=ck_sha,
                seed=args.seed, overrides=args.set, dt=raw.step_dt, num_envs=raw.num_envs, git_commit=commit, kp=args.kp,
                frame="z-up, pos/goal relative to env origin, quat wxyz", episode_length_s=raw.cfg.episode_length_s)
    os.makedirs(os.path.dirname(os.path.abspath(args.traj)), exist_ok=True)
    arrs = {k: np.stack(v).astype(np.float32 if k != "done" else bool) for k, v in traj.items()}
    np.savez_compressed(args.traj, **arrs, dt=raw.step_dt, meta=json.dumps(meta))
    print(f"[score_basic] wrote trajectory {args.traj} ({arrs['pos'].shape[0]} steps x {arrs['pos'].shape[1]} envs)", flush=True)


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    import ast
    for kv in args.set:
        key, value = kv.split("=", 1)
        setattr(env_cfg, key, ast.literal_eval(value))
    genv = gym.make(args.task, cfg=env_cfg)
    raw = genv.unwrapped
    env = SkrlVecEnvWrapper(genv, ml_framework="torch")

    policy = None
    if args.checkpoint:
        from skrl.utils.runner.torch import Runner

        agent_cfg = load_cfg_from_registry(args.task, "skrl_cfg_entry_point")
        agent_cfg["trainer"]["close_environment_at_exit"] = False
        agent_cfg["agent"]["experiment"]["write_interval"] = 0
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        runner = Runner(env, agent_cfg)
        runner.agent.load(os.path.abspath(args.checkpoint))
        if hasattr(runner.agent, "set_running_mode"):
            runner.agent.set_running_mode("eval")
        else:
            runner.agent.enable_training_mode(False)
        policy = runner.agent

    n, dev = raw.num_envs, raw.device
    T = raw.max_episode_length
    dist_hist = torch.zeros(n, T + 2, device=dev)
    k = torch.zeros(n, dtype=torch.long, device=dev)
    ret = torch.zeros(n, device=dev)
    eps = []

    tk = min(args.traj_envs, n)
    t_steps = int((args.traj_seconds or raw.cfg.episode_length_s) / raw.step_dt)
    traj = {key: [] for key in ("pos", "quat", "vel", "action", "goal", "reward", "done")} if args.traj else None

    obs, _ = env.reset()
    while len(eps) < args.episodes or (traj is not None and len(traj["done"]) < t_steps):
        goal_err = raw._goal_pos_w - raw._robot.data.root_pos_w
        with torch.inference_mode():
            if policy is not None:
                out = policy.act(obs, None, timestep=0, timesteps=0)
                act = out[-1].get("mean_actions", out[0])
            elif args.baseline == "pd":
                act = torch.zeros(n, 4, device=dev)
                act[:, :3] = (args.kp * goal_err).clamp(-1.0, 1.0)
            else:
                act = torch.zeros(n, 4, device=dev)
        # distance at the state the action is taken in (pre-step, so auto-reset can't pollute it)
        d = torch.linalg.norm(goal_err, dim=-1)
        if traj is not None:
            org = raw.scene.env_origins[:tk]
            pre = dict(pos=(raw._robot.data.root_pos_w[:tk] - org).cpu().numpy(),
                       quat=raw._robot.data.root_quat_w[:tk].cpu().numpy(),
                       vel=raw._robot.data.root_lin_vel_w[:tk].cpu().numpy(),
                       goal=(raw._goal_pos_w[:tk] - org).cpu().numpy())
        dist_hist[torch.arange(n, device=dev), k.clamp(max=T + 1)] = d
        k += 1
        obs, rew, term, trunc, _ = env.step(act)
        ret += rew.view(-1)
        done = (term | trunc).view(-1)
        if traj is not None and len(traj["done"]) < t_steps:
            # state the action was taken in + the action + its outcome (pre-step state saved below)
            traj["pos"].append(pre["pos"]); traj["quat"].append(pre["quat"]); traj["vel"].append(pre["vel"])
            traj["goal"].append(pre["goal"]); traj["action"].append(act[:tk].float().cpu().numpy())
            traj["reward"].append(rew.view(-1)[:tk].float().cpu().numpy()); traj["done"].append(done[:tk].cpu().numpy())
            if len(traj["done"]) == t_steps:
                write_traj(traj, raw)
        for i in done.nonzero(as_tuple=False).flatten().tolist():
            L = int(k[i])
            h = dist_hist[i, :L]
            tail = h[-max(L // 3, 1):]
            eps.append(dict(len=L, mean_dist=h.mean().item(), tail_dist=tail.mean().item(),
                            within_20cm=(h < 0.2).float().mean().item(),
                            died=bool(term.view(-1)[i].item()), ret=ret[i].item()))
            k[i] = 0
            ret[i] = 0.0

    eps = eps[: args.episodes]
    m = lambda key: sum(float(e[key]) for e in eps) / len(eps)  # noqa: E731
    label = args.checkpoint or f"baseline:{args.baseline or 'zero'}"
    rec = dict(task=args.task, label=label, episodes=len(eps), seed=args.seed, overrides=args.set,
               mean_dist=m("mean_dist"), tail_dist=m("tail_dist"), within_20cm=m("within_20cm"),
               died_rate=m("died"), mean_len_s=m("len") * raw.step_dt, mean_return=m("ret"))
    print("\n=== %s on %s ===" % (label, args.task))
    for key, v in rec.items():
        print(f"{key:12s} {v:.4f}" if isinstance(v, float) else f"{key:12s} {v}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
