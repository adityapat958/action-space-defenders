"""Per-episode randomized velocity model: the fast stand-in for the cascade firmware model (ws-fast-env).

Same law as ``TelloVelocityController`` (first-order command filter + force tracking, both with time constant tau,
acceleration clamps), but tau, the accel clamps and a policy-step command latency are sampled per environment at
reset. Used by AS-Defend-Live-v0 only when one of the ``fast_*`` cfg lists is non-empty; otherwise the env keeps the
plain controller, so existing runs are unchanged.

Pure torch, no Isaac imports: the pure-torch env (Deliverable B) reuses it.
"""

from __future__ import annotations

import torch

from action_space_kit.control import TelloCommandCfg

G = 9.81


def _sample(rng: list, nominal: float, n: int, device) -> torch.Tensor:
    if not rng:
        return torch.full((n,), float(nominal), device=device)
    lo, hi = float(rng[0]), float(rng[-1])
    return lo + (hi - lo) * torch.rand(n, device=device)


class RandomizedVelocityController:
    def __init__(self, cfg: TelloCommandCfg, num_envs: int, mass: float, inertia_zz: float, device,
                 response_time: list, max_accel: list, max_accel_z: list, latency_steps: list,
                 response_time_z: list | None = None):
        self.cfg, self.n, self.mass, self.inertia_zz, self.device = cfg, num_envs, mass, inertia_zz, device
        self.rt_rng, self.acc_rng = list(response_time), list(max_accel)
        self.rtz_rng = list(response_time_z or []) or list(response_time)
        self.accz_rng = list(max_accel_z) or list(max_accel)
        self.lat_choices = [int(x) for x in latency_steps] or [0]
        n = num_envs
        self.tau = torch.full((n,), cfg.response_time, device=device)
        self.tau_z = torch.full((n,), cfg.response_time, device=device)
        self.acc_xy = torch.full((n,), cfg.max_lin_accel, device=device)
        self.acc_z = torch.full((n,), cfg.max_lin_accel, device=device)
        self.latency = torch.zeros(n, dtype=torch.long, device=device)
        self._hist = torch.zeros(n, max(self.lat_choices) + 1, 4, device=device)
        self._cmd = torch.zeros(n, 4, device=device)
        self._lin_vel_cmd = torch.zeros(n, 3, device=device)
        self._yaw_rate_cmd = torch.zeros(n, device=device)
        self.reset(None)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = torch.arange(self.n, device=self.device) if env_ids is None else torch.as_tensor(env_ids, device=self.device)
        k = len(ids)
        self.tau[ids] = _sample(self.rt_rng, self.cfg.response_time, k, self.device)
        self.tau_z[ids] = _sample(self.rtz_rng, self.cfg.response_time, k, self.device)
        self.acc_xy[ids] = _sample(self.acc_rng, self.cfg.max_lin_accel, k, self.device)
        self.acc_z[ids] = _sample(self.accz_rng, self.cfg.max_lin_accel, k, self.device)
        ch = torch.tensor(self.lat_choices, device=self.device)
        self.latency[ids] = ch[torch.randint(len(ch), (k,), device=self.device)]
        self._hist[ids] = 0.0
        self._cmd[ids] = 0.0
        self._lin_vel_cmd[ids] = 0.0
        self._yaw_rate_cmd[ids] = 0.0

    def set_command(self, actions: torch.Tensor) -> None:
        """Latch a new policy action (once per policy step), applying the per-env latency."""
        self._hist = torch.roll(self._hist, shifts=1, dims=1)
        self._hist[:, 0] = actions.clamp(-1.0, 1.0)
        self._cmd = torch.gather(self._hist, 1, self.latency.view(-1, 1, 1).expand(-1, 1, 4)).squeeze(1)

    def accel(self, lin_vel_w: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance the command filter one physics step and return the commanded world acceleration (n, 3)."""
        tau = self.tau.clamp_min(1e-6)
        tau3 = torch.stack([self.tau, self.tau, self.tau_z], dim=-1).clamp_min(1e-6)
        lin_target = self._cmd[:, :3] * self.cfg.max_lin_speed
        yaw_target = self._cmd[:, 3] * self.cfg.max_yaw_rate
        alpha = (dt / tau).clamp(max=1.0)
        lim = torch.stack([self.acc_xy, self.acc_xy, self.acc_z], dim=-1)
        step = ((lin_target - self._lin_vel_cmd) * (dt / tau3).clamp(max=1.0)).clamp(-lim * dt, lim * dt)
        self._lin_vel_cmd = self._lin_vel_cmd + step
        ystep = ((yaw_target - self._yaw_rate_cmd) * alpha).clamp(-self.cfg.max_yaw_accel * dt, self.cfg.max_yaw_accel * dt)
        self._yaw_rate_cmd = self._yaw_rate_cmd + ystep
        acc = (self._lin_vel_cmd - lin_vel_w) / tau3
        return torch.maximum(torch.minimum(acc, lim), -lim)

    def compute(self, lin_vel_w, ang_vel_b, projected_gravity_b, dt: float):
        """(force_w, torque_b), each (n, 1, 3), same convention as TelloVelocityController.compute."""
        force_w = self.mass * self.accel(lin_vel_w, dt)
        force_w[:, 2] += self.mass * G
        c = self.cfg
        torque_b = torch.zeros_like(force_w)
        torque_b[:, 0] = -c.attitude_kp * projected_gravity_b[:, 1] - c.attitude_kd * ang_vel_b[:, 0]
        torque_b[:, 1] = c.attitude_kp * projected_gravity_b[:, 0] - c.attitude_kd * ang_vel_b[:, 1]
        torque_b[:, 2] = self.inertia_zz * (self._yaw_rate_cmd - ang_vel_b[:, 2]) / self.tau.clamp_min(1e-6)
        torque_b[:, :2] *= self.inertia_zz
        return force_w.unsqueeze(1), torque_b.unsqueeze(1)
