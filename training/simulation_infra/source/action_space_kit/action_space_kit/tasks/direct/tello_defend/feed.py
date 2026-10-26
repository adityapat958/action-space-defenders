"""Position-feed model for the defend task (pure torch, no Isaac imports).

On the day we get attacker positions from the organizers' AprilTag tracker, with unknown lag,
noise and missed frames, and positions only (no velocities). This module turns true positions
into that feed and runs the **same** alpha-beta filter the live runner uses
(``drone_deployment_infra/defend_live.py`` should copy :func:`alpha_beta_step` verbatim, in numpy).
Training on the filtered estimate keeps the policy's inputs identical between sim and hardware.
"""

from __future__ import annotations

import torch


def alpha_beta_step(est_pos, est_vel, meas, measured, dt: float, alpha: float, beta: float):
    """One predict + correct step of a constant-velocity alpha-beta filter.

    Works on any matching shapes (..., 3); ``measured`` is (...,) bool, False = frame missed
    (predict only). Returns the new (est_pos, est_vel). Pure arithmetic, so the numpy port is a copy.
    """
    pred = est_pos + est_vel * dt
    resid = (meas - pred) * measured.unsqueeze(-1).to(pred.dtype)
    return pred + alpha * resid, est_vel + (beta / dt) * resid


class AttackerFeed:
    """Delayed, noisy, dropping position feed for (E, A) targets, plus the alpha-beta estimate.

    ``delay_s`` is sampled per env per episode from ``delay_range``; ``noise`` is the position
    std-dev [m]; ``dropout`` the per-frame miss probability.
    """

    def __init__(self, num_envs: int, num_targets: int, step_dt: float, delay_range, noise: float, dropout: float,
                 alpha: float, beta: float, device):
        self.E, self.A, self.dt = num_envs, num_targets, step_dt
        self.delay_range, self.noise, self.dropout = tuple(delay_range), noise, dropout
        self.alpha, self.beta = alpha, beta
        self.device = device
        self.max_lag = max(0, int(round(self.delay_range[1] / step_dt)))
        L = self.max_lag + 1
        self.buf = torch.zeros(num_envs, L, num_targets, 3, device=device)
        self.ptr = 0
        self.lag = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.est_pos = torch.zeros(num_envs, num_targets, 3, device=device)
        self.est_vel = torch.zeros(num_envs, num_targets, 3, device=device)

    def reset(self, env_ids: torch.Tensor, true_pos: torch.Tensor) -> None:
        """New episode for ``env_ids``: fill history with the current position, resample the lag."""
        if len(env_ids) == 0:
            return
        pos = true_pos[env_ids]
        self.buf[env_ids] = pos.unsqueeze(1)
        self.est_pos[env_ids] = pos
        self.est_vel[env_ids] = 0.0
        lo, hi = self.delay_range
        lag_s = torch.rand(len(env_ids), device=self.device) * (hi - lo) + lo
        self.lag[env_ids] = torch.clamp(torch.round(lag_s / self.dt).long(), 0, self.max_lag)

    def reset_mask(self, mask: torch.Tensor, true_pos: torch.Tensor) -> None:
        """Re-seed the (E, A) slots in ``mask`` (e.g. a respawned wave) at their current position."""
        if not mask.any():
            return
        m = mask.unsqueeze(-1)
        self.buf = torch.where(m.unsqueeze(1), true_pos.unsqueeze(1).expand_as(self.buf), self.buf)
        self.est_pos = torch.where(m, true_pos, self.est_pos)
        self.est_vel = torch.where(m, torch.zeros_like(true_pos), self.est_vel)

    def step(self, true_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Push this step's true positions, return the filtered (est_pos, est_vel)."""
        L = self.buf.shape[1]
        self.ptr = (self.ptr + 1) % L
        self.buf[:, self.ptr] = true_pos
        idx = (self.ptr - self.lag) % L  # (E,)
        meas = self.buf[torch.arange(self.E, device=self.device), idx]  # (E, A, 3)
        if self.noise > 0.0:
            meas = meas + torch.randn_like(meas) * self.noise
        measured = torch.rand(self.E, self.A, device=self.device) >= self.dropout
        self.est_pos, self.est_vel = alpha_beta_step(
            self.est_pos, self.est_vel, meas, measured, self.dt, self.alpha, self.beta
        )
        return self.est_pos, self.est_vel
