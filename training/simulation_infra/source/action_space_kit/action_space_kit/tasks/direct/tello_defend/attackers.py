"""Scripted attacker behaviours with one difficulty knob (pure torch, no Isaac imports).

We don't know how the attackers will fly on the day, so training samples a *mixture* of
behaviours whose hardness scales with ``difficulty`` in [0, 1]:

==========  ===========  ====================================================================
difficulty  speed [m/s]  behaviour mix (each attacker samples one per wave)
==========  ===========  ====================================================================
0.0         0.2-0.4      straight dive at the site
0.3         0.4-0.8      + sinusoidal weave (lateral + vertical)
0.6         0.6-1.0      + detour through a random waypoint, loiter-then-dash
1.0         0.8-1.2      + reactive evasion: sidestep the nearest defender inside ``evade_radius``
==========  ===========  ====================================================================
"""

from __future__ import annotations

import math

import torch

STRAIGHT, WEAVE, WAYPOINT, DASH = 0, 1, 2, 3


def difficulty_params(d: float) -> dict:
    """Speed range, behaviour probabilities and evasion probability for a difficulty."""
    d = float(min(max(d, 0.0), 1.0))
    probs = torch.tensor([
        max(0.1, 1.0 - 1.2 * d),          # straight
        d,                                 # weave
        max(0.0, 1.5 * (d - 0.3)),         # waypoint detour
        max(0.0, 1.5 * (d - 0.5)),         # loiter then dash
    ])
    return dict(
        speed=(0.2 + 0.6 * d, 0.4 + 0.8 * d),
        probs=probs / probs.sum(),
        evade_prob=min(max(1.6 * (d - 0.5), 0.0), 0.8),
        weave_amp=0.4 + 0.6 * d,
    )


class AttackerBrain:
    """Per-(env, attacker) behaviour state and the velocity command it produces."""

    def __init__(self, num_envs: int, num_attackers: int, device):
        E, A = num_envs, num_attackers
        self.device = device
        self.behaviour = torch.zeros(E, A, dtype=torch.long, device=device)
        self.speed = torch.zeros(E, A, device=device)
        self.evasive = torch.zeros(E, A, dtype=torch.bool, device=device)
        self.waypoint = torch.zeros(E, A, 3, device=device)  # world frame
        self.reached = torch.zeros(E, A, dtype=torch.bool, device=device)
        self.hold = torch.zeros(E, A, device=device)  # [s] loiter time for DASH
        self.phase = torch.zeros(E, A, device=device)
        self.amp = torch.zeros(E, A, device=device)
        self.t = torch.zeros(E, A, device=device)  # [s] since this attacker spawned

    def sample(self, mask: torch.Tensor, spawn_pos: torch.Tensor, site: torch.Tensor, difficulty: float,
               speed_range=None) -> None:
        """(Re)draw behaviours for the (E, A) slots in ``mask``. ``site`` is (E, 3) world."""
        if not mask.any():
            return
        p = difficulty_params(difficulty)
        E, A = mask.shape
        lo, hi = speed_range if speed_range is not None else p["speed"]
        beh = torch.multinomial(p["probs"].to(self.device), E * A, replacement=True).view(E, A)
        self.behaviour = torch.where(mask, beh, self.behaviour)
        spd = torch.rand(E, A, device=self.device) * (hi - lo) + lo
        self.speed = torch.where(mask, spd, self.speed)
        ev = torch.rand(E, A, device=self.device) < p["evade_prob"]
        self.evasive = torch.where(mask, ev, self.evasive)
        # waypoint: halfway to the site, pushed sideways by 1.5-3 m, at 1-2.5 m height
        to_site = site.unsqueeze(1) - spawn_pos
        lateral = torch.stack([-to_site[..., 1], to_site[..., 0], torch.zeros_like(to_site[..., 0])], -1)
        lateral = lateral / lateral.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        side = torch.where(torch.rand(E, A, device=self.device) < 0.5, -1.0, 1.0)
        off = (torch.rand(E, A, device=self.device) * 1.5 + 1.5) * side
        wp = spawn_pos + 0.5 * to_site + lateral * off.unsqueeze(-1)
        wp[..., 2] = site[:, 2].unsqueeze(1) - 1.25 + torch.rand(E, A, device=self.device) * 1.5 + 1.0
        self.waypoint = torch.where(mask.unsqueeze(-1), wp, self.waypoint)
        self.reached = torch.where(mask, torch.zeros_like(self.reached), self.reached)
        self.hold = torch.where(mask, torch.rand(E, A, device=self.device) * 3.0 + 1.0, self.hold)
        self.phase = torch.where(mask, torch.rand(E, A, device=self.device) * 2 * math.pi, self.phase)
        self.amp = torch.where(mask, torch.full_like(self.amp, p["weave_amp"]), self.amp)
        self.t = torch.where(mask, torch.zeros_like(self.t), self.t)

    def advance(self, dt: float) -> None:
        self.t += dt

    def velocity(self, pos: torch.Tensor, site: torch.Tensor, def_pos: torch.Tensor, def_alive: torch.Tensor,
                 evade_radius: float, max_speed: float) -> torch.Tensor:
        """Desired world velocity (E, A, 3) [m/s]. ``def_pos`` (E, D, 3), ``def_alive`` (E, D) bool."""
        to_site = site.unsqueeze(1) - pos
        to_wp = self.waypoint - pos
        self.reached |= (self.behaviour == WAYPOINT) & (to_wp.norm(dim=-1) < 0.5)
        go_wp = (self.behaviour == WAYPOINT) & ~self.reached
        goal = torch.where(go_wp.unsqueeze(-1), to_wp, to_site)
        d = goal / goal.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        lateral = torch.stack([-d[..., 1], d[..., 0], torch.zeros_like(d[..., 0])], -1)
        lateral = lateral / lateral.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        weave = (self.behaviour == WEAVE).float() * self.amp
        s = torch.sin(2.0 * self.t + self.phase)
        up = torch.zeros_like(d)
        up[..., 2] = 1.0
        d = d + weave.unsqueeze(-1) * (s.unsqueeze(-1) * lateral + 0.3 * torch.cos(1.3 * self.t + self.phase).unsqueeze(-1) * up)

        # reactive evasion: sidestep (perpendicular to the goal direction) away from the nearest live defender
        rel = pos.unsqueeze(2) - def_pos.unsqueeze(1)  # (E, A, D, 3) defender -> attacker
        dist = rel.norm(dim=-1)
        dist = torch.where(def_alive.unsqueeze(1), dist, torch.full_like(dist, 1e3))
        dmin, j = dist.min(dim=-1)
        away = torch.gather(rel, 2, j[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
        away = away - (away * d).sum(-1, keepdim=True) * d / (d * d).sum(-1, keepdim=True).clamp(min=1e-6)
        away = away / away.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        w = (self.evasive & (dmin < evade_radius)).float() * 2.0 * (1.0 - dmin / evade_radius).clamp(min=0.0)
        d = d + w.unsqueeze(-1) * away
        d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        factor = torch.ones_like(self.speed)
        dash = self.behaviour == DASH
        factor = torch.where(dash & (self.t < self.hold), torch.full_like(factor, 0.15), factor)
        factor = torch.where(dash & (self.t >= self.hold), torch.full_like(factor, 1.4), factor)
        speed = (self.speed * factor).clamp(max=max_speed)
        return d * speed.unsqueeze(-1)
