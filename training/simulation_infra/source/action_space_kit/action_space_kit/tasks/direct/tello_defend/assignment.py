"""Defender -> attacker target assignment for the defend task (pure torch, no Isaac imports).

The stock reward pulls every defender towards its *nearest* attacker, so in a 2v2 both
defenders usually chase the same one. Here each step picks one joint assignment that
minimises the slowest interception (then the total), with every live attacker covered
when there are enough defenders. Brute force over all A^D mappings: 4 for 2v2, 256 for 4v4,
which is trivial on a GPU.

Shared with ``scripts/evaluate.py`` (assign-* baselines) and DAgger so reward, observation
and expert all agree on who is chasing whom.
"""

from __future__ import annotations

import itertools

import torch

DEAD_TTG = 1.0e4  # [s] time-to-go given to downed attackers
TTG_CAP = 100.0  # [s] cap for live pairs. An attacker as fast as the defender, flying away, gives
# t -> inf from the intercept quadratic; uncapped, that outweighs the coverage penalty and
# sends both defenders after the same attacker.


def intercept_time(rel: torch.Tensor, v_att: torch.Tensor, speed: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Time and aim vector to reach a constant-velocity target at ``speed``.

    ``rel`` = target - self and ``v_att`` = target velocity, both (..., 3). Solves
    |rel + v_att t| = speed t for the smallest t > 0. Falls back to pure pursuit
    (t = |rel| / speed, aim = rel) when the target can't be caught.
    """
    a = (v_att * v_att).sum(-1) - speed**2
    b = 2.0 * (rel * v_att).sum(-1)
    c = (rel * rel).sum(-1)
    direct = torch.sqrt(c) / speed
    disc = b * b - 4.0 * a * c
    sq = torch.sqrt(torch.clamp(disc, min=0.0))
    safe_a = torch.where(a.abs() < 1e-6, torch.full_like(a, -1e-6), a)
    t1, t2 = (-b - sq) / (2.0 * safe_a), (-b + sq) / (2.0 * safe_a)
    big = torch.full_like(t1, float("inf"))
    t = torch.minimum(torch.where(t1 > 0, t1, big), torch.where(t2 > 0, t2, big))
    ok = (disc >= 0) & torch.isfinite(t)
    t = torch.where(ok, t, direct)
    aim = torch.where(ok.unsqueeze(-1), rel + v_att * t.unsqueeze(-1), rel)
    return t, aim


_COMBO_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}


def all_assignments(num_defenders: int, num_attackers: int, device) -> torch.Tensor:
    """Every defender->attacker mapping (repeats allowed), shape (K, D)."""
    key = (num_defenders, num_attackers, str(device))
    if key not in _COMBO_CACHE:
        _COMBO_CACHE[key] = torch.tensor(
            list(itertools.product(range(num_attackers), repeat=num_defenders)), device=device, dtype=torch.long
        )
    return _COMBO_CACHE[key]


def assignment_costs(
    ttg: torch.Tensor, alive: torch.Tensor, mode: str = "minmax_ttg", def_alive: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cost of every joint assignment.

    Args:
        ttg: (E, D, A) time-to-go per defender/attacker pair (dead attackers already masked or not).
        alive: (E, A) bool.
        mode: ``minmax_ttg`` (slowest intercept, then total: the assign-* baseline rule) or
            ``min_sum_ttg`` (total time-to-go).
        def_alive: optional (E, D) bool. A downed defender costs nothing and covers nothing.
    Returns:
        combos (K, D) and cost (E, K).
    """
    E, D, A = ttg.shape
    ttg = torch.nan_to_num(ttg, nan=TTG_CAP, posinf=TTG_CAP).clamp(max=TTG_CAP)
    ttg = torch.where(alive.unsqueeze(1), ttg, torch.full_like(ttg, DEAD_TTG))
    if def_alive is not None:
        ttg = torch.where(def_alive.unsqueeze(-1), ttg, torch.zeros_like(ttg))
    combos = all_assignments(D, A, ttg.device)  # (K, D)
    K = combos.shape[0]
    # cost_per[e, k, d] = ttg[e, d, combos[k, d]]
    cost_per = torch.gather(ttg.unsqueeze(1).expand(E, K, D, A), 3, combos.view(1, K, D, 1).expand(E, K, D, 1)).squeeze(-1)
    if mode == "minmax_ttg":
        base = cost_per.max(-1).values + 0.01 * cost_per.sum(-1)
    elif mode == "min_sum_ttg":
        base = cost_per.sum(-1)
    else:
        raise ValueError(f"unknown assignment mode {mode!r}")
    # coverage: each live attacker left without a defender (while one could be spared) costs a lot
    if def_alive is None:
        onehot = torch.nn.functional.one_hot(combos, A).sum(1).clamp(max=1)  # (K, A)
        covered = (onehot.unsqueeze(0) * alive.unsqueeze(1)).sum(-1)  # (E, K)
        need = torch.clamp(alive.sum(-1, keepdim=True), max=D)
    else:
        # only live defenders cover: (E, K, D, A) one-hot masked by def_alive
        oh = torch.nn.functional.one_hot(combos, A).unsqueeze(0) * def_alive.view(E, 1, D, 1)
        onehot = oh.sum(2).clamp(max=1)  # (E, K, A)
        covered = (onehot * alive.unsqueeze(1)).sum(-1)
        need = torch.minimum(alive.sum(-1, keepdim=True), def_alive.sum(-1, keepdim=True))
    cost = base + 1.0e3 * (need - covered).clamp(min=0)
    return combos, cost


def assign_targets(
    ttg: torch.Tensor,
    alive: torch.Tensor,
    mode: str = "minmax_ttg",
    previous: torch.Tensor | None = None,
    hysteresis: float = 0.0,
    def_alive: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pick one target per defender, shape (E, D) long.

    ``previous`` (E, D) is the last assignment. It is kept unless the best alternative is at
    least ``hysteresis`` seconds cheaper, or it points at a downed attacker. This stops
    near-tie flip-flopping, which would otherwise make the dense reward chatter.
    """
    ttg = torch.nan_to_num(ttg, nan=TTG_CAP, posinf=TTG_CAP).clamp(max=TTG_CAP)
    if mode == "nearest":
        masked = torch.where(alive.unsqueeze(1), ttg, torch.full_like(ttg, DEAD_TTG))
        return masked.argmin(-1)
    combos, cost = assignment_costs(ttg, alive, mode, def_alive)
    best = cost.argmin(-1)  # (E,)
    target = combos[best]
    if previous is not None and hysteresis > 0.0:
        # index of the previous assignment in combos: base-A digits, first defender most significant
        D, A = ttg.shape[1], ttg.shape[2]
        weights = (A ** torch.arange(D - 1, -1, -1, device=ttg.device)).long()
        prev_idx = (previous.long() * weights).sum(-1)
        prev_cost = cost.gather(1, prev_idx.unsqueeze(-1)).squeeze(-1)
        best_cost = cost.gather(1, best.unsqueeze(-1)).squeeze(-1)
        prev_valid = alive.gather(1, previous.long()).all(-1)
        keep = prev_valid & (prev_cost - best_cost < hysteresis)
        target = torch.where(keep.unsqueeze(-1), previous.long(), target)
    return target


def slot_order(ttg: torch.Tensor, alive: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-defender attacker slot permutation, shape (E, D, A).

    Slot 0 holds the defender's assigned attacker. The remaining live attackers follow by
    time-to-go, and downed ones come last. Sorted slots give the MLP cheap permutation
    invariance.
    """
    E, D, A = ttg.shape
    ttg = torch.nan_to_num(ttg, nan=TTG_CAP, posinf=TTG_CAP).clamp(max=TTG_CAP)
    key = torch.where(alive.unsqueeze(1), ttg, torch.full_like(ttg, DEAD_TTG)).clone()
    key = key + DEAD_TTG * (~alive).unsqueeze(1).float()  # dead strictly after any live one
    key.scatter_(2, target.unsqueeze(-1), -1.0)
    return key.argsort(dim=-1)
