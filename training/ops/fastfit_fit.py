#!/usr/bin/env python3
"""Fit the fast velocity model (tau, accel clamp) to logged step responses (ws-fast-env, Deliverable A).

Input: npz files from scripts/log_step_response.py. For every drone we grid-search (tau, acc) for the xy axes and,
separately, z, by simulating the fast model's exact law (first-order command filter with step clamp, then
acc = clamp((v_cmd_f - v) / tau)) at 120 Hz with the logged held commands and the drone's known command latency.
Prints per-drone fit quantiles and writes step-response plots.

    python ops/fastfit_fit.py outputs/fastfit/cascade.npz outputs/fastfit/velocity.npz --plot docs/results/fastfit
"""
import argparse
import os

import numpy as np
import torch

TAUS = torch.tensor([0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30, 0.35, 0.40, 0.50, 0.65, 0.80])
ACCS = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 14.0])
SUB = 9  # 120 Hz substeps per policy step


def delayed(a, lat):
    """a (T, N, k), lat (N,) policy steps -> command actually applied at each step."""
    T = a.shape[0]
    idx = (torch.arange(T).unsqueeze(1) - lat.unsqueeze(0)).clamp(min=-1)  # (T, N)
    out = torch.zeros_like(a)
    ok = idx >= 0
    gathered = torch.gather(a, 0, idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, a.shape[2]))
    out[ok] = gathered[ok]
    return out


def simulate(target, v0, tau, acc, dt_pol):
    """target (T, M, k) m/s held per policy step; v0 (M, k); tau/acc (M,) -> velocity at the end of each step (T, M, k)."""
    dt = dt_pol / SUB
    T = target.shape[0]
    v = v0.clone()
    vf = v0.clone()  # command filter starts at the current velocity (drones are hovering at t0 anyway)
    tau = tau.unsqueeze(-1)
    lim = acc.unsqueeze(-1)
    alpha = (dt / tau).clamp(max=1.0)
    out = torch.empty_like(target)
    for t in range(T):
        tg = target[t]
        for _ in range(SUB):
            vf = vf + ((tg - vf) * alpha).clamp(-lim * dt, lim * dt)
            v = v + ((vf - v) / tau).clamp(-lim, lim) * dt
        out[t] = v
    return out


def fit(path, max_drones=None):
    d = np.load(path)
    A = torch.tensor(d["actions"])  # (T, E, D, 4)
    V = torch.tensor(d["vel"])
    dt = float(d["dt"])
    T, E, D, _ = A.shape
    A = A.reshape(T, E * D, 4)
    V = V.reshape(T, E * D, 3)
    lat = torch.tensor(d["latency"]).reshape(-1).long() if "latency" in d else torch.zeros(E * D, dtype=torch.long)
    N = E * D if max_drones is None else min(max_drones, E * D)
    A, V, lat = A[:, :N], V[:, :N], lat[:N]
    tgt = delayed(A[..., :3], lat)  # world-frame velocity target, max_lin_speed = 1 m/s
    G = len(TAUS) * len(ACCS)
    tg, ag = torch.meshgrid(TAUS, ACCS, indexing="ij")
    tg, ag = tg.reshape(-1), ag.reshape(-1)
    res = {}
    for name, axes in (("xy", [0, 1]), ("z", [2])):
        k = len(axes)
        # (T, N*G, k): every drone against every grid point
        target = tgt[..., axes].repeat_interleave(G, dim=1)
        meas = V[..., axes].repeat_interleave(G, dim=1)
        tau = tg.repeat(N)
        acc = ag.repeat(N)
        # target at step t applies during step t; V[t] is measured after it. Start from V[0] using steps 1..T-1.
        pred = simulate(target[1:], meas[0], tau, acc, dt)
        err = ((pred - meas[1:]) ** 2).mean(dim=(0, 2)).reshape(N, G)
        best = err.argmin(dim=1)
        rms = err.min(dim=1).values.sqrt()
        # nominal fast model (0.15, 4.0) error for reference
        gi = int(((tg - 0.15).abs() + (ag - 4.0).abs()).argmin())
        res[name] = dict(tau=tg[best].numpy(), acc=ag[best].numpy(), rms=rms.numpy(),
                         rms_nominal=err[:, gi].sqrt().numpy(),
                         rms_zero=((V[1:, :, axes]) ** 2).mean(dim=(0, 2)).sqrt().numpy())
    res["lat"] = lat.numpy()
    res["dt"] = dt
    return res, (A, V, lat, dt)


def q(x):
    p = np.percentile(x, [10, 25, 50, 75, 90])
    return " ".join(f"{v:.3g}" for v in p)


def step_plot(data, fits, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A, V, lat, dt = data
    T, N, _ = A.shape
    # collect single-axis x steps from hover-ish to +-a (|a| > 0.6), align at command change
    win = int(2.0 / dt)
    segs_m, segs_c, segs_a = [], [], []
    for n in range(N):
        a = A[:, n, 0].numpy()
        others = A[:, n, 1:3].abs().sum(-1).numpy()
        ch = np.where(np.abs(np.diff(a)) > 1e-6)[0] + 1
        for t0 in ch:
            if t0 + win >= T or abs(a[t0]) < 0.6 or others[t0] > 1e-6 or abs(a[t0 - 1]) > 1e-6:
                continue
            s = np.sign(a[t0])
            segs_m.append(s * V[t0 - 1:t0 + win, n, 0].numpy())
            segs_c.append(s * a[t0 - 1:t0 + win])
            segs_a.append(n)
    if not segs_m:
        print("no clean steps for plot")
        return
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tt = (np.arange(win + 1) - 1) * dt
    fig, ax = plt.subplots(figsize=(7, 4))
    m = np.array(segs_m)
    ax.plot(tt, np.median(m, 0), "k", lw=2, label=f"logged (median of {len(m)})")
    ax.fill_between(tt, np.percentile(m, 10, 0), np.percentile(m, 90, 0), color="k", alpha=0.15, label="10-90%")
    ax.step(tt, np.median(np.array(segs_c), 0), "r--", where="post", label="command (scaled)")
    for lab, tau, acc, L in fits:
        # simulate a clean step 0 -> 1 m/s with latency L
        steps = win + 1
        tgt = torch.zeros(steps, 1, 1)
        tgt[1 + L:] = 1.0
        p = simulate(tgt, torch.zeros(1, 1), torch.tensor([tau]), torch.tensor([acc]), dt)[:, 0, 0].numpy()
        ax.plot(tt, np.concatenate([[0.0], p[:-1]]), label=lab)
    ax.set_xlabel("time since command step [s]")
    ax.set_ylabel("v along step / |cmd| [m/s]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--max_drones", type=int, default=None)
    ap.add_argument("--plot", default=None, help="output dir for step-response plots")
    args = ap.parse_args()
    for f in args.files:
        res, data = fit(f, args.max_drones)
        name = os.path.splitext(os.path.basename(f))[0]
        print(f"== {name}  ({len(res['lat'])} drones, latency counts {np.bincount(res['lat']).tolist()})")
        for ax in ("xy", "z"):
            r = res[ax]
            print(f"  {ax:2s} tau  p10/25/50/75/90: {q(r['tau'])}")
            print(f"  {ax:2s} acc  p10/25/50/75/90: {q(r['acc'])}")
            print(f"  {ax:2s} rms fit {np.median(r['rms']):.3f}  nominal(0.15,4) {np.median(r['rms_nominal']):.3f}  zero-model {np.median(r['rms_zero']):.3f} m/s (medians)")
        if args.plot:
            mt, ma = float(np.median(res["xy"]["tau"])), float(np.median(res["xy"]["acc"]))
            step_plot(data, [("fast nominal (0.15 s, 4 m/s², L=0)", 0.15, 4.0, 0),
                             (f"fast fitted median ({mt:.2f} s, {ma:.1f} m/s², L=1)", mt, ma, 1)],
                      os.path.join(args.plot, f"step_x_{name}.png"), f"x step response: {name}")
        np.savez(os.path.join(os.path.dirname(f), f"fit_{name}.npz"),
                 **{f"{ax}_{k}": v for ax in ("xy", "z") for k, v in res[ax].items()}, lat=res["lat"])


if __name__ == "__main__":
    main()
