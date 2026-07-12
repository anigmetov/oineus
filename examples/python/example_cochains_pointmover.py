"""Point-cloud optimization with birth/death cochains (Weighill & Zhou,
arXiv:2603.25575, Sec. "Point cloud optimization").

Maximizes the persistence content of the longest H1 bar of a small noisy
circle under a unit-disk penalty, and compares against the singleton
("simplices") baseline that puts the whole gradient on the critical
birth/death cells (gradient_method="dgm-loss"). Regular polygons are
critical points of the true persistence, so both runs should drift
toward a rounder, larger loop; the multi-eps cochain loss spreads the
gradient over the cochain supports.

Usage: python example_cochains_pointmover.py [outdir]
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import oineus.diff as od
from oineus.diff.cochains import _select_bar, persistence_content_loss

N_POINTS = 10
N_STEPS = 300
LR = 0.02
EPS_LIST = (0.01, 0.05, 0.1)
SEED = 1


def initial_cloud():
    rng = np.random.default_rng(SEED)
    phi = np.sort(rng.uniform(0, 2 * np.pi, N_POINTS))
    pts = np.column_stack([np.cos(phi), np.sin(phi)]) + 0.1 * rng.standard_normal((N_POINTS, 2))
    return pts


def unit_disk_penalty(pts):
    # keep the cloud from inflating without bound: penalize leaving the unit disk
    r = torch.linalg.norm(pts, dim=1)
    return torch.relu(r - 1.0).pow(2).sum()


def longest_h1_persistence(pts):
    fil = od.vr_filtration(pts.detach().requires_grad_(False), max_dim=2)
    b, d, *_ = _select_bar(fil, 1, "longest")
    return d - b


def run_cochains(pts0):
    pts = torch.tensor(pts0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([pts], lr=LR)
    history, snapshots = [], {}
    for step in range(N_STEPS):
        opt.zero_grad()
        fil = od.vr_filtration(pts, max_dim=2)
        loss = -persistence_content_loss(fil, dim=1, eps=EPS_LIST) + unit_disk_penalty(pts)
        loss.backward()
        opt.step()
        history.append(longest_h1_persistence(pts))
        if step % 100 == 0:
            snapshots[step] = pts.detach().numpy().copy()
    snapshots[N_STEPS] = pts.detach().numpy().copy()
    return history, snapshots


def run_singleton_baseline(pts0):
    pts = torch.tensor(pts0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([pts], lr=LR)
    history, snapshots = [], {}
    for step in range(N_STEPS):
        opt.zero_grad()
        fil = od.vr_filtration(pts, max_dim=2)
        dgm1 = od.persistence_diagram(fil, gradient_method="dgm-loss")[1]
        i = torch.argmax(dgm1[:, 1] - dgm1[:, 0])
        loss = -(dgm1[i, 1] - dgm1[i, 0]) + unit_disk_penalty(pts)
        loss.backward()
        opt.step()
        history.append(longest_h1_persistence(pts))
        if step % 100 == 0:
            snapshots[step] = pts.detach().numpy().copy()
    snapshots[N_STEPS] = pts.detach().numpy().copy()
    return history, snapshots


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    pts0 = initial_cloud()

    hist_c, snap_c = run_cochains(pts0)
    hist_s, snap_s = run_singleton_baseline(pts0)
    print(f"longest H1 persistence after {N_STEPS} steps: "
          f"cochains {hist_c[-1]:.4f}, singleton {hist_s[-1]:.4f} "
          f"(initial {longest_h1_persistence(torch.tensor(pts0)):.4f})")

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    axs[0].plot(hist_c, label="cochain content (multi-eps)")
    axs[0].plot(hist_s, label="singleton (dgm-loss)")
    axs[0].set_xlabel("step")
    axs[0].set_ylabel("longest H1 persistence")
    axs[0].legend()
    for ax, snaps, title in ((axs[1], snap_c, "cochains"), (axs[2], snap_s, "singleton")):
        steps = sorted(snaps)
        for i, step in enumerate(steps):
            p = snaps[step]
            ax.scatter(p[:, 0], p[:, 1], alpha=0.3 + 0.7 * i / max(1, len(steps) - 1),
                       label=f"step {step}", s=25)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.legend(fontsize=7)
    fig.tight_layout()
    out = f"{outdir}/cochains_pointmover.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
