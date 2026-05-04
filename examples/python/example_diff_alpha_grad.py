#!/usr/bin/env python3
"""Visualize gradients of the differentiable alpha (and Cech-Delaunay)
filtration on a 2D point cloud.

The point cloud has interesting topology: a large outer ring, a small
nested ring, and noisy clutter that produces low-persistence H1 features.

We pick four H1 diagram points -- 2 high-persistence (the rings) and
2 low-persistence (clutter) -- assign each a target (b, d), build a
quadratic loss, backprop, and visualize:

    Figure 1: Alpha-only.
        - left:  the point cloud, with gradient arrows on the four
                 simplices' vertices that contributed to the loss.
        - right: the H1 diagram with gradient arrows on every diagram
                 point (oin.plot_diagram_gradient).

    Figure 2: Alpha vs Cech-Delaunay overlay.
        - left:  point cloud with point gradients in two colors,
                 alpha (orange) and Cech-Delaunay (blue), for the same
                 targets. The diagrams are identical so the same loss
                 applies to both.
        - right: combined diagram with two gradient fields overlaid.

Run::

    PYTHONPATH=<diode-build>:<oineus-bindings> python example_diff_alpha_grad.py

It writes ``diff_alpha_grad.png`` and ``diff_alpha_vs_cd_grad.png`` in
the current directory.
"""
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import oineus as oin
import oineus.diff as oin_diff


# ---------------------------------------------------------------------
# Point cloud
# ---------------------------------------------------------------------

def make_points(seed: int = 0) -> np.ndarray:
    """A topology-rich 2D point cloud with two nested noisy rings and clutter.

    Note on alpha vs Cech-Delaunay: in 2D Delaunay, every triangle is
    Gabriel (the circumdisk is empty), and the triangles that pair as
    H1 deaths tend to be acute -- because Delaunay maximizes the minimum
    angle. As a result, the alpha and Cech-Delaunay critical values
    coincide on every simplex involved in finite H1 pairs, so the two
    differentiable filtrations also return identical *gradients* on
    those pairs. The example demonstrates this empirically: the two
    point-gradient fields lie on top of one another. (Gradients diverge
    in 3D, where many tetrahedra have non-circumsphere MEBs and many
    triangles are obtuse.)
    """
    rng = np.random.default_rng(seed)

    # Outer ring at radius 1, ~50 points, light radial noise.
    theta_outer = np.linspace(0.0, 2 * np.pi, 50, endpoint=False)
    outer = np.stack([np.cos(theta_outer), np.sin(theta_outer)], axis=1)
    outer = outer + 0.04 * rng.standard_normal(outer.shape)

    # Inner ring at radius 0.30, ~20 points.
    theta_inner = np.linspace(0.0, 2 * np.pi, 20, endpoint=False)
    inner = 0.30 * np.stack([np.cos(theta_inner), np.sin(theta_inner)], axis=1)
    inner = inner + 0.02 * rng.standard_normal(inner.shape)

    # Clutter: 20 random points to seed low-persistence H1.
    clutter = rng.uniform(-1.2, 1.2, size=(20, 2))

    return np.concatenate([outer, inner, clutter], axis=0).astype(np.float64)


# ---------------------------------------------------------------------
# Target selection: 2 high-persistence + 2 low-persistence H1 points
# ---------------------------------------------------------------------

def select_targets(dgm1_np: np.ndarray):
    """Pick 2 high-persistence and 2 mid/low-persistence finite H1 points
    and assign each a target ``(b, d)``.

    Targets:
        - high-persistence: kill the cycle (target death = birth).
        - low-persistence: grow the bar to large persistence (target
          death = birth + max_pers).

    The "low" picks come from a percentile band, not the very tail
    near the diagonal -- otherwise they're invisible on the plot.

    Returns a list of ``(row_index, target_b, target_d)``.
    """
    finite = np.isfinite(dgm1_np[:, 1])
    rows = np.flatnonzero(finite)
    pers = dgm1_np[rows, 1] - dgm1_np[rows, 0]
    order = np.argsort(pers)
    rows_sorted = rows[order]
    pers_sorted = pers[order]

    max_pers = pers_sorted[-1]

    # Low: pick from positions ~50%-70% of the sorted list (mid-low band).
    n = len(rows_sorted)
    low_idx = [int(0.55 * n), int(0.70 * n)]
    low_rows = rows_sorted[low_idx]

    # High: top two.
    high_rows = rows_sorted[-2:]

    targets = []
    for r in high_rows:
        b, d = dgm1_np[r]
        # Push toward the diagonal (kill the cycle).
        targets.append((int(r), float(b), float(b)))
    for r in low_rows:
        b, d = dgm1_np[r]
        # Push the death far above birth so the bar would grow.
        targets.append((int(r), float(b), float(b + max_pers)))
    return targets


def loss_from_targets(dgm1: torch.Tensor, targets):
    """Quadratic loss pulling selected dgm1 rows toward target (b, d)."""
    losses = []
    for r, tb, td in targets:
        cur = dgm1[r]
        tgt = torch.tensor([tb, td], dtype=cur.dtype, device=cur.device)
        losses.append((cur - tgt).pow(2).sum())
    return torch.stack(losses).sum()


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def plot_points_with_grads(ax, points_np, grads_dict, title):
    """Scatter the point cloud and overlay one gradient quiver per
    entry in ``grads_dict`` (mapping label -> (gradient_array, color)).
    """
    ax.scatter(points_np[:, 0], points_np[:, 1], s=8,
               color="black", alpha=0.6, zorder=2)

    for label, (g, color) in grads_dict.items():
        mag = np.linalg.norm(g, axis=1)
        if mag.max() <= 0:
            continue
        # Show arrows for any nonzero gradient. Auto-scale so the largest
        # arrow has length ~25% of the cloud's extent.
        extent = float(np.ptp(points_np[:, 0]) + np.ptp(points_np[:, 1])) * 0.5
        scale_for_quiver = mag.max() / (0.25 * extent + 1e-12)
        mask = mag > 1e-12
        ax.quiver(
            points_np[mask, 0], points_np[mask, 1],
            -g[mask, 0], -g[mask, 1],   # negative grad = descent direction
            color=color, label=label,
            angles="xy", scale_units="xy",
            scale=scale_for_quiver,
            width=0.005, alpha=0.85, zorder=3,
        )

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)


# ---------------------------------------------------------------------
# Main: build filtrations, apply loss, plot
# ---------------------------------------------------------------------

def compute_grads(filtration_fn, points_np, targets=None):
    """Build a differentiable filtration via filtration_fn(points), pick
    targets if not supplied, run backward, return:

        grad_points : (n, 2) ndarray
        dgm1_np     : (m, 2) ndarray of H1 birth/death values
        dgm1_grad   : (m, 2) ndarray of dL/d(b,d) on the diagram
        targets     : list of (row, target_b, target_d)
    """
    pts = torch.tensor(points_np, dtype=torch.float64, requires_grad=True)
    fil = filtration_fn(pts)
    dgms = oin_diff.persistence_diagram(fil, dualize=False)
    dgm1 = dgms[1]
    dgm1.retain_grad()
    if targets is None:
        targets = select_targets(dgm1.detach().cpu().numpy())
    loss = loss_from_targets(dgm1, targets)
    loss.backward()
    return (
        pts.grad.detach().cpu().numpy(),
        dgm1.detach().cpu().numpy(),
        dgm1.grad.detach().cpu().numpy() if dgm1.grad is not None else np.zeros_like(dgm1.detach().cpu().numpy()),
        targets,
    )


def main():
    out_dir = os.getcwd()

    points_np = make_points(seed=0)
    print(f"point cloud: n={len(points_np)}")

    # Pick targets from the alpha diagram first; reuse for Cech-Delaunay
    # (combinatorics are identical, so the same row indices align).
    grad_pts_alpha, dgm1_alpha, dgm1_grad_alpha, targets = compute_grads(
        oin_diff.alpha_filtration, points_np, targets=None
    )
    print("targets (row, target_b, target_d):")
    for t in targets:
        print(" ", t)

    # ----- Figure 1: alpha-only -----
    fig1, axes1 = plt.subplots(1, 2, figsize=(11, 5))
    plot_points_with_grads(
        axes1[0],
        points_np,
        {"alpha (descent)": (grad_pts_alpha, "tab:orange")},
        "Point gradients (alpha filtration)",
    )

    oin.plot_diagram_gradient(
        {1: dgm1_alpha},
        gradient={1: dgm1_grad_alpha},
        ax=axes1[1],
        descent=True,
        title="H1 gradient on diagram (alpha)",
        grad_color="tab:orange",
    )

    # Mark targets on the diagram for context.
    for r, tb, td in targets:
        axes1[1].plot([tb], [td], marker="*", color="tab:red",
                      markersize=12, zorder=5)
    fig1.tight_layout()
    out1 = os.path.join(out_dir, "diff_alpha_grad.png")
    fig1.savefig(out1, dpi=300)
    plt.close(fig1)
    print(f"saved {out1}")

    # ----- Figure 2: alpha vs Cech-Delaunay overlay -----
    grad_pts_cd, dgm1_cd, dgm1_grad_cd, _ = compute_grads(
        oin_diff.cech_delaunay_filtration, points_np, targets=targets
    )

    # Sanity: combinatorics identical -> dgm1 row count matches.
    assert dgm1_alpha.shape == dgm1_cd.shape, (
        f"H1 diagram shape mismatch: alpha={dgm1_alpha.shape}, cd={dgm1_cd.shape}"
    )

    diff_pt_max = float(np.abs(grad_pts_alpha - grad_pts_cd).max())
    diff_dgm_max = float(np.abs(dgm1_grad_alpha - dgm1_grad_cd).max())
    print(f"max |alpha_grad - cd_grad| on points: {diff_pt_max:.3e}")
    print(f"max |alpha_grad - cd_grad| on diagram: {diff_dgm_max:.3e}")

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5.5))

    # Plot CD first (thicker, semi-transparent), then alpha on top
    # (thinner, opaque) so both are visible when they overlap.
    plot_points_with_grads(
        axes2[0],
        points_np,
        {"Cech-Delaunay (descent)": (grad_pts_cd, "tab:blue")},
        "Point gradients: alpha vs Cech-Delaunay\n"
        f"(max grad diff: {diff_pt_max:.2e})",
    )
    # Overlay alpha. Reuse the helper but shrink arrow widths for the
    # top layer so both are distinguishable at points where they differ.
    mag = np.linalg.norm(grad_pts_alpha, axis=1)
    if mag.max() > 0:
        extent = float(np.ptp(points_np[:, 0]) + np.ptp(points_np[:, 1])) * 0.5
        scale = mag.max() / (0.25 * extent + 1e-12)
        mask = mag > 1e-12
        axes2[0].quiver(
            points_np[mask, 0], points_np[mask, 1],
            -grad_pts_alpha[mask, 0], -grad_pts_alpha[mask, 1],
            color="tab:orange", label="alpha (descent)",
            angles="xy", scale_units="xy", scale=scale,
            width=0.0025, alpha=0.9, zorder=4,
        )
    axes2[0].legend(loc="upper right", fontsize=8)

    # Overlay both gradient fields on the same diagram. Use plot_points
    # only on the first call so we don't double-scatter.
    oin.plot_diagram_gradient(
        {1: dgm1_cd},
        gradient={1: dgm1_grad_cd},
        ax=axes2[1],
        descent=True,
        title="H1 gradients (alpha = orange, CD = blue)\n"
              f"(max diagram-grad diff: {diff_dgm_max:.2e})",
        grad_color="tab:blue",
    )
    oin.plot_diagram_gradient(
        {1: dgm1_alpha},
        gradient={1: dgm1_grad_alpha},
        ax=axes2[1],
        descent=True,
        plot_points=False,
        grad_color="tab:orange",
        quiver_style={"width": 0.003, "alpha": 0.9},
    )
    for _, tb, td in targets:
        axes2[1].plot([tb], [td], marker="*", color="tab:red",
                      markersize=12, zorder=5)

    fig2.tight_layout()
    out2 = os.path.join(out_dir, "diff_alpha_vs_cd_grad.png")
    fig2.savefig(out2, dpi=300)
    plt.close(fig2)
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
