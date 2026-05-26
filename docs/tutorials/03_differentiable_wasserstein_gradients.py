#!/usr/bin/env python3
"""Differentiable Wasserstein gradient visualization.

This is the script version of
``03_differentiable_wasserstein_gradients.ipynb``. It samples clean and
noisy unit-circle point clouds as torch tensors, computes differentiable
H1 Vietoris-Rips diagrams, and overlays diagram-space descent directions
from Wasserstein and sliced Wasserstein losses.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch


def make_circle_samples(n: int, noise_scale: float, seed: int):
    """Return clean and noisy unit-circle samples as torch tensors."""
    angles = torch.arange(n) * (2.0 * math.pi / n)
    clean_points = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)

    torch.manual_seed(seed)
    noisy_points = (clean_points + noise_scale * torch.randn_like(clean_points)).clone()
    noisy_points.requires_grad_(True)

    return clean_points, noisy_points


def h1_vr_diagram(points: torch.Tensor, oin_diff, n_threads: int) -> torch.Tensor:
    """Compute a differentiable H1 diagram from a point cloud."""
    fil = oin_diff.vr_filtration(points, max_dim=2, n_threads=n_threads)
    dgms = oin_diff.persistence_diagram(
        fil,
        dualize=True,
        include_inf_points=False,
        n_threads=n_threads,
    )
    return dgms[1]


def diagram_bounds(*diagrams: torch.Tensor, pad_fraction: float = 0.12) -> dict[str, float]:
    """Axis bounds shared by the diagram plots."""
    pts = torch.cat([d.detach() for d in diagrams], dim=0)
    pts = pts[torch.isfinite(pts).all(dim=1)]
    lo = float(pts.min())
    hi = float(pts.max())
    span = max(hi - lo, 1.0)
    pad = pad_fraction * span
    return {"xmin": lo - pad, "xmax": hi + pad, "ymin": lo - pad, "ymax": hi + pad}


def save_or_show(fig, path: Path, show: bool):
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if show:
        fig.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("diff_wasserstein_gradient_figures"))
    parser.add_argument("--show", action="store_true", help="Show figures interactively after saving them.")
    parser.add_argument("--n", type=int, default=50, help="Number of clean unit-circle samples.")
    parser.add_argument("--noise-scale", type=float, default=0.08, help="Gaussian coordinate noise scale.")
    parser.add_argument("--seed", type=int, default=7, help="Seed for the noisy point cloud.")
    parser.add_argument("--slice-seed", type=int, default=11, help="Seed for sliced Wasserstein directions.")
    parser.add_argument("--n-directions", type=int, default=256, help="Number of sliced Wasserstein directions.")
    parser.add_argument("--n-threads", type=int, default=4, help="Oineus reduction thread count.")
    args = parser.parse_args()

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import oineus as oin
    import oineus.diff as oin_diff

    torch.set_default_dtype(torch.float64)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    clean_points, noisy_points = make_circle_samples(args.n, args.noise_scale, args.seed)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(clean_points[:, 0], clean_points[:, 1], s=30, label="clean", color="0.25")
    ax.scatter(
        noisy_points.detach()[:, 0],
        noisy_points.detach()[:, 1],
        s=30,
        label="noisy",
        color="tab:red",
        alpha=0.8,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Unit-circle samples")
    ax.legend(frameon=False)
    save_or_show(fig, args.output_dir / "circle_samples.png", args.show)

    clean_h1 = h1_vr_diagram(clean_points, oin_diff, args.n_threads).detach()
    noisy_h1 = h1_vr_diagram(noisy_points, oin_diff, args.n_threads)
    noisy_h1.retain_grad()
    axis_bounds = diagram_bounds(clean_h1, noisy_h1)

    fig, ax = plt.subplots(figsize=(5, 5))
    oin.plot_diagram(
        {1: clean_h1.detach().cpu().numpy()},
        ax=ax,
        color={1: "0.2"},
        axis_bounds=axis_bounds,
        point_style={"s": 60},
        dim_label_fmt="clean H{dim}",
    )
    oin.plot_diagram(
        {1: noisy_h1.detach().cpu().numpy()},
        ax=ax,
        color={1: "tab:red"},
        axis_bounds=axis_bounds,
        point_style={"s": 60},
        dim_label_fmt="noisy H{dim}",
    )
    ax.set_title("Clean and noisy H1 diagrams")
    ax.legend(frameon=False)
    save_or_show(fig, args.output_dir / "h1_diagrams.png", args.show)

    wasserstein_loss = oin_diff.wasserstein_cost(
        noisy_h1,
        clean_h1,
        wasserstein_q=2.0,
        wasserstein_delta=0.01,
        ignore_inf_points=True,
    )
    wasserstein_loss.backward(retain_graph=True)
    wasserstein_grad = noisy_h1.grad.detach().clone()

    noisy_h1.grad = None
    noisy_points.grad = None

    torch.manual_seed(args.slice_seed)
    sliced_loss = oin_diff.sliced_wasserstein_distance(
        noisy_h1,
        clean_h1,
        n_directions=args.n_directions,
        ignore_inf_points=True,
    )
    sliced_loss.backward()
    sliced_grad = noisy_h1.grad.detach().clone()

    fig, ax = plt.subplots(figsize=(6, 6))
    oin.plot_diagram_gradient(
        {1: noisy_h1.detach()},
        {1: wasserstein_grad},
        ax=ax,
        descent=True,
        plot_points=True,
        axis_bounds=axis_bounds,
        title="H1 descent directions on the noisy diagram",
        point_style={"s": 70, "alpha": 0.65, "color": "0.25"},
        quiver_style={
            "color": "tab:blue",
            "alpha": 0.85,
            "width": 0.006,
            "scale_units": "xy",
            "scale": 4.0,
            "angles": "xy",
            "label": "Wasserstein descent",
        },
    )
    oin.plot_diagram_gradient(
        {1: noisy_h1.detach()},
        {1: sliced_grad},
        ax=ax,
        descent=True,
        plot_points=False,
        quiver_style={
            "color": "tab:orange",
            "alpha": 0.9,
            "width": 0.004,
            "scale_units": "xy",
            "scale": 4.0,
            "angles": "xy",
            "label": "sliced Wasserstein descent",
        },
    )
    ax.scatter(
        clean_h1[:, 0].detach().cpu().numpy(),
        clean_h1[:, 1].detach().cpu().numpy(),
        s=90,
        facecolors="none",
        edgecolors="black",
        linewidths=1.4,
        label="clean H1 target",
    )
    ax.legend(frameon=False, loc="upper left")
    ax.set_aspect("equal", adjustable="datalim")
    save_or_show(fig, args.output_dir / "diagram_gradient_overlay.png", args.show)

    if args.show:
        plt.show()
    else:
        plt.close("all")

    print(f"Saved figures to {args.output_dir.resolve()}")
    print(f"Wasserstein loss: {wasserstein_loss.item():.6f}")
    print(f"Sliced Wasserstein loss: {sliced_loss.item():.6f}")
    print(f"Wasserstein gradient norm: {wasserstein_grad.norm().item():.6f}")
    print(f"Sliced Wasserstein gradient norm: {sliced_grad.norm().item():.6f}")


if __name__ == "__main__":
    main()
