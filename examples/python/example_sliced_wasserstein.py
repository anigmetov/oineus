#!/usr/bin/env python3
"""
Example demonstrating sliced Wasserstein distance on differentiable diagrams.

This script:
1. Creates random persistence diagrams as PyTorch tensors
2. Computes sliced Wasserstein distance
3. Visualizes gradients flowing back to diagram points
4. Shows optimization to align one diagram to another

Run with: python example_sliced_wasserstein.py
          python example_sliced_wasserstein.py --output <dir>  # save figures
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import oineus.diff as oin_diff


def random_diagram(n_points, birth_mean=0.5, persistence_mean=1.0, noise=0.3, seed=None):
    """Generate a random persistence diagram."""
    if seed is not None:
        torch.manual_seed(seed)

    births = torch.randn(n_points, dtype=torch.float64) * noise + birth_mean
    births = torch.clamp(births, min=0.0)

    persistence = torch.abs(torch.randn(n_points, dtype=torch.float64) * noise + persistence_mean) + 0.2
    deaths = births + persistence

    return torch.stack([births, deaths], dim=1)


def plot_gradient_arrows(ax, dgm_np, grad_np, color, scale=0.1, show_descent=True):
    """Plot gradient arrows, optionally flipped to show the descent/update direction."""
    if grad_np is None:
        return

    direction = -grad_np if show_descent else grad_np
    for i in range(len(dgm_np)):
        b, d = dgm_np[i]
        db, dd = direction[i]
        ax.arrow(b, d, db * scale, dd * scale,
                head_width=0.05, head_length=0.05,
                fc=color, ec=color, alpha=0.6, zorder=4, linewidth=1.5)


def plot_diagram_with_gradients(ax, dgm, grad, title, color='blue', label='Points', show_diagonal=True):
    """Plot a persistence diagram with descent-direction arrows."""
    dgm_np = dgm.detach().numpy()
    grad_np = grad.numpy() if grad is not None else None

    # Plot diagonal
    if show_diagonal:
        max_val = max(dgm_np[:, 1].max(), dgm_np[:, 0].max()) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1, label='Diagonal')

    # Plot points with specified color
    ax.scatter(dgm_np[:, 0], dgm_np[:, 1], c=color, s=40, alpha=0.7, zorder=3, label=label)

    # Plot descent direction so the arrows match the optimizer update.
    grad_color = color if color != 'blue' else 'darkblue'
    plot_gradient_arrows(ax, dgm_np, grad_np, grad_color, scale=0.1, show_descent=True)

    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(frameon=False, loc='upper left')
    ax.grid(True, alpha=0.3)


def plot_combined_diagrams_with_gradients(ax, dgm1, grad1, dgm2, grad2, title):
    """Plot both diagrams together with descent-direction arrows."""
    dgm1_np = dgm1.detach().numpy()
    dgm2_np = dgm2.detach().numpy()
    grad1_np = grad1.numpy() if grad1 is not None else None
    grad2_np = grad2.numpy() if grad2 is not None else None

    all_points = np.vstack([dgm1_np, dgm2_np])
    max_val = all_points.max() * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1, label='Diagonal')

    ax.scatter(dgm1_np[:, 0], dgm1_np[:, 1], c='#1f77b4', s=40, alpha=0.7, zorder=3, label='Diagram 1')
    ax.scatter(dgm2_np[:, 0], dgm2_np[:, 1], c='#ff7f0e', s=40, alpha=0.7, zorder=3, label='Diagram 2')

    plot_gradient_arrows(ax, dgm1_np, grad1_np, '#0d3d63', scale=0.1, show_descent=True)
    plot_gradient_arrows(ax, dgm2_np, grad2_np, '#cc6600', scale=0.1, show_descent=True)

    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(frameon=False, loc='upper left')
    ax.grid(True, alpha=0.3)


def example_gradient_visualization():
    """Visualize gradients from sliced Wasserstein distance."""
    print("=" * 70)
    print("Example 1: Gradient Visualization")
    print("=" * 70)

    # Create two diagrams
    dgm1 = random_diagram(5, birth_mean=0.3, persistence_mean=1.2, noise=0.2, seed=42)
    dgm2 = random_diagram(5, birth_mean=0.7, persistence_mean=1.8, noise=0.2, seed=43)

    dgm1.requires_grad_(True)
    dgm2.requires_grad_(True)

    # Compute distance
    torch.manual_seed(42)
    dist = oin_diff.sliced_wasserstein_distance(dgm1, dgm2, n_directions=100)

    print(f"Sliced Wasserstein distance: {dist.item():.6f}")

    # Backpropagate
    dist.backward()

    print(f"Gradient norm on dgm1: {dgm1.grad.norm().item():.6f}")
    print(f"Gradient norm on dgm2: {dgm2.grad.norm().item():.6f}")

    # Visualize both diagrams in one plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Compute combined extent for diagonal
    all_points = torch.cat([dgm1, dgm2], dim=0).detach().numpy()
    max_val = all_points.max() * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1, label='Diagonal')

    # Plot first diagram in blue
    dgm1_np = dgm1.detach().numpy()
    grad1_np = dgm1.grad.numpy()
    ax.scatter(dgm1_np[:, 0], dgm1_np[:, 1], c='#1f77b4', s=40, alpha=0.7, zorder=3, label='Diagram 1')
    plot_gradient_arrows(ax, dgm1_np, grad1_np, '#0d3d63', scale=0.1, show_descent=True)

    # Plot second diagram in orange
    dgm2_np = dgm2.detach().numpy()
    grad2_np = dgm2.grad.numpy()
    ax.scatter(dgm2_np[:, 0], dgm2_np[:, 1], c='#ff7f0e', s=40, alpha=0.7, zorder=3, label='Diagram 2')
    plot_gradient_arrows(ax, dgm2_np, grad2_np, '#cc6600', scale=0.1, show_descent=True)

    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_title(f'Gradient Visualization (descent direction, distance = {dist.item():.4f})', fontsize=13)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(frameon=False, loc='upper left')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    return fig


def example_optimization():
    """Optimize one diagram to match another using gradient descent."""
    print("\n" + "=" * 70)
    print("Example 2: Optimization to Match Target Diagram")
    print("=" * 70)

    # Target diagram
    target = random_diagram(6, birth_mean=0.5, persistence_mean=1.5, noise=0.25, seed=100)

    # Initial diagram (far from target in both birth and persistence)
    dgm = random_diagram(6, birth_mean=1.5, persistence_mean=2.0, noise=0.3, seed=101)
    dgm.requires_grad_(True)

    optimizer = torch.optim.Adam([dgm], lr=0.05)

    n_steps = 50
    distances = []

    print(f"Running {n_steps} optimization steps...")

    for step in range(n_steps):
        optimizer.zero_grad()

        torch.manual_seed(42)
        dist = oin_diff.sliced_wasserstein_distance(dgm, target, n_directions=100)
        distances.append(dist.item())

        dist.backward()
        optimizer.step()

        if step % 10 == 0 or step == n_steps - 1:
            print(f"  Step {step:3d}: distance = {dist.item():.6f}")

    print(f"Initial distance: {distances[0]:.6f}")
    print(f"Final distance: {distances[-1]:.6f}")
    print(f"Reduction: {(1 - distances[-1]/distances[0]) * 100:.1f}%")

    # Visualize optimization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot convergence
    axes[0].plot(distances, linewidth=2)
    axes[0].set_xlabel('Optimization Step')
    axes[0].set_ylabel('Sliced Wasserstein Distance')
    axes[0].set_title('Convergence')
    axes[0].grid(True, alpha=0.3)

    # Plot final diagrams
    target_np = target.detach().numpy()
    dgm_np = dgm.detach().numpy()

    max_val = max(target_np.max(), dgm_np.max()) * 1.1
    axes[1].plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1, label='Diagonal')
    axes[1].scatter(target_np[:, 0], target_np[:, 1], c='green', s=100, alpha=0.6,
                   marker='o', label='Target', zorder=3)
    axes[1].scatter(dgm_np[:, 0], dgm_np[:, 1], c='blue', s=100, alpha=0.6,
                   marker='x', label='Optimized', zorder=3)

    axes[1].set_xlabel('Birth')
    axes[1].set_ylabel('Death')
    axes[1].set_title(f'Final Result (distance = {distances[-1]:.4f})')
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].legend(frameon=False, loc='upper left')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    return fig


def example_compare_variants():
    """Compare standard and diagonal-corrected sliced Wasserstein."""
    print("\n" + "=" * 70)
    print("Example 3: Standard vs Diagonal-Corrected")
    print("=" * 70)

    dgm1 = random_diagram(7, birth_mean=0.5, persistence_mean=1.5, noise=0.3, seed=200)
    dgm2 = random_diagram(5, birth_mean=0.6, persistence_mean=1.6, noise=0.3, seed=201)

    dgm1_std = dgm1.clone().requires_grad_(True)
    dgm2_std = dgm2.clone().requires_grad_(True)

    dgm1_corr = dgm1.clone().requires_grad_(True)
    dgm2_corr = dgm2.clone().requires_grad_(True)

    # Standard version
    torch.manual_seed(42)
    dist_std = oin_diff.sliced_wasserstein_distance(dgm1_std, dgm2_std, n_directions=100)
    dist_std.backward()

    # Corrected version
    torch.manual_seed(42)
    dist_corr = oin_diff.sliced_wasserstein_distance_diag_corrected(dgm1_corr, dgm2_corr, n_directions=100)
    dist_corr.backward()

    print(f"Standard distance: {dist_std.item():.6f}")
    print(f"Diagonal-corrected distance: {dist_corr.item():.6f}")
    print(f"Difference: {abs(dist_std.item() - dist_corr.item()):.6f}")

    print(f"\nGradient norms on dgm1:")
    print(f"  Standard: {dgm1_std.grad.norm().item():.6f}")
    print(f"  Corrected: {dgm1_corr.grad.norm().item():.6f}")

    # Visualize both diagrams together for each variant.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_combined_diagrams_with_gradients(
        axes[0], dgm1_std, dgm1_std.grad, dgm2_std, dgm2_std.grad,
        f'Standard sliced Wasserstein\n(distance = {dist_std.item():.4f})'
    )
    plot_combined_diagrams_with_gradients(
        axes[1], dgm1_corr, dgm1_corr.grad, dgm2_corr, dgm2_corr.grad,
        f'Diagonal-corrected sliced Wasserstein\n(distance = {dist_corr.item():.4f})'
    )

    fig.suptitle('Arrows show descent direction (the optimizer update)', fontsize=14)
    fig.tight_layout()

    return fig


def main():
    parser = argparse.ArgumentParser(description='Sliced Wasserstein distance examples with visualization')
    parser.add_argument('--output', type=str, default=None, help='Save figures to this directory')
    parser.add_argument('--no-show', action='store_true', help='Do not display interactive plots')
    args = parser.parse_args()

    # Run examples
    fig1 = example_gradient_visualization()
    fig2 = example_optimization()
    fig3 = example_compare_variants()

    # Save or show
    if args.output:
        import os
        os.makedirs(args.output, exist_ok=True)
        fig1.savefig(f"{args.output}/gradients.png", dpi=150, bbox_inches='tight')
        fig2.savefig(f"{args.output}/optimization.png", dpi=150, bbox_inches='tight')
        fig3.savefig(f"{args.output}/comparison.png", dpi=150, bbox_inches='tight')
        print(f"\nFigures saved to {args.output}/")

    if not args.no_show:
        print("\nDisplaying plots (close windows to exit)...")
        plt.show()
    elif not args.output:
        print("\nUse --output <dir> to save figures or remove --no-show to display them.")

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
