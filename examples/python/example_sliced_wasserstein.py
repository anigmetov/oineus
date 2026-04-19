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


def random_diagram(n_points, birth_mean=0.5, death_mean=2.0, noise=0.3, seed=None):
    """Generate a random persistence diagram."""
    if seed is not None:
        torch.manual_seed(seed)

    births = torch.randn(n_points, dtype=torch.float64) * noise + birth_mean
    births = torch.clamp(births, min=0.0)

    persistence = torch.abs(torch.randn(n_points, dtype=torch.float64) * noise + 0.5) + 0.2
    deaths = births + persistence

    return torch.stack([births, deaths], dim=1)


def plot_diagram_with_gradients(ax, dgm, grad, title, show_diagonal=True):
    """Plot a persistence diagram with gradient vectors."""
    dgm_np = dgm.detach().numpy()
    grad_np = grad.numpy() if grad is not None else None

    # Plot diagonal
    if show_diagonal:
        max_val = max(dgm_np[:, 1].max(), dgm_np[:, 0].max()) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1, label='Diagonal')

    # Plot points
    ax.scatter(dgm_np[:, 0], dgm_np[:, 1], c='blue', s=50, alpha=0.6, zorder=3, label='Points')

    # Plot gradients as arrows
    if grad_np is not None:
        for i in range(len(dgm_np)):
            b, d = dgm_np[i]
            db, dd = grad_np[i]
            # Scale gradients for visibility
            scale = 0.1
            ax.arrow(b, d, db * scale, dd * scale,
                    head_width=0.05, head_length=0.05,
                    fc='red', ec='red', alpha=0.7, zorder=4)

        ax.scatter([], [], c='red', marker='>', s=50, alpha=0.7, label='Gradients')

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
    dgm1 = random_diagram(5, birth_mean=0.3, death_mean=1.5, noise=0.2, seed=42)
    dgm2 = random_diagram(5, birth_mean=0.7, death_mean=2.5, noise=0.2, seed=43)

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

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_diagram_with_gradients(axes[0], dgm1, dgm1.grad, 'Diagram 1 with Gradients')
    plot_diagram_with_gradients(axes[1], dgm2, dgm2.grad, 'Diagram 2 with Gradients')

    fig.suptitle(f'Sliced Wasserstein Distance = {dist.item():.4f}', fontsize=14)
    fig.tight_layout()

    return fig


def example_optimization():
    """Optimize one diagram to match another using gradient descent."""
    print("\n" + "=" * 70)
    print("Example 2: Optimization to Match Target Diagram")
    print("=" * 70)

    # Target diagram
    target = random_diagram(6, birth_mean=0.5, death_mean=2.0, noise=0.25, seed=100)

    # Initial diagram (far from target)
    dgm = random_diagram(6, birth_mean=1.5, death_mean=3.5, noise=0.3, seed=101)
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

    # Draw connections
    for i in range(min(len(target_np), len(dgm_np))):
        axes[1].plot([target_np[i, 0], dgm_np[i, 0]],
                    [target_np[i, 1], dgm_np[i, 1]],
                    'gray', alpha=0.3, linewidth=1, zorder=1)

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

    dgm1 = random_diagram(7, birth_mean=0.5, death_mean=2.0, noise=0.3, seed=200)
    dgm2 = random_diagram(5, birth_mean=0.6, death_mean=2.2, noise=0.3, seed=201)

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

    # Visualize gradient differences
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    plot_diagram_with_gradients(axes[0, 0], dgm1_std, dgm1_std.grad, 'Diagram 1 - Standard')
    plot_diagram_with_gradients(axes[0, 1], dgm2_std, dgm2_std.grad, 'Diagram 2 - Standard')
    plot_diagram_with_gradients(axes[1, 0], dgm1_corr, dgm1_corr.grad, 'Diagram 1 - Corrected')
    plot_diagram_with_gradients(axes[1, 1], dgm2_corr, dgm2_corr.grad, 'Diagram 2 - Corrected')

    fig.suptitle(f'Standard ({dist_std.item():.4f}) vs Corrected ({dist_corr.item():.4f})',
                fontsize=14)
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
