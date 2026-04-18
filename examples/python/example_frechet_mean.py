import argparse

import matplotlib.pyplot as plt
import numpy as np

import oineus as oin


def random_diagram(rng: np.random.Generator, n_points: int, birth_shift: float, persistence_shift: float) -> np.ndarray:
    births = rng.normal(loc=birth_shift, scale=0.35, size=n_points)
    persistence = np.abs(rng.normal(loc=1.2 + persistence_shift, scale=0.25, size=n_points)) + 0.05
    deaths = births + persistence
    return np.column_stack((births, deaths)).astype(np.float64)


def frechet_objective(diagrams: list[np.ndarray], barycenter: np.ndarray, *, delta: float) -> float:
    return sum(
        oin.wasserstein_distance(barycenter, diagram, q=2.0, delta=delta, internal_p=np.inf) ** 2
        for diagram in diagrams
    )


def diagram_bounds(diagrams: list[np.ndarray], padding: float = 0.5) -> dict[str, float]:
    all_points = np.vstack(diagrams)
    finite_points = all_points[np.isfinite(all_points).all(axis=1)]
    if finite_points.size == 0:
        return {"xmin": -1.0, "xmax": 1.0, "ymin": -1.0, "ymax": 1.0}

    xmin = float(np.min(finite_points[:, 0])) - padding
    xmax = float(np.max(finite_points[:, 0])) + padding
    ymin = float(np.min(finite_points[:, 1])) - padding
    ymax = float(np.max(finite_points[:, 1])) + padding
    return {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(7)
    diagrams = [
        random_diagram(rng, 6, birth_shift=-0.4, persistence_shift=-0.1),
        random_diagram(rng, 6, birth_shift=0.3, persistence_shift=0.2),
        random_diagram(rng, 6, birth_shift=0.8, persistence_shift=-0.05),
    ]
    input_colors = ["#4c78a8", "#72b7b2", "#b279a2"]
    wasserstein_delta = 1e-4

    max_iter = 1000

    bary_first = oin.frechet_mean(
        diagrams,
        init_strategy=oin.FrechetMeanInit.FirstDiagram,
        wasserstein_delta=wasserstein_delta,
        max_iter=max_iter,
    )
    bary_medoid = oin.frechet_mean(
        diagrams,
        init_strategy=oin.FrechetMeanInit.MedoidDiagram,
        wasserstein_delta=wasserstein_delta,
        max_iter=max_iter,
    )
    bary_grid = oin.frechet_mean(
        diagrams,
        init_strategy=oin.FrechetMeanInit.Grid,
        grid_n_x_bins=6,
        grid_n_y_bins=6,
        wasserstein_delta=wasserstein_delta,
        max_iter=max_iter,
    )
    bary_multistart, multistart_details = oin.frechet_mean_multistart(
        diagrams,
        starts=("medoid", "second_medoid"),
        grid_n_x_bins=6,
        grid_n_y_bins=6,
        wasserstein_delta=wasserstein_delta,
        max_iter=max_iter,
        return_details=True,
    )
    bary_progressive, progressive_details = oin.progressive_frechet_mean_multistart(
        diagrams,
        starts=("medoid", "second_medoid"),
        grid_n_x_bins=6,
        grid_n_y_bins=6,
        wasserstein_delta=wasserstein_delta,
        max_iter=max_iter,
        return_details=True,
    )

    loss_first = frechet_objective(diagrams, bary_first, delta=wasserstein_delta)
    loss_medoid = frechet_objective(diagrams, bary_medoid, delta=wasserstein_delta)
    loss_grid = frechet_objective(diagrams, bary_grid, delta=wasserstein_delta)
    loss_multistart = frechet_objective(diagrams, bary_multistart, delta=wasserstein_delta)
    loss_progressive = frechet_objective(diagrams, bary_progressive, delta=wasserstein_delta)

    barycenter_colors = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b"]
    barycenter_titles = [
        f"Barycenter: First\nloss={loss_first:.4f}",
        f"Barycenter: Medoid\nloss={loss_medoid:.4f}",
        f"Barycenter: Grid\nloss={loss_grid:.4f}",
        f"Barycenter: Multistart\nloss={loss_multistart:.4f}",
        f"Barycenter: Progressive\nloss={loss_progressive:.4f}",
    ]
    barycenters = [bary_first, bary_medoid, bary_grid, bary_multistart, bary_progressive]
    barycenter_names = ["First", "Medoid", "Grid", "Multistart", "Progressive"]
    axis_bounds = diagram_bounds(diagrams + barycenters)

    print("Pairwise W_2 distances between barycenters:")
    for i in range(len(barycenters)):
        for j in range(i + 1, len(barycenters)):
            print(
                f"  {barycenter_names[i]} vs {barycenter_names[j]}: "
                f"{oin.wasserstein_distance(barycenters[i], barycenters[j], q=2.0, delta=wasserstein_delta, internal_p=np.inf):.6f}"
            )

    print(f"Multistart selected objective: {multistart_details['objective']:.6f}")
    for run in multistart_details["runs"]:
        print(f"  start={run['start']}: objective={run['objective']:.6f}, size={len(run['barycenter'])}")

    print(f"Progressive multistart selected objective: {progressive_details['objective']:.6f}")
    print(f"Progressive thresholds: {progressive_details['thresholds']}")
    for run in progressive_details["runs"]:
        print(f"  start={run['start']}: objective={run['objective']:.6f}, size={len(run['barycenter'])}")

    fig, axes = plt.subplots(1, 6, figsize=(26, 5))
    oin.plot_persistence_diagram(
        {idx: dgm for idx, dgm in enumerate(diagrams)},
        ax=axes[0],
        title="Input Diagrams",
        color=input_colors,
        axis_bounds=axis_bounds,
        use_density=False,
    )
    handles, _ = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, [f"Diagram {idx}" for idx in range(1, len(diagrams) + 1)], frameon=False, loc="upper left")

    for ax, barycenter, color, title in zip(axes[1:], barycenters, barycenter_colors, barycenter_titles):
        oin.plot_persistence_diagram(
            barycenter,
            ax=ax,
            title=title,
            color=color,
            axis_bounds=axis_bounds,
            use_density=False,
        )
    fig.tight_layout()

    print(f"Barycenter 1st: {len(bary_first)}")
    print(f"Barycenter medoid: {len(bary_medoid)}")
    print(f"Barycenter grid: {len(bary_grid)}")
    print(f"Barycenter multistart: {len(bary_multistart)}")
    print(f"Barycenter progressive: {len(bary_progressive)}")

    if args.output:
        fig.savefig(args.output, dpi=160)
    else:
        plt.show()


if __name__ == "__main__":
    main()
