---
sd_hide_title: true
---

# Oineus

```{raw} html
<div class="oineus-hero">
<h1 style="margin-bottom: 0.2em;">Oineus</h1>
<p style="font-size: 1.15em; opacity: 0.85;">
Parallel persistent homology, diagram geometry, and differentiable topology for Python.
</p>
</div>
```

Oineus is a Python library for **persistent homology**, **diagram distances & barycenters**, **map-induced & two-population persistence**, and **differentiating through topological pipelines** in **PyTorch** and **JAX**—backed by a shared-memory parallel C++20 engine with full PEP 561 `.pyi` type annotations.

::::{grid} 1 1 2 2
:gutter: 3
:margin: 2

:::{grid-item-card} Quickstart in 5 Minutes
:link: getting_started/quickstart
:link-type: doc

Compute persistence diagrams on scalar fields and point clouds, compare them with Wasserstein matchings, and plot them in a few lines of Python.
:::

:::{grid-item-card} Core Concepts & Decision Guide
:link: getting_started/concepts
:link-type: doc

Understand Oineus's dataflow (`Data -> Filtration -> Decomposition -> Diagrams`) and pick the right filtration, distance, and optimization strategy for your problem.
:::

:::{grid-item-card} Executable Tutorials (16 Guides)
:link: tutorials/index
:link-type: doc

Six progressive tracks with runnable code and inline figures covering grids, point clouds, cycles, barycenters, KICR, mixup, `oineus.diff`, cochains, and vineyards.
:::

:::{grid-item-card} Complete Python API Reference
:link: api/index
:link-type: doc

Every class, method, property, enum, function, and style constant across `oineus`, `oineus.diff`, `oineus.diff.cochains`, and `oineus.vis`.
:::
::::

## Highlights

- **Rich Filtrations**: 1D–4D **Freudenthal** and **cubical** grids (`values_on="vertices"` or `"cells"`, periodic `wrap=True`), **Vietoris–Rips** (in-order generation), **Alpha** (weighted, periodic, exact via `diode`), **Weak Alpha**, **Čech**, **Čech–Delaunay**, **Function–Delaunay** bifiltrations, mapping cylinders, and custom simplicial/cubical/product complexes.
- **Fast Parallel Reduction & Dynamic Updates**: Lockfree matrix reduction ($R = D V$ and selected-inversion $U = V^{-1}$) with clearing, apparent pairs, canonical ELZ restoration, slim & bit-packed encodings, `float32`/`float64` precision, dynamic vineyards, Luo–Nelson warm starts, and SiRUP cell removal.
- **Distances, Matchings & Barycenters**: Exact and approximate **Bottleneck** and **$q$-Wasserstein** distances & explicit matchings (with `InfKind` essential pairings), **Sliced Wasserstein** (standard and diagram-corrected), and multistart / progressive **Wasserstein Fréchet means**.
- **Maps & Multi-Population Topology**: Relative homology $H_*(K, L)$, **Kernel / Image / Cokernel (KICR)** persistence, Bauer–Lesnick induced matchings, and two-population **Mixup Barcodes**.
- **Differentiable TDA (`oineus.diff`)**: Reverse-mode autodiff in **PyTorch** and **JAX** for filtrations, persistence diagrams, KICR, mixup, Wasserstein losses, critical-set big-step optimization (`TopologyOptimizer`), and Weighill–Zhou **birth/death cochain content losses** (`oineus.diff.cochains`).
- **Publication-Ready Visualization (`oineus.vis`)**: Matplotlib helpers (`plot_diagram`, `plot_matching`, `plot_diagram_gradient`, `plot_chain`) with Okabe–Ito palettes and hybrid scatter-density plots.

## Name & Pronunciation

Oineus is named after a king of Calydon in Greek mythology. Say it **OY-neh-oos**, IPA `/ˈɔɪneʊs/`, with stress on the first syllable:

- **Oi** as in **oil** or **coin**.
- **-eus** as in Orpheus, Theseus, or Prometheus, roughly "eh-oos".

## Citing Oineus

If you use Oineus in academic research, please cite the relevant papers:

- **Parallel persistent homology**: D. Morozov and A. Nigmetov, *"Towards lockfree persistent homology,"* Proceedings of the 32nd ACM Symposium on Parallelism in Algorithms and Architectures (SPAA), pp. 555–557, 2020 ([doi:10.1145/3350755.3400244](https://doi.org/10.1145/3350755.3400244)).
- **Critical-set topological optimization**: A. Nigmetov and D. Morozov, *"Topological Optimization with Big Steps,"* [arXiv:2203.16748](https://arxiv.org/abs/2203.16748), 2022.

```{toctree}
:caption: Getting Started
:maxdepth: 2
:hidden:

getting_started/install
getting_started/quickstart
getting_started/concepts
```

```{toctree}
:caption: Tutorials
:maxdepth: 2
:hidden:

tutorials/index
```

```{toctree}
:caption: Reference
:maxdepth: 2
:hidden:

api/index
examples/index
```
