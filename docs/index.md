---
sd_hide_title: true
---

# Oineus

```{raw} html
<div class="oineus-hero">
<h1 style="margin-bottom: 0.2em;">Oineus</h1>
<p style="font-size: 1.15em; opacity: 0.85;">
Parallel persistent homology and topology-aware optimization for Python and C++.
</p>
</div>
```

Oineus is a C++20 library (with first-class Python bindings) for computing
**persistent homology** on large datasets and for **differentiating through
the persistence pipeline** so you can train models that care about topology.
It supports cubical, Vietoris–Rips, and alpha filtrations; kernel / image /
cokernel persistence for simplicial maps; Wasserstein barycenters; and a
PyTorch-compatible differentiable layer.

::::{grid} 1 1 2 2
:gutter: 3
:margin: 2

:::{grid-item-card} 🚀 Get started in 10 lines
:link: getting_started/quickstart
:link-type: doc

Compute your first persistence diagram from a NumPy array. No C++ build
required — `pip install oineus`.
:::

:::{grid-item-card} 📓 Tutorial for TDA beginners
:link: tutorials/01_tda_for_beginners
:link-type: doc

Walk through sublevel sets, birth/death events, and persistence diagrams on
small 2D examples. Runs as a notebook you can edit.
:::

:::{grid-item-card} 📚 User guide
:link: user_guide
:link-type: doc

A task-oriented walkthrough: alpha and VR diagrams, function-data
diagrams, distances, manual workflow, Fréchet means, plotting,
differentiable diagrams, kernel/image/cokernel, zero persistence.
:::

:::{grid-item-card} 🔎 API reference
:link: api/index
:link-type: doc

Auto-generated function and class reference for `oineus` and `oineus.diff`.
:::
::::

## Name and pronunciation

Oineus is named after a king in Greek mythology.

Say it **OY-neh-oos**, IPA `/ˈɔɪneʊs/`, with the stress on the first syllable:

- **Oi** as in **oil** or **coin**.
- **-eus** as in Orpheus, Theseus, or Prometheus, roughly "eh-oos".

## What's inside

- **Filtrations** on cubical complexes, Vietoris–Rips complexes, alpha shapes,
  and user-specified simplicial complexes.
- **Parallel persistence reduction** with clearing optimization and optional
  V/U-matrix computation; the core algorithm is from [Morozov & Nigmetov,
  SPAA 2020](https://doi.org/10.1145/3350755.3400244).
- **Kernel, image, and cokernel persistence** for simplicial maps, including
  induced matchings.
- **Topology-aware optimization** via the critical-set method from
  [Nigmetov & Morozov, 2022](https://arxiv.org/abs/2203.16748) — solve
  singleton losses, pick conflict-resolution strategies, and push gradients
  back into scalar fields or point clouds.
- **Differentiable diagrams** in `oineus.diff` for end-to-end PyTorch
  training with topological loss terms (sliced Wasserstein, matching-based).
- **Wasserstein / bottleneck distances** and **Fréchet means** for diagram
  datasets (single, multistart, and progressive barycenters).

## How the documentation is organized

```{toctree}
:caption: Getting started
:maxdepth: 2

getting_started/install
getting_started/quickstart
getting_started/tour
```

```{toctree}
:caption: User guide
:maxdepth: 2

user_guide
```

```{toctree}
:caption: Tutorials
:maxdepth: 1

tutorials/index
```

```{toctree}
:caption: Topics
:maxdepth: 1

topics/index
```

```{toctree}
:caption: Reference
:maxdepth: 1

api/index
examples/index
```

## Citing

If you use Oineus in academic work, please cite the two papers listed under
_What's inside_. A dedicated citation page will appear here once a preferred
citation is finalized.
