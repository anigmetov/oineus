Oineus
======

Oineus is a parallel persistent homology and differentiable topology library for
Python (backed by C++20 and nanobind):

* Shared-memory parallel reduction of boundary and coboundary matrices (`R = D V` and `U = V^{-1}`).
* Filtrations on regular 1D--4D grids (Freudenthal and cubical, including periodic domains), point clouds (Vietoris--Rips, Alpha, Weak Alpha, Cech, Cech--Delaunay, Function--Delaunay), mapping cylinders, and custom cell complexes.
* Bottleneck, Wasserstein, and Sliced Wasserstein distances, optimal matchings, and Wasserstein Frechet means (barycenters).
* Relative persistence, kernel / image / cokernel (KICR) persistence, and two-population mixup barcodes.
* Differentiable persistence diagrams, KICR, mixup, critical-set topological optimization, and Weighill--Zhou birth/death cochain losses (`oineus.diff`) for PyTorch and JAX.
* Publication-ready Matplotlib visualization (`oineus.vis`) for diagrams, matchings, gradients, and representative cycles/chains.

Documentation
-------------

Full tutorials and API reference are hosted at https://oineus.readthedocs.io.
