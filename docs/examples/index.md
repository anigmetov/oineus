# Examples

The repository ships a collection of runnable scripts under
[`examples/python/`](https://github.com/anigmetov/oineus/tree/master/examples/python).
Each corresponds to a pytest test that validates its output, so they stay
in sync with the library.

| Script | What it shows |
| --- | --- |
| `example.py` | First steps: Freudenthal filtration, reduction, diagram. |
| `example_vr.py` | Vietoris–Rips on a point cloud. |
| `example_cube.py` | Cubical complex on a 3D array. |
| `example_manual.py` | A hand-built simplicial complex via {py:class}`oineus.Simplex`. |
| `example_opt.py` | Topology-aware optimization of a scalar field. |
| `example_opt_vr.py` | Same, for point clouds (VR filtration). |
| `example_kernel.py` | Kernel / image / cokernel persistence via KICR. |
| `example_kernel_cyl.py` | Kernel / cokernel via mapping cylinder. |
| `example_cone.py` | Cone construction — the simplest relative example. |
| `example_cone_diff.py` | Differentiable version of the cone. |
| `example_diff_cyl.py` | Differentiable mapping cylinder. |
| `example_diff_vr_pts.py` | Backprop through VR on a point cloud. |
| `example_diff_vr_dists.py` | Backprop through VR on a distance matrix. |
| `example_frechet_mean.py` | Single / multistart / progressive barycenter. |
| `example_sliced_wasserstein.py` | Sliced Wasserstein as a differentiable loss. |
| `example_wasserstein_opt.py` | Persistence-aware Wasserstein matching. |
| `visualize_critical_set_vr.py` | Visualize critical cells on a VR filtration. |
| `visualize_critical_set_alpha.py` | Visualize critical cells on an alpha filtration. |

Run any of them after installing `oineus` and (for the differentiable
scripts) `torch`:

```bash
python examples/python/example_vr.py
```
