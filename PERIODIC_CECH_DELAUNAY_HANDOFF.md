# Differentiable Periodic Cech-Delaunay Handoff

Date: 2026-08-26

Status: implemented, reviewed, and ready for the Oineus release integration.

## Goal

Implement a differentiable periodic Cech-Delaunay filtration in Oineus for 2D
and 3D point clouds. Prefer this over periodic alpha for the first version.

For every simplex in CGAL's periodic Delaunay triangulation, assign the squared
radius of the minimum enclosing ball (MEB) of one coherent lift of its vertices
to the universal cover. The Delaunay combinatorics and integer periodic offsets
are detached; the MEB value is recomputed from the input tensor so gradients
flow to point coordinates.

This is the standard Delaunay-Cech rule: restrict the Cech filtration to
Delaunay simplices and use MEB radii as filtration values. Keep Oineus's current
squared-radius convention.

Periodic alpha is a later extension. It additionally needs offset-aware Gabriel
attacher/coface information, whereas periodic Cech-Delaunay needs only coherent
vertex offsets for each Delaunay simplex.

## Implementation outcome

Diode now exports coherent periodic lifts through
`fill_periodic_delaunay_lifts_arrays`. The new exporter supports 2D/3D and
exact/inexact kernels, returns contiguous `int64` vertex and offset arrays,
normalizes only the common offset gauge, and rejects duplicate vertex-ID rows
instead of silently merging them.

Oineus now accepts `periodic`, `bbox_min`, `bbox_max`, and `exact` in
`cech_delaunay_filtration`. It obtains detached periodic combinatorics and
offsets from diode, reconstructs coherent lifted Torch coordinates, and applies
the existing differentiable squared-MEB formulas. Both fat and packed
filtrations route through the backend matching float32/float64 point tensors.
Invalid or unwrapped domains are rejected before entering CGAL; this boundary
validation is intentionally absent from the minimal optimization example.

The implementation also adds:

- `tests/test_diff_periodic_cech_delaunay.py` for geometric, topological,
  differentiability, optimizer, dtype, encoding, and input-contract coverage.
- `examples/python/example_opt_periodic_cech_delaunay.py`, the required short
  SGD example with explicit in-place wrapping after every step.
- `examples/python/bench_periodic_cech_delaunay.py` for diode export, forward,
  and backward timing.
- Periodic usage and differentiability documentation in
  `docs/topics/differentiable.md`.

## Repository state and branches

The implementation uses the public repositories requested by the user.

- Oineus public remote: `git@github.com:anigmetov/oineus.git`.
- Oineus base: refreshed public `origin/master` at
  `53c343b657038a1b5db7fb3ba83e191d674340d2`.
- Oineus branch: `codex/diff-periodic-cech-delaunay`.
- Reviewed Oineus core commit:
  `039e675c4ebd7fb4519cdd5d4bdd8f38508088df`.
- Diode public upstream: `git@github.com:mrzv/diode.git`.
- Diode base: public `origin/master` at
  `2985e8ecdd735ac7a26adf9dc70009ddbb17ce8e`.
- Diode branch: `codex/periodic-delaunay-lifts`, available in the local public
  diode checkout at `/Users/anigmetov/code/diode`.
- Reviewed diode commit:
  `79a27d2f568dbc8aab80f68e6e4e412fc90d0aea`.

The documentation, example, benchmark, and this handoff are in the Oineus
commit containing this file, immediately after the core commit above. Diode
1.2.2 publishes the required periodic-lift exporter on PyPI.

## Existing implementation to reuse

- `bindings/python/oineus/diff/cech_delaunay.py`
  - `triangle_meb`
  - `tetrahedron_meb`
  - `cech_delaunay_filtration`
- `bindings/python/oineus/__init__.py`
  - `_delaunay_combinatorics`
  - array-to-filtration construction and dtype/packed routing
- `bindings/python/oineus/diff/diff_filtration.py`
- `tests/test_diff_delaunay_arrays.py`
- `tests/test_alpha_diff.py`

The current non-periodic path detaches the points for diode/CGAL, builds the
Delaunay complex, recomputes squared MEB values in Torch, installs detached
values into the underlying Oineus filtration, and retains a differentiable
sorted value tensor in `DiffFiltration`. Preserve this architecture.

The current diode dependency is `diode>=1.2.2`. Diode 1.2.1 provides
`fill_periodic_delaunay` and `fill_periodic_delaunay_arrays`, but Diode 1.2.2
also provides `fill_periodic_delaunay_lifts_arrays` with the per-vertex lattice
offsets needed to reconstruct coherent lifted simplices.

## Correct handling of periodic representatives

### What a global lattice translation means

Let an edge in the unit domain join points at `0.95` and `0.05`. One coherent
lift uses offsets `(0, 1)` and positions `(0.95, 1.05)`. Translating the entire
lift by `-1` produces offsets `(-1, 0)` and positions `(-0.05, 0.05)`. The two
offset tuples differ by the same integer on every vertex, have identical
relative offsets, and describe the same torus edge.

Such duplicate stored representatives can occur in a multiply sheeted covering.
They should not normally occur in the intended diode traversal because diode
first requires `is_triangulation_in_1_sheet()` and calls
`convert_to_1_sheeted_covering()`. CGAL documents that `STORED` and `UNIQUE`
geometric iteration agree in a one-sheeted covering.

The same face viewed through two incident cells can also use two cell-local
offset frames differing by a common translation. This does not imply two
simplices: use the face/edge iterator once and take one coherent incident-cell
frame.

### Corrected rule

Do not add a normal-path "deduplicate global translations" stage.

Instead:

1. Traverse only after successful conversion to one sheet.
2. For each emitted simplex, carry `(vertex_id, offset)` pairs together.
3. Sort by vertex ID while applying the same permutation to offsets.
4. Normalize the offset gauge by subtracting the first sorted vertex's offset
   from all offsets. This is deterministic frame selection, not deduplication.
5. Assert that each sorted vertex-ID tuple is emitted exactly once.

If the same vertex-ID tuple occurs twice with different normalized relative
offsets, do not merge the rows and do not take the minimum value. They are not
related by a global translation. They may be genuinely distinct periodic cells
or a degeneracy, and Oineus's current simplex identity cannot represent both.
Treat this as a representation blocker and diagnose it explicitly.

Independent per-edge minimum-image displacements are not a substitute for the
exported offsets. A triangle or tetrahedron needs one mutually consistent lift
of all its vertices.

## Diode change

Work from the newest diode master on a separate feature branch. The diode
master inspected during planning was
`2985e8ecdd735ac7a26adf9dc70009ddbb17ce8e`.

Add a backward-compatible API rather than changing the result of
`fill_periodic_delaunay_arrays`, for example:

```python
verts_by_dim, offsets_by_dim = diode.fill_periodic_delaunay_lifts_arrays(
    points, exact=False, bbox_min=bbox_min, bbox_max=bbox_max
)
```

Contract:

- `verts_by_dim[q]`: contiguous `int64`, shape `(n_q, q + 1)`.
- `offsets_by_dim[q]`: contiguous signed integer array, shape
  `(n_q, q + 1, ambient_dim)`.
- Rows are aligned across the two lists.
- Vertex IDs are sorted within each row; offsets follow the same permutation.
- Offsets are normalized by one common translation as described above.
- No duplicate vertex-ID rows are silently removed.

Implementation points in diode:

- `include/diode/diode.hpp`
  - 3D: obtain the periodic point/offset associated with each `Cell_handle` and
    local vertex index. For a facet or edge, use the vertices and offsets from
    the single cell supplied by its iterator representation.
  - 2D: use the point-offset data from the periodic face/edge representation,
    again retaining a coherent common frame.
- `bindings/python/diode.cpp`
  - Add flat per-dimension vertex and offset buffers and pack them as NumPy
    arrays without per-simplex Python objects.

Do not add a public C++ struct unless it is useful outside the binding. If a
public struct is added, follow the repository convention and provide a concise
`operator<<`; make its Python binding pickleable when practical.

### Diode validation

- 2D and 3D, `exact=False` and `exact=True`.
- Validate input domain shape/order before entering CGAL.
- Reconstruct every lifted coordinate as
  `point[id] + offset * (bbox_max - bbox_min)` and compare it with CGAL's
  constructed periodic point.
- Verify that normalization preserves all pairwise displacement vectors.
- Assert unique sorted vertex-ID tuples after one-sheet conversion.
- Preserve the simplex set and per-dimension counts of the existing periodic
  Delaunay exporter.
- Preserve the expected Euler characteristic of the final torus complex.
- Include boundary-crossing fixtures and random clouds large enough to satisfy
  CGAL's one-sheet requirement.
- Probe structured and near-degenerate clouds specifically for repeated ID
  tuples with different relative offsets.

## Oineus change

### Internal periodic combinatorics helper

The implemented internal helper near `_delaunay_combinatorics` consumes
diode's lift arrays and returns the Oineus filtration plus the aligned vertex
and offset arrays. It gives every row a strictly increasing temporary value,
with all rows in lower dimensions first. This preserves diode row order through
`_filtration_from_arrays` and its packed equivalent without constructing a
Python tuple or dictionary per simplex. It then checks every materialized
`get_vertices()`, `get_edges()`, `get_triangles()`, or `get_tetrahedra()` array
against the diode row order and fails explicitly if the alignment contract ever
changes. Point dtype, packed/fat selection, and `n_threads` behavior are
preserved.

### Public differentiable API

Extend the existing function using keyword-only options, approximately:

```python
cech_delaunay_filtration(
    points,
    eps=0.0,
    *,
    periodic=False,
    bbox_min=None,
    bbox_max=None,
    exact=False,
    packed=False,
    print_time=False,
)
```

For `periodic=True`:

- Require both `bbox_min` and `bbox_max`; do not infer a moving box from the
  point extrema.
- Validate a finite, nonempty domain of the correct dimension.
- Require points to be wrapped into the documented half-open fundamental
  domain.
- Treat the box and integer offsets as fixed, detached geometry in v1.
- Construct lifted Torch coordinates as
  `points[ids] + offsets * (bbox_max - bbox_min)` using the input dtype/device.
- Feed coherent lifted coordinates to the existing edge, triangle, and
  tetrahedron MEB formulas.
- Keep vertices at zero and keep squared-radius filtration values.
- Install a detached real buffer into the underlying filtration and retain the
  differentiable sorted values in `DiffFiltration` exactly as the current path
  does.
- Preserve `FiltrationKind.CechDelaunay`.

Expose `exact` to the detached CGAL construction. Keep the default non-periodic
behavior unchanged.

### Differentiability contract

The result is piecewise differentiable. Delaunay flips, changes of periodic
offset representative, MEB-support changes, and exact ties are nondifferentiable
boundaries. This is consistent with the existing detached-combinatorics
differentiable-filtration model.

## Oineus validation plan

The original validation checklist is retained below. Completed checks and
remaining extensions are separated immediately after it so this plan is not
mistaken for a claim that every proposed stress test was implemented.

1. Non-periodic regression
   - Existing values, diagrams, and gradients remain unchanged when
     `periodic=False`.
2. Boundary fixtures
   - Edges crossing every box boundary use the short coherent lift.
   - Triangles/tetrahedra spanning multiple faces of the box use a single
     consistent lift.
3. Value oracle
   - Compare each value with an independent MEB calculation on the lifted
     coordinates exported by diode.
4. Gradient oracle
   - Compare Torch autograd with central finite differences on random clouds.
   - Rebuild diode combinatorics for each perturbation and assert simplex keys,
     offsets, and MEB support remain unchanged before comparing derivatives.
5. Periodic invariance
   - Translate and rewrap the entire cloud; diagrams and value multisets should
     agree.
6. Topology
   - The final 2-torus complex should have Betti numbers `(1, 2, 1)`.
   - The final 3-torus complex should have Betti numbers `(1, 3, 3, 1)` when
     all needed dimensions are retained.
7. Replicated-cloud oracle
   - Compare lifted simplices/values against the appropriate central part of a
     `3^d` Euclidean replication on small fixtures.
8. Backend/encoding matrix
   - 2D/3D, float32/float64, packed/fat, `exact=False/True` where practical.
9. Performance
   - Benchmark diode export, full forward, and backward on representative
     512-point clouds and at least one larger scientific input.
   - Verify offset handling remains linear in the number of Delaunay simplices.

### Validation completed

- Diode focused lift tests: 16 passed.
- Diode periodic test selection: 23 passed, 38 deselected.
- Full diode suite: 234 passed, 21 skipped, 3 xfailed.
- Oineus focused periodic suite: 32 passed.
- Oineus periodic plus related alpha/Delaunay suites: 72 passed.
- Full Oineus Python suite from `tests/`: 1204 passed and 10 skipped, with one
  pre-existing order-dependent JAX configuration failure. The failing file
  passes 5/5 in isolation with JAX x64 disabled, which is its documented
  premise.
- The optimization example exits successfully and its selected squared H0
  death decreases from `1e-4` to approximately `3.91e-7` in five steps. Its
  focused regression also verifies that the first raw step crosses both
  opposite box boundaries before wrapping.
- The central finite-difference regression verifies that the selected periodic
  edge's vertex IDs and coherent offsets remain unchanged under both
  perturbations.
- Translation and rewrapping are checked for both filtration values and finite
  persistence diagrams.
- The warmed, repeated-median benchmark runs successfully at 512 and 2048
  points in 2D and at 512, 1024, and 2048 points in 3D. It reports labeled
  seconds and microseconds per simplex; the 2048-point 3D run exports 59,688
  simplices.
- The mandatory review found finite-endpoint box widths that overflowed in
  float32; the implementation now rejects a non-finite width before lift
  arithmetic and includes a regression. Re-review found no remaining issues.

### Remaining validation extensions

- Add explicit deterministic fixtures crossing every axis-aligned box face;
  the current boundary loss fixture crosses one pair of opposite 2D faces,
  while the all-simplex MEB and replicated-cloud oracles cover 2D/3D lifts more
  broadly.
- Extend the finite-difference test to a random higher-dimensional simplex and
  record its MEB support set as well as its simplex IDs and offsets.
- Run the benchmark on a representative large scientific periodic cloud and
  isolate offset marshaling from triangulation to assess its empirical scaling.
  The current repeated benchmark is an end-to-end timing tool, not a proof that
  the offset component is linear.

## Required periodic optimization example

Add a short executable example, preferably
`examples/python/example_opt_periodic_cech_delaunay.py`. This example is a
required end-to-end acceptance artifact, not optional documentation polish.

The example must demonstrate the specifically periodic gradient, not merely run
an optimizer on points that happen to be wrapped:

- Use a deterministic 2D point cloud in a square periodic box.
- Put two distinguished vertices close on the torus but near opposite vertical
  sides of the square, for example at x-coordinates `0.01` and `0.99` with the
  same y-coordinate.
- Include enough fixed deterministic background points for CGAL to construct a
  valid nondegenerate one-sheet periodic Delaunay triangulation.
- Arrange the cloud so the boundary-crossing edge is the unique shortest
  periodic edge and therefore supplies the selected finite H0 death.
- Build `cech_delaunay_filtration(..., periodic=True, bbox_min=...,
  bbox_max=...)`, compute its H0 persistence diagram, and minimize the death
  value corresponding to that shortest edge.
- Choose the initial separation and SGD learning rate so the first raw
  `opt.step()` moves both distinguished points outside the fundamental domain:
  the left point through the left boundary and the right point through the
  right boundary. This proves the gradient follows the short torus lift rather
  than the long Euclidean segment across the square.
- Immediately after every optimizer step, wrap all points back into the
  half-open bounding box.

Keep the script as close as possible in structure and length to the shortest
existing topological-optimization examples. It must contain no plotting, CLI, optional modes,
availability guards, try/except fallback, or other protective/general-purpose
code. A small `get_pts()` and a small `wrap()` are acceptable if they make the
core loop clearer.

The core loop should look essentially like this:

```python
pts = get_pts()
opt = torch.optim.SGD([pts], lr=lr)

for step_idx in range(n_steps):
    opt.zero_grad()
    fil = od.cech_delaunay_filtration(
        pts, periodic=True, bbox_min=bbox_min, bbox_max=bbox_max
    )
    dgm0 = od.persistence_diagram(fil)[0]
    loss = dgm0[:, 1].min()
    loss.backward()
    opt.step()
    with torch.no_grad():
        pts.copy_(wrap(pts, bbox_min, bbox_max))
```

Do not write `pts = wrap(pts, ...)` after constructing the optimizer. Rebinding
would leave SGD holding the old leaf tensor. The in-place `copy_` under
`torch.no_grad()` is required so the optimizer and subsequent filtrations keep
using the same differentiable parameter.

Add a focused automated test alongside the example. The test, rather than the
minimal example script, should verify that:

- The distinguished points leave opposite sides of the box immediately after
  the first raw optimizer step.
- The explicit wrap returns every coordinate to the half-open box.
- The optimizer still references the same `pts` tensor.
- The periodic distance between the distinguished vertices decreases.
- The selected H0 death/loss decreases over the short run.

## V1 scope

Include:

- Unweighted periodic Cech-Delaunay.
- 2D and 3D.
- Torch, matching the current Cech-Delaunay path.
- Fixed axis-aligned periodic domains accepted by diode/CGAL.

Exclude initially:

- Periodic alpha attachment reconstruction.
- Weighted/regular periodic triangulations.
- JAX support for Cech-Delaunay.
- Learnable box geometry.
- Triclinic cells.

Document the exact CGAL domain restriction rather than promising arbitrary
orthorhombic boxes without verification. Also distinguish the implemented
periodic Delaunay-Cech filtration from claims about a full torus Cech nerve at
radii where periodic balls cease to form the required good cover.

## Dependency and commit sequence

The completed sequence is:

1. Diode commit `79a27d2`: lift exporter, binding, tests, and API documentation.
2. Oineus commit `039e675`: periodic combinatorics helper, differentiable value
   path, and focused tests.
3. The following Oineus commit: user documentation, example, benchmark, and
   this handoff.

Integration was validated against PyPI Diode 1.2.2, which includes the periodic
lift exporter. Oineus declares `diode>=1.2.2`; the capability gate still raises
a clear error naming `fill_periodic_delaunay_lifts_arrays` if an incompatible
Diode installation is forced into the environment.

Before every commit, ask the `@review` subagent to inspect the focused diff. If
the review finds an issue, fix it and request another review before committing.
Inspect staged changes so unrelated work is never included. Do not push without
explicit authorization.

## Primary references

- Bauer and Edelsbrunner, *The Morse Theory of Cech and Delaunay Complexes*:
  https://arxiv.org/abs/1312.1231
- CGAL periodic 3D triangulation reference, including one-sheet and iterator
  semantics:
  https://doc.cgal.org/latest/Periodic_3_triangulation_3/classCGAL_1_1Periodic__3__triangulation__3.html
- CGAL periodic 2D triangulation manual:
  https://doc.cgal.org/latest/Periodic_2_triangulation_2/index.html
- Diode source inspected during planning:
  https://github.com/mrzv/diode/tree/2985e8ecdd735ac7a26adf9dc70009ddbb17ce8e

## Main correctness gate result

The tested one-sheet periodic triangulations emitted at most one simplex for
each sorted vertex-ID tuple. The exporter still checks this at runtime: a
repeated row with identical offsets and a repeated row with different
normalized offsets both raise. Nothing is concealed by deduplication, minimum
values, or independently chosen minimum-image edges.
