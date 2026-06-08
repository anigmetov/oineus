# Low-level critical-set interface

```{note}
For the **standard** topology-optimization workflow, see
{doc}`differentiable`. The PyTorch-driven version is simpler, more
flexible, and is what most users want.
```

This page documents {py:class}`oineus.TopologyOptimizer`, the non-PyTorch
interface to the critical-set method ([Nigmetov & Morozov, 2022](https://arxiv.org/abs/2203.16748)).
Reach for it when you need explicit critical-set output to drop into a
custom solver, or when bringing in `torch` as a dependency is not an
option.

The method itself turns "make this diagram point shorter" or "make this
feature exist" into a concrete list of which cells to move and in which
direction. The standard workflow is:

```{code-block} python
import oineus as oin

fil = oin.vr_filtration(points, max_dim=2, max_diameter=2.0)
opt = oin.TopologyOptimizer(fil)

# What does the diagram look like now?
dgms = opt.compute_diagram(include_inf_points=False)

# Move every H1 point with persistence < eps to the diagonal (denoise).
eps = opt.get_nth_persistence(dim=1, n=2)         # eps = 2nd-longest H1 bar
indices, values = opt.simplify(eps, oin.DenoiseStrategy.BirthBirth, dim=1)

# Walk the critical set of each pair...
critical_sets = opt.singletons(indices, values)

# ...and resolve conflicts when multiple critical sets touch the same cell.
crit_indices, crit_values = opt.combine_loss(
    critical_sets, oin.ConflictStrategy.Max
)
```

`crit_indices` is the list of cell indices in the filtration whose values
need to change; `crit_values` is the target value for each. From there,
the typical move is to update the filtration in place and re-reduce.
(If you want PyTorch to consume the same `crit_indices` /
`crit_values`, use the differentiable variant -- see
{doc}`differentiable`.)

## Denoise strategies

{py:class}`oineus.DenoiseStrategy` selects how a "low-persistence" pair is
canceled:

- `BirthBirth` -- move the death simplex back to the birth value
  (canceling at the birth).
- `DeathDeath` -- move the birth simplex forward to the death value
  (canceling at the death).
- `Midway` -- meet in the middle.

The choice depends on whether the underlying data lets you move only the
death cell, only the birth cell, or both. For grid filtrations
(`compute_diagrams_ls` and friends), all three are usable; for VR /
alpha, moving the birth cell requires changing point positions, which
might be more expensive than moving the death cell.

## Conflict strategies

A single pair has one critical set, but a typical "denoise H1 below eps"
loss produces several critical sets that often overlap. `combine_loss`
resolves the overlaps according to {py:class}`oineus.ConflictStrategy`:

- `Max` -- on a contested cell, take the largest target value.
- `Avg` -- average the contested target values.
- `Sum` -- sum the contested moves (the differentiable analogue prefers
  this when the loss is squared distance).
- `FixCritAvg` -- average but pin the critical cells of each set.

`Max` is usually the safe default for denoising; `Avg` and `Sum` are more
useful inside an autograd loop where you want a single, well-behaved
gradient signal.

## Other primitives

`TopologyOptimizer` also exposes a handful of single-pair operations,
useful when you want to do something custom rather than "denoise":

- `increase_death(idx)`, `decrease_death(idx)` -- critical set for
  pushing the death of a pair to $\pm \infty$.
- `increase_birth(idx)`, `decrease_birth(idx)` -- the same on the birth
  side.
- `singleton(idx, target_value)` -- critical set for moving a single
  pair to a single target value.

## Performance and determinism

`TopologyOptimizer(fil, n_threads=N)` caches the boundary matrix once and, on
the first `ensure_*_reduced` / `compute_diagram` / `singletons` call, reduces it
through the *fused, keep-working* path: it builds the working columns straight
from the cached boundary and keeps the reduced $R/V$ in working form (no
copy-back), reading them through per-column accessors and computing $U$ on
demand. Set `n_threads` to the cores you can spare; see {doc}`performance`.

Because the optimizer restores the canonical ELZ form of $V$ in the optimized
dimensions, and that form is unique, the critical sets it returns are
**deterministic across thread counts** -- `n_threads` is purely a speed knob.

## See also

- {doc}`differentiable` -- the canonical PyTorch-driven path.
- {doc}`decomposition` -- under the hood, `TopologyOptimizer` holds a
  cached boundary matrix and builds decompositions lazily.
- {doc}`performance` -- the fused reduction and its per-phase timings.
- `examples/python/example_opt_vr.py` -- runnable demo of the non-diff
  path.
