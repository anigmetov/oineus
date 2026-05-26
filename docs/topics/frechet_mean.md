# Fréchet mean of persistence diagrams

The Wasserstein barycenter of $N$ diagrams $D_1, \dots, D_N$ is the diagram
$D^*$ that minimizes the weighted Fréchet functional

$$ F(D) \;=\; \sum_{i=1}^N w_i \, W_q(D, D_i)^q. $$

The functional is highly non-convex (the matching from $D$ to each $D_i$
is itself an optimization problem), so the answer you get depends on the
initialization. Oineus exposes a single-run optimizer, a multi-start
wrapper, and a "progressive" version that walks down a persistence-
threshold schedule. For most inputs the progressive multistart is the
right default.

## Single run

```{code-block} python
import numpy as np
import oineus as oin

# diagrams is a list of (n_i, 2) numpy arrays, one per input
bary = oin.frechet_mean(
    diagrams,
    init_strategy=oin.FrechetMeanInit.MedoidDiagram,
    wasserstein_delta=1e-4,
    max_iter=200,
)
print(bary.shape)
```

Returns the barycenter as an `(m, 2)` NumPy array. Common knobs:

- `init_strategy` -- {py:class}`oineus.FrechetMeanInit` enum:
  `FirstDiagram`, `MedoidDiagram`, `Grid`, `Random`, `Diagonal`,
  `Custom`. `MedoidDiagram` is a good single-shot default; `Grid` seeds
  the optimization from a uniform grid above the diagonal; `Custom`
  requires you to pass `custom_initial_barycenter`.
- `weights` -- 1D array, one entry per input diagram. Defaults to uniform.
- `wasserstein_delta` -- relative accuracy of each Hera call. Tighten
  to ~1e-4 for "publication-quality" barycenters; the default `0.01` is
  fine for plotting and exploration.
- `max_iter`, `tol` -- iteration cap and convergence tolerance on the
  objective.
- `ignore_infinite_points` -- if `True`, essential pairs are stripped
  before optimization. Use this whenever the input diagrams have
  inconsistent essential counts.

## Multistart

The single-run optimizer can get stuck in shallow local minima. The
multistart wrapper runs `frechet_mean` from a small set of seeds and
returns the lowest-objective result:

```{code-block} python
bary, details = oin.frechet_mean_multistart(
    diagrams,
    starts=("medoid", "second_medoid", "farthest_from_medoid"),
    wasserstein_delta=1e-4,
    max_iter=200,
    return_details=True,
)
print(details["objective"])
for run in details["runs"]:
    print(run["start"], run["objective"], run["barycenter"].shape)
```

The `starts` tuple lists named seeds (a small library of medoid-based
choices) or `dict`s containing explicit `init_strategy` arguments. Pass
`return_details=True` to inspect the per-run objective and barycenters.

## Progressive

Heavy-tailed input diagrams (a few long bars plus many near-diagonal
ones) often produce bad local minima for the one-shot optimizer: the
near-diagonal mass dominates the gradient and the optimizer never finds
the high-persistence features. The "progressive" variant solves a
sequence of sub-problems on diagrams filtered by a decreasing persistence
threshold, warm-starting each stage from the previous solution:

```{code-block} python
bary, details = oin.progressive_frechet_mean_multistart(
    diagrams,
    starts=("medoid", "second_medoid"),
    wasserstein_delta=1e-4,
    max_iter=200,
    return_details=True,
)
print(details["thresholds"])
for stage in details["history"]:
    print(stage["threshold"], stage["n_active_points"], stage["objective"])
```

The persistence schedule is built automatically from the input diagrams
(top of the schedule = "only the most persistent points"; bottom = "all
points"). You can supply your own schedule via the `thresholds` argument
if you have a problem-specific reason to do so.

For most exploratory work, `progressive_frechet_mean_multistart` is the
recommended default -- it is the most robust against the bad-local-minimum
pathology, and the extra cost is dominated by the final stage (which
matches what the one-shot optimizer would have done anyway).

## Reading `return_details`

The `details` dict (returned when `return_details=True`) contains, for the
non-progressive multistart:

- `"objective"`: the final objective value of the winning run.
- `"runs"`: a list of `{"start", "barycenter", "objective"}` dicts, one
  per seed.

And for `progressive_frechet_mean_multistart`:

- `"thresholds"`: the persistence schedule that was used.
- `"history"`: per-stage records with `stage_index`, `threshold`,
  `n_active_points`, `barycenter`, and `objective`.

## Plotting the result

The barycenter is just an `(m, 2)` array, so {py:func}`oineus.plot_diagram`
plots it directly:

```{code-block} python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
oin.plot_diagram({i: d for i, d in enumerate(diagrams)}, ax=axes[0],
                 title="Inputs")
oin.plot_diagram(bary, ax=axes[1], title="Barycenter", color="C3")
plt.show()
```

See `examples/python/example_frechet_mean.py` for a multi-panel comparison
of the different init strategies on the same input.

## See also

- {doc}`diagrams_distances` -- the Wasserstein distance is the metric
  the Fréchet mean is taking the mean *of*.
- {doc}`performance` -- the inner Hera calls are the dominant cost; tune
  `wasserstein_delta` and `n_threads` first.
- `examples/python/example_frechet_mean.py`, `tests/test_frechet.py`,
  `tests/test_fr_dgm_random.py`, `tests/test_fr_dgm_vertebra.py` --
  runnable examples and ground-truth tests.
