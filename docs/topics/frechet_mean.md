# Fréchet mean of persistence diagrams

The Wasserstein barycenter of $N$ diagrams $D_1, \dots, D_N$ is the
diagram $D^*$ that minimizes the weighted Fréchet functional

$$ F(D) \;=\; \sum_{i=1}^N w_i \, W_q(D, D_i)^q. $$

The functional is non-convex, so the answer depends on the
initialization. For most inputs a single run seeded at the medoid is
enough.

## The simple recipe

```{code-block} python
import oineus as oin

# diagrams: list of (n_i, 2) numpy arrays, one per input (single dim).
bary = oin.frechet_mean(
    diagrams,
    init_strategy=oin.FrechetMeanInit.MedoidDiagram,
)
print(bary.shape)
```

That is the call you reach for first. It picks the input diagram closest
to the (uniform) center as the seed, runs Lloyd-style updates until
convergence, and returns the barycenter as an `(m, 2)` NumPy array.

Knobs you actually want to know about:

- `weights` -- 1D array, one entry per input diagram. Defaults to
  uniform.
- `wasserstein_delta` -- relative accuracy of each Hera call inside the
  iteration. The default `0.01` is fine for plotting; tighten to
  `1e-4` for publication-quality barycenters.
- `max_iter`, `tol` -- iteration cap and convergence tolerance on the
  objective.
- `ignore_infinite_points=True` -- strip essential pairs before
  optimization; use this whenever inputs have inconsistent essential
  counts.

The full argument list (other init strategies, custom seed, grid
parameters, ...) is in {py:func}`oineus.frechet_mean`.

## When the default is visibly wrong

The single-run optimizer can get stuck in shallow local minima. Two
escape hatches in increasing order of effort:

**Multistart.** Run from several seeds and keep the best:

```{code-block} python
bary, details = oin.frechet_mean_multistart(
    diagrams,
    starts=("medoid", "second_medoid", "farthest_from_medoid"),
    return_details=True,
)
print(details["objective"])
for run in details["runs"]:
    print(run["start"], run["objective"])
```

**Progressive multistart.** Heavy-tailed diagrams (a few long bars plus a
lot of near-diagonal noise) often need the high-persistence features to
lock in before the noise dominates the gradient. The progressive variant
solves a sequence of sub-problems on diagrams filtered by a decreasing
persistence threshold, warm-starting each stage from the previous
solution:

```{code-block} python
bary, details = oin.progressive_frechet_mean_multistart(
    diagrams,
    starts=("medoid", "second_medoid"),
    return_details=True,
)
print(details["thresholds"])
```

`details["history"]` contains a record per stage (threshold, number of
active points, barycenter, objective). The schedule is generated
automatically; supply `thresholds=...` to hand-tune.

## Plotting

The barycenter is just an `(m, 2)` array, so {py:func}`oineus.plot_diagram`
plots it directly. See {doc}`visualization` and
`examples/python/example_frechet_mean.py` for a multi-panel comparison of
the different init strategies on the same input.

## See also

- {doc}`diagrams_distances` -- the Wasserstein distance is the metric the
  Fréchet mean is taking the mean *of*.
- {doc}`performance` -- the inner Hera calls dominate the cost; tune
  `wasserstein_delta` and `n_threads` first.
- `examples/python/example_frechet_mean.py`, `tests/test_frechet.py`,
  `tests/test_fr_dgm_random.py`, `tests/test_fr_dgm_vertebra.py` --
  runnable examples and ground-truth tests.
