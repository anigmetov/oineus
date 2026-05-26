---
orphan: true
---

# Topological optimization: doing it economically

```{note}
Advanced / internals. Most users do not need this page;
{py:func}`oineus.diff.persistence_diagram` makes the right choices
automatically. The notes below explain *why* it does what it does, for
readers who want to follow or extend the implementation.
```

This note is about how to *spend the least computation* on a
gradient step through a persistence diagram. It is independent of
any specific library; it just assumes that the building blocks
below are available, and asks: given a concrete loss, which of
them do we actually need?

Reading prerequisite: the boundary-matrix reduction `R = D V` (or
the cohomology dual `R_coh = D_coh V_coh`), the matrix `U = V^{-1}`,
and the pairing read off `low(R)`.

## 0. Building blocks we assume

For each filtration we can build a *decomposition*. A decomposition
exposes a small toolbox:

- **Reduce, serial or parallel.** Serial is deterministic and can
  produce extra invariants for free (see ELZ below). Parallel is
  faster on large inputs but its column-add order is non-deterministic.
- **Reduce with or without `V`.** Building `V` alongside `R` is a
  small constant-factor extra cost.
- **Reduce with or without `U` in-band.** "In-band" means we record
  every column-add operation into U as the reduction proceeds.
- **Reduce with or without clearing.** Clearing is a per-column
  trick that skips the reduction of every column known a priori to
  be zero (positive simplices once their negative counterpart is
  reduced).
- **Restore ELZ.** A post-processing pass over `(R, V)` that
  rewrites V so that the resulting decomposition is the same as the
  one a serial-no-clearing reduction would have produced.
- **Compute U from V, fully or partially.** Given a known-good V
  (in ELZ form) we can recover any subset of the rows of `U` by
  selected inversion, without ever building U during the reduction.

Not every combination of knobs is realizable:

- **Parallel + in-band U** is fundamentally incompatible. The
  in-band construction of U is a global, ordered ledger of
  column-add operations; reproducing it under racing threads costs
  more than the work it would record.
- **Clearing + in-band U** is incompatible too: clearing skips
  entire columns, and skipping them loses the very operations that
  in-band U would write down.

What *is* realizable:

| Reduction | Clearing | V | U (in-band) | ELZ form? |
|---|---|---|---|---|
| serial | off | optional | optional | yes, by construction |
| serial | on | optional | -- | not until `restore_elz` |
| parallel | off | optional | -- | not until `restore_elz` |
| parallel | on | optional | -- | not until `restore_elz` |

The ELZ column is the part most people overlook; we will come back
to it.

## 1. Why we are even talking about ELZ

The selected inversion of V (computing rows of `U = V^{-1}` on
demand) is a residual-style forward elimination against `V^T`. It
terminates because each elimination step strictly increases the
lowest live entry of the residual. *That termination property is
exactly the ELZ property of V.*

If V is not in ELZ form, two things can go wrong inside the row
solver:

- a residual entry points at a column whose `R` is zero (a positive
  simplex that was cleared, or a stale column never written cleanly
  by a parallel thread). Eliminating it does nothing -- you spin
  forever on the same lowest entry.
- a residual entry brings in a smaller pivot than the one you just
  eliminated. The walk "goes backward" and either loops or returns
  garbage.

So: **partial / full U-from-V is correct iff V is in ELZ form.**

ELZ is produced by exactly two situations:

- Serial reduction without clearing, end-to-end. ELZ falls out.
- An explicit `restore_elz` pass run *after* any other reduction.
  It rewrites V column by column to satisfy the ELZ predicate.

Anything else (clearing on, parallel, mix-and-match) leaves V in a
formally valid reduced state -- the *pairing* `low(R)` is correct --
but V itself is not ELZ until you patch it.

## 2. Computing the diagram (the cheap part)

Before there is any loss, there is a forward pass: we need the
diagram to even define the function we want to differentiate.

To read the diagram we need only the pairing, which is `low(R)`.
We do *not* need V; we do *not* need U; we do *not* need ELZ.
So the cheapest forward is the one that produces a valid R and
nothing else: parallel reduction with clearing on, no V, no U,
no ELZ restore. Read off the diagram values, hand them to the
loss, done.

This is the recipe to use for the forward pass *whenever* the loss
in question only needs the diagram values themselves and does not
ask us to do anything more sophisticated with the matrices.

The choice of *which* side to reduce (homology or cohomology) is
orthogonal to all of the above. Pick whichever side reduces faster
on this filtration shape -- for Vietoris-Rips, cohomology under
clearing is the standard default; for lower-star on grids, homology
tends to be similar; for one-sided constructions it can go either
way. The diagram is invariant.

## 3. What the loss tells us

Suppose we evaluate the loss `L` and compute its gradient with
respect to the diagram coordinates `(b_i, d_i)`. We now have a
vector of "desired moves":
- `dL / db_i > 0` says we want to decrease `b_i`,
- `dL / db_i < 0` says we want to increase it,
- and analogously for `d_i`.

There are two qualitatively different ways to propagate this
gradient back to the filtration values.

### 3.1 Diagram loss (cheap, narrow)

Treat the diagram-to-filtration-values map as a plain index lookup:
`(b_i, d_i) = (fil_values[birth_idx_i], fil_values[death_idx_i])`.
The chain rule then says the gradient lands on exactly two
simplices per pair -- the birth-creator and the death-creator -- and
on nothing else. No critical sets, no inversion, no second
decomposition.

If this is the gradient mode you are using, *nothing beyond the
diagram is ever needed.* The forward recipe of Section 2 is the
whole computation. The backward is a scatter into the
filtration-values gradient. Done.

### 3.2 Critical-set loss (expensive, dense)

Treat the gradient on each pair as a *singleton loss*: the desired
target is `(b_i', d_i') = (b_i, d_i) - step * grad_i`, and to
realize that target the filtration values of an entire **critical
set** of simplices have to move. The critical set is the support of
a column of V or a row of U, truncated by the value bound implied
by the target.

There are four directions a pair can move, and each direction reads
exactly one of the four matrices `(V_hom, U_hom, V_coh, U_coh)`:

| Move | Matrix read |
|---|---|
| increase birth | V (cohomology) |
| decrease birth | U (cohomology) |
| increase death | U (homology) |
| decrease death | V (homology) |

(Filtration `negate` flips value direction against filtration
direction; the *matrix* mapping above is unchanged.)

The first economical observation: the loss only asks for the
*non-zero* sign patterns of the diagram gradient. A pure
death-decreasing loss needs `V_hom` and nothing else; a pure
birth-increasing loss needs `V_coh` and nothing else. We can
inspect the per-coordinate min and max of the diagram-gradient
tensor in one pass and read off, for each of the four matrices,
whether *any* pair needs it. This classification takes a single
linear traversal of the gradient and tells us exactly which of the
following four reductions we will (and won't) need to materialize.

## 4. The economical reduction plan for critical-set loss

We always reduce *one* side in the forward pass (Section 2). Call
that the "primary" side. The other side is the "secondary".

Once we know which matrices the loss asks for, here is the
cheapest way to obtain them.

### 4.1 V is essentially free; U is essentially never

The cheapest reduction is parallel + clearing + R only. V can be
added on top of this at modest extra cost. U cannot be added on
top of this *at all* (the table in Section 0 forbids it). Once we
decide we are going down the critical-set path, *every reduction
we run should compute V*. We will need V either to walk a column
critical set directly, or to derive U from it.

### 4.2 If we already need U, build V the right way the first time

To compute U-from-V (partial or full), V must be in ELZ form. The
two ways to get an ELZ V are:

- Reduce serially without clearing. ELZ falls out for free, but we
  pay the full no-clearing cost.
- Reduce however we like (parallel, clearing, ...) and then call
  `restore_elz` afterwards.

For any nontrivial filtration the parallel-with-clearing path plus
a single `restore_elz` is *much* cheaper than a serial-without-
clearing pass, so this is the recipe to standardize on whenever U
is in the picture.

### 4.3 Treat U as on-demand, never in-band

In-band U construction is incompatible with both parallelism and
clearing -- the very tricks that make the reduction fast. So we
should never compute U in-band when efficiency matters. Instead,
once we have a clean ELZ V, recover U on demand via selected
inversion:

- if the loss only touches a few rows of U on this side, recover
  those rows only (partial-U-from-V),
- if it touches almost all rows, recover them all (full-U-from-V).

The crossover point between "few" and "almost all" is a tunable
threshold; in practice it sits around 60-80% of the dimension's
column range. Below the threshold, partial is cheaper because each
row solve is bounded by a target value; above it, the overhead of
the per-row setup dominates and a single full pass wins.

### 4.4 Touch the secondary side only if you have to

If the gradient sign pattern needs nothing on the secondary side
(`V_secondary`, `U_secondary` both unused), skip it entirely. If
it needs only V, reduce it once with the ELZ-producing recipe of
Section 4.2 and stop. If it also needs U, follow the recipe of
Section 4.3.

In particular, the primary side reduction done in the forward
pass *already* produced its V (Section 4.1 made that a standing
rule for critical-set mode), so on the primary side we never
re-reduce -- we only run partial-U or full-U on top of the V we
already have.

### 4.5 Always read the dispatch off the diagram, not the matrices

To resolve critical sets we need to know, for each moving
simplex, whether it is a birth-creator (positive) or a
death-creator (negative). That distinction is implicit in
`low(R)` and therefore in the diagram itself: every birth index
in the diagram is a positive simplex, every death index is a
negative one. We get this classification *for free* from the
diagram we already computed; we should not introduce a separate
"is this positive or negative" matrix lookup that requires another
side's reduction.

## 5. Putting it together

A complete recipe for one step of optimization through the
diagram:

**Forward:**

1. Build the decomposition with a deferred-reduction posture (do
   not pay anything yet).
2. Reduce the chosen side with the cheapest valid recipe for the
   loss mode you are using:
   - diagram loss: parallel + clearing, R only, no V, no ELZ.
   - critical-set loss: parallel + clearing, R + V, then
     `restore_elz`. This is the only place the forward path
     spends more on V; the cost is small and is precisely what
     unlocks cheap U recovery in the backward.
3. Read the diagram off `low(R)` and hand it to the user as the
   differentiable tensor.

**Backward:**

1. If the loss is a diagram loss, scatter the diagram gradient
   into the filtration-values gradient. Done.
2. If the loss is a critical-set loss:
   a. Classify the diagram gradient sign pattern in one pass and
      decide which of `(V_hom, U_hom, V_coh, U_coh)` are needed.
   b. For the primary side, V is already there; if its U is
      needed, run partial-U-from-V (fall back to full-U-from-V
      above the threshold).
   c. For the secondary side, if V is needed, reduce it with the
      ELZ-producing recipe (Section 4.2). If its U is needed,
      additionally run partial-U-from-V.
   d. With the matrices in hand, walk each pair's critical set
      using the corresponding row or column.
   e. Resolve conflicts (multiple pairs prescribing different
      targets for the same simplex) with whatever strategy the
      user picked (`Avg`, `Max`, `Sum`, `FixCritAvg`, ...).
   f. Scatter the resolved per-simplex targets as gradient
      contributions into the filtration-values gradient.

## 6. Invariants to keep in mind

The recipe above respects two structural invariants worth naming:

- **No reduction is repeated.** Each side is touched by at most
  one reduction across the forward and backward of a single step,
  and that reduction is at the highest level (R, V, ELZ) the step
  will ever need. Anything else is built on top of what is
  already there.
- **U is never built in-band when speed matters.** Partial- or
  full-U-from-V composes with the cheap parallel+clearing+ELZ
  reduction recipe; in-band U does not. The few situations where
  one might still want in-band U (a serial cross-check, a
  debugging path) are explicit choices, not the default.

## 7. When *not* to follow this plan

This plan is the right plan when the loss is small and dense in
the diagram-coordinate sense: a typical topological loss that
asks to push features off or onto the diagonal, match a template,
etc. There are situations where a different posture is sensible:

- The loss reads from the diagram *and* from some downstream
  matrix object directly (rare, but possible in research code).
  Then you may need to commit to a heavier reduction up front
  rather than reasoning about which matrices the gradient picks.
- The same filtration is used across many optimization steps with
  values changing only slightly between steps. Here it pays to
  cache reductions across steps and invalidate lazily, rather
  than rebuilding from scratch each forward. The reduction
  recipes are the same; the caching layer is on top.
- The loss is purely on infinite ("essential") features. Those
  contribute zero gradient by construction in the standard
  finite-points view of the diagram; if you really need to act on
  them, you are outside the scope of this note.

For everything else, the rule of thumb is small:

> Reduce one side as cheaply as the loss allows. If you will need
> U, build V the right way the first time and recover U later by
> selected inversion. Touch the second side only when the gradient
> sign pattern proves you must.
