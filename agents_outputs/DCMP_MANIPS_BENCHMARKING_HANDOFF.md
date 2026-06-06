# Handoff: dcmp_manips — benchmarking on bigger filtrations

Next session focus: **benchmark the dynamic decomposition-update methods on a
range of larger, realistic filtrations** and characterize where warm updates
beat from-scratch reduction. The implementation is done and validated; this is
a measurement/characterization task.

Branch: `dcmp_manips` (off `master`). Do NOT push. Commit per logical step.

---

## 1. What exists (commits, newest last)

```
8b51749  Add decomposition-manipulation algorithms (vineyards, moves, warm starts)
8d47dd3  Speed up moves with donor-style conjugate-once update      (superseded)
f3d0955  Fix correctness/honesty issues (review stage 1)
9b6d2fe  Localize vineyard transpose via a row-incidence index (stage 2 prototype)
d1c4486  Localize moves by reusing the localized transpose
dc29a93  Schedule moves on the changed support (stage 2)
```

The methods live on `VRUDecomposition<Int>` (`include/oineus/decomposition.h`),
bound to the Python `Decomposition` class. **Homology only** (throw if
`dualize`); require a reduced decomposition with V (`compute_v=True`).

Python API (all take an optional `DecompositionManipStats`):
- `make_dynamic(n_threads=1)` / `is_dynamic()` -- opt into dynamic mode: builds
  row-incidence indices so updates are localized. Static decomps pay nothing.
  `reduce()` and the batch methods invalidate it; first manipulation rebuilds.
- `transpose(i, stats)` / `transpose_to(new_to_old, stats)` -- vineyards.
- `move(i,j)` / `move_right` / `move_left` / `apply_move_schedule(new_to_old, stats)`
  -- moves (now realized as localized transposition sequences) + LIS-minimal,
  support-windowed schedule.
- `update_with_permutation(new_to_old, stats)` -- Luo-Nelson Alg 2 (batch
  O(nnz) one-shot; NOT localized -- use apply_move_schedule for small changes).
- `update_with_edits(new_to_old, new_boundary, new_dim_first, new_dim_last, stats)`
  -- Luo-Nelson Alg 3 (insert/delete + resize; batch one-shot).
- `is_reduced_consistent()` -- R reduced and D V == R (no V-upper-tri needed).

`new_to_old` is a permutation vector: `new_to_old[k]` = old sorted-id now at
position k. For `update_with_edits`, `new_to_old[k] = -1` means inserted;
deletions must be coface-closed (any valid filtration edit is).

`DecompositionManipStats` fields to record: `elapsed_total`,
`elapsed_schedule_build`, `elapsed_transpose`, `elapsed_move`,
`elapsed_permute`, `elapsed_rereduce`, `elapsed_resize`; `n_transpositions`,
`n_moves`, `n_column_additions_r/_v` (and `.n_column_additions()`),
`n_queries`; **`n_columns_scanned`** (the honest metric: columns visited in
whole-matrix passes -- this is what actually drives wall-clock, NOT the
column-op count); `nnz_r/v_before/after`. Pickleable.

---

## 2. Key findings so far (don't re-derive)

- **Localized transpose is O(star), independent of n.** One transpose at 391k
  cells scans ~330 columns vs ~1.5M for the old full-scan (~4700x). `make_dynamic`
  (8 threads) builds the index in ~47ms at 391k.
- **Warm CAN beat 16-core from-scratch pairing-only reduce, but only in a narrow
  regime.** 391k-cell lower-star, 1 perturbed pixel (7 cells move): warm total
  2.81ms (schedule_build 0.69ms, matrix work 0.01ms) vs full@16t ~3ms.
- **Win regime = small support AND small DISPLACEMENT.** Displacement, not just
  the number of changed cells, is what kills it: a perturbed cell whose value
  jumps to a random rank travels a huge distance (many transpositions). With
  **random** grid data, even a 3-pixel change shifts ~60% of cells -> warm loses
  by 100x+. Need smooth/well-separated values or genuinely local edits so cells
  move only a few ranks.
- **For large changes, from-scratch parallel wins decisively** (the warm methods
  are then ~O(nnz) and serial). See bench_dcmp_vs_parallel.py: global
  perturbation, warm_perm 84ms vs full@8t 12ms at 391k.
- **Residual floor: ~0.7ms O(n) prep** in apply_move_schedule (`invert_perm` +
  position arrays), inherent to taking a full `new_to_old`. A support-explicit
  API (pass only changed cells; order-statistics position tracking) would remove
  it -- this is the known next optimization, NOT yet built.

---

## 3. Existing benchmark scripts (`agents_outputs/`)

- `bench_dcmp_manips.py` -- correctness+metrics harness over 1-parameter families
  (lower-star, VR reorder, VR insert/delete) and an optimization step. Compares
  full vs vineyards vs moves vs warm_perm vs warm_edits; validates every diagram
  vs full recompute; emits CSV+MD with col_ops, **col_scans**, transp, moves,
  wall-clock. `--quick` for small sizes. Reuses `tests/data_utils.py`.
- `bench_dcmp_vs_parallel.py` -- warm vs from-scratch **parallel pairing-only**
  reduce across thread counts {1,2,4,8,16}, global vs localized perturbation.
- `bench_localized_warm.py` -- the tiny-change demo (make_dynamic + a 1-pixel
  delta, breaks out schedule_build vs transpositions vs full@{1,8,16}t).

Helpers: cell matching across two filtrations by `cell.uid` (an int):
`new_to_old[k] = {c.uid: c.sorted_id for c in fil_old.cells()}[fil_new.cells()[k].uid]`.
New boundary for edits = `[list(c) for c in oin.Decomposition(fil_new, False).d_data]`.
Filtration exposes `.cells()`, `.dim_first`, `.dim_last`, `cell.uid`, `cell.dim`,
`cell.sorted_id`, `cell_value_by_sorted_id`, `id_by_sorted_id`, `sorted_id_by_id`.

---

## 4. What to do next (the actual task)

Benchmark across **filtration types and sizes**, focusing on the
small-change/dynamic regime where the methods are meant to help:

Filtration types to cover (oineus builders + tests/data_utils.py):
- lower-star Freudenthal grids (2D and 3D), white-noise and SMOOTH data
- cubical (`cube_filtration`)
- Vietoris-Rips on point clouds (`vr_filtration`; `du.sample_tori`, etc.)
- alpha (`alpha_filtration`) -- near-linear reduction, interesting baseline

Sizes: sweep up to ~1e6 cells (256^2 grid ~ 391k; 3D grids; larger VR). Watch
memory: the row index is ~O(nnz) extra (3 indices for R,V,D).

For each: build, reduce(compute_v), `make_dynamic(n_threads)`, then apply a
**controlled small change** and measure. The crux is generating changes with
**small displacement**:
- grids: nudge a few pixel values by a small delta on SMOOTH base data (so a
  cell moves only a few ranks); vary delta and #pixels to sweep displacement.
- VR/alpha: perturb a few point positions slightly.
Report, per (input, size, change-magnitude): total displacement
(= n_transpositions), n_moves, **n_columns_scanned**, schedule_build vs
transposition time, warm total, and from-scratch pairing-only at
n_threads in {1,4,8,16}. Find the crossover (how small must the change be for
warm to win) per filtration type/size.

ALWAYS validate correctness: `pairing(dcmp.r_data) == pairing(from_scratch(dcmp.d_data))`
where `pairing(R) = frozenset((max(col), c) for c,col in enumerate(R) if col)`.

Open questions worth answering with data:
- How does the win crossover scale with n and with nnz/n (avg star size)?
- Per-transpose constant factor (~2.3us at 391k) -- is the index maintenance
  (sorted-vector insert/erase) the bottleneck? Would unordered_set or a
  different row-index rep help? (memory tradeoff: sets ~10x the vectors.)
- Is the ~0.7ms O(n) schedule prep worth removing (support-explicit API) before
  benchmarking, or benchmark first and let the data justify it?

---

## 5. Build / run (IMPORTANT: sandbox quirks)

- `uv run` is BLOCKED (cache dir not writable). Use the venv python directly:
  `./.venv/bin/python`. Never `uv sync`/`uv add` (reinstates the editable
  install that shadows the build tree).
- Always set `MPLCONFIGDIR="$TMPDIR/mpl"` (oineus.vis imports matplotlib).
- Build: `cmake --build build -j4` (Release; venv python on Python_EXECUTABLE).
  After editing C++, rebuild then run with
  `PYTHONPATH="$PWD/build/bindings/python" ./.venv/bin/python ...`.
- Tests via ctest from `build/` (sets correct CWD; the C++ binary's
  `a_6.txt` test fails if run directly from repo root -- not a real failure):
  `cd build && ctest -R "^all$|dcmp-manips" --output-on-failure`.
- C++ manip tests: `tests/tests_dcmp_manips.cpp` (Catch2, in `tests` binary).
  Python: `tests/test_dcmp_manips.py` (ctest `py-dcmp-manips`).
- Bash gotcha: `!=` in inline `python -c` triggers shell history expansion --
  write a script file instead of `-c`.
- Memory note: `~/.claude/.../memory/dcmp_manips_branch.md` has the running
  state.

---

## 6. Gotchas

- Manipulations are homology-only (dualize=false) and need compute_v=True.
- Filtrations are **dimension-blocked** (all dim-0, then dim-1, ...); permutations
  are within-dimension. `dim_first[d]/dim_last[d]` give the blocks.
- `update_with_edits` deletions must be coface-closed; inserted cells must keep
  the order dimension-blocked (append top cells at the end, or insert in-dim).
- `make_dynamic` memory: ~3x an extra copy of the nnz (row indices for R,V,D).
- The per-op localized cost has a real constant factor; for LARGE total
  displacement the warm path loses to parallel from-scratch -- benchmark both
  and report the crossover, don't assume warm always wins.
