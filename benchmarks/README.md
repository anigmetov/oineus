# Benchmarks

Off by default. Enable with `-Doin_build_benchmarks=ON`:

```
cmake -S . -B build -Doin_build_benchmarks=ON
cmake --build build --target bench_diagram -j8
./build/benchmarks/bench_diagram             # flags are documented at the top of bench_diagram.cpp
```

## `bench_diagram` -- parallel diagram extraction

Builds a large filtration (random 3D Freudenthal grid, or VR of random points),
reduces it once (NOT timed), then times three extractions -- serial,
taskflow-parallel (`diagram_general_par`), and raw-`std::thread`-parallel
(`diagram_general_par_stdthread`) -- at several thread counts. All three are
checked to produce identical diagrams (sorted-multiset equality) before timing.

```
./build/benchmarks/bench_diagram --grid-side 75 --reps 5            # ~10M cells, homology
./build/benchmarks/bench_diagram --grid-side 75 --reps 5 --dualize  # cohomology
./build/benchmarks/bench_diagram --mode vr --n-points 400           # VR (huge low-dim diagram)
```

A Python end-to-end timer (GIL release + numpy export included) is in
`bench_diagram.py`:

```
PYTHONPATH=build/bindings/python python benchmarks/bench_diagram.py --grid-side 75
```

## Results (8 physical cores, AppleClang, macOS arm64, jemalloc on)

Random 3D Freudenthal grid, ~10.6M cells, fused reduce, homology:

| threads | taskflow ms | std::thread ms | speedup (tf) | tf vs std::thread |
|--------:|------------:|---------------:|-------------:|------------------:|
| 1       | 79.8        | 81.1           | 1.02x        | -1.7%             |
| 2       | 63.9        | 67.1           | 1.27x        | -4.7%             |
| 4       | 39.6        | 42.7           | 2.04x        | -7.2%             |
| 8       | 27.5        | 29.5           | 2.94x        | -6.5%             |

(serial baseline 81.0 ms; 135924 diagram points.)

Cohomology on the same grid: 2.16x at 8 threads. Python end-to-end (numpy
export included): 2.59x at 8 threads.

VR of 350 random points, ~7.0M *diagram* points: only 1.65x at 8 threads.

## Per-phase breakdown (8 threads, measured)

The extraction has four phases: A = invert `_pivots -> col_to_low` (serial),
B = pivot-row bitmap (parallel), C = emit into per-worker diagrams (parallel),
merge = concatenate per-worker diagrams (serial). Wall-clock ms:

| input (8 threads)              | A invert | B bitmap | C emit | merge | total |
|--------------------------------|---------:|---------:|-------:|------:|------:|
| grid, 10.6M cells, 136K points |     10.5 |      2.5 |   12.5 |   0.6 |  26.0 |
| VR, 7.1M cells, 7.0M points    |      3.0 |      1.5 |   22.0 |  31.0 |  58.0 |

This is the real story behind the speedup, and it differs by input:

- **Per-element work is modest and linear, but not negligible.** Most columns
  emit nothing (just a `col_is_zero` test + a bitmap read); columns that do
  emit pay a handful of bounds-checked filtration lookups. So C is real work
  that parallelizes well (e.g. VR: ~22 ms wall for 7M emitted points).

- **The merge dominates *only* when the diagram is large** (|diagram| ~ n_cols,
  i.e. low-dim VR). It is effectively a `memmove` of N x sizeof(DgmPoint)
  (~56 B) -- 7M points is ~400 MB read + 400 MB write, ~31 ms single-threaded
  (~26 GB/s), which is ~53% of the VR total. So yes: for VR, extraction is
  cheap-ish linear work and the serial memmove-like merge is the bottleneck.

- **For grids the merge is negligible** (0.6 ms; the diagram is tiny). There the
  serial limiter is **pass A, the `_pivots` inversion** (~10.5 ms, ~40% of the
  parallel run) -- a single-threaded scatter over n_cols. The parallel phases
  B+C are memory-bandwidth-bound.

## Findings

1. **Parallelism buys ~2-3x at 8 threads** (2.94x grid homology, 2.16x
   cohomology, 1.65x large-diagram VR). Sublinear, and the limiter depends on
   the input: the serial pass-A inversion for small diagrams (grids), the
   serial merge for large diagrams (VR). Both are serial O(n) passes.

2. **taskflow has no overhead vs raw std::thread here -- it is slightly faster**
   (negative "overhead", -1% to -9%). `diagram_general_par` creates one
   `tf::Executor` and reuses its pool across both parallel passes, whereas the
   std::thread variant spawns a fresh set of threads for each pass (two
   spawns). Pool reuse wins. So taskflow is the right choice for the default
   path; there is no reason to hand-roll threads.

3. **Follow-up to lift 8+ thread scaling:** parallelize the two serial passes.
   The merge -> per-(thread,dim) prefix-sum offsets + parallel copy into a
   presized result (helps VR most). Pass A -> a race-free parallel scatter into
   `col_to_low` (helps grids most). Both were left serial because the plan
   assumed |diagram| << n_cols and an O(n_cols) inversion was "cheap" -- the
   measurements above show each becomes the bottleneck for one of the two
   input regimes.
