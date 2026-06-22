# Benchmarks

Off by default. Enable with `-Doin_build_benchmarks=ON`:

```
cmake -S . -B build -Doin_build_benchmarks=ON
cmake --build build --target bench_diagram bench_boundary -j8
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

---

# `bench_boundary` -- packed-uid boundary / coboundary construction

Prototype + measurement for a compact simplex representation. Instead of a
filtration holding a vector of standalone simplices (each an explicit
`std::vector<vertex>` on the heap), each cell is a single packed integer (the
uid); the per-type domain info lives once in the filtration; and a policy
computes boundary/coboundary directly from the packed uid -- with no per-face
vector allocation, and, for Freudenthal/cubical, a **direct coboundary that
skips the antitranspose**.

This benchmark builds real filtrations with the current ("master") code, then
times master's `boundary_matrix` / `coboundary_matrix` against the packed
schemes, producing the identical `MatrixData`. Every packed result is verified
equal to the master ground truth (boundary) / the plain boundary transpose
(coboundary) before its timing is reported. Single-threaded, to isolate
per-column cost (all of these builds parallelize the same way -- except the
antitranspose, which is a global scatter that does not).

Encodings:
- **VR** -- (a) Bauer combinatorial uid, unranked on demand (Ripser-style);
  (b) simple bit packing, k bits per vertex id. master = `Simplex<Int>` (sorted
  vertex vector). VR has no direct coboundary (cofacets need the neighbor
  graph), so coboundary stays antitranspose.
- **Freudenthal** -- packed `(anchor_vertex_id << type_bits | simplex_type)`; a
  precomputed table maps `(type, facet)` and `(type, cofacet)` to id-offsets, so
  boundary and coboundary are pure integer arithmetic. master = `Simplex<Int>`.
- **Cubical** -- packed cube uid (`anchor << 3 | face-bits`) + shared domain;
  direct coboundary from `Cube`'s own math. master = `Cube<Int,D>` carrying a
  per-cell `GridDomain` copy + antitranspose coboundary.

For Freudenthal/cubical the geometric uid is a small dense integer, so the
`uid -> sorted_id` map can be a flat direct-address array; hash vs flat is timed
as an ablation. Maps are prebuilt (untimed) in all cases, matching master (whose
map is built during filtration construction).

```
./build/benchmarks/bench_boundary --only vr   --n-points 150 --vr-max-dim 3
./build/benchmarks/bench_boundary --only grid --grid-side 72
./build/benchmarks/bench_boundary --only cube --grid-side 96 --reps 7
```

## Results (16 cores, AppleClang, macOS arm64, jemalloc on, single-thread, ms)

**VR**, complete 3-skeleton of random points in R^3 (boundary build only):

| size            | master | bit-packed   | Bauer (unrank) |
|-----------------|-------:|-------------:|---------------:|
| 8.5M (120 pts)  | 2963   | 769  (3.85x) | 2078  (1.43x)  |
| 20.8M (150 pts) | 8419   | 2439 (3.45x) | 6625  (1.27x)  |

**Freudenthal** 3D grid, side 72 (9.40M cells):

| op         | master (antitr. for cob) | new + hash   | new + flat    |
|------------|-------------------------:|-------------:|--------------:|
| boundary   | 2928                     | 365 (8.0x)   | 226 (12.9x)   |
| coboundary | 3106                     | 585 (5.3x)   | 324 ( 9.6x)   |

**Cubical** 3D grid, side 96 (6.97M cells):

| op         | master (antitr. for cob) | new + hash   | new + flat   |
|------------|-------------------------:|-------------:|-------------:|
| boundary   | 1275                     | 470 (2.7x)   | 195 (6.6x)   |
| coboundary | 1422                     | 557 (2.6x)   | 198 (7.2x)   |

Per-cell footprint: VR/Freudenthal master cell = 80 B **plus a separate heap
vertex array per cell**; cubical master cell = 48 B (carries a `GridDomain`
copy). Packed cells: VR 16 B, Freudenthal 8 B, cubical 4 B -- all heap-free.

## Findings

1. **The compact representation wins decisively everywhere, and the win grows
   with size** (cache effects): VR boundary 3.5-3.9x, Freudenthal boundary up to
   12.9x and direct coboundary up to 9.6x, cubical 6.6x / 7.2x.

2. **Master's bottleneck is allocation, not arithmetic.** For VR/Freudenthal the
   cost is the per-face `std::vector` allocation in `Simplex::boundary()` plus
   the pointer-chase to each cell's heap vertex array -- on top of jemalloc.
   Even Bauer-as-storage (which *adds* unranking) beats master by removing the
   allocation; bit-packing removes both and wins outright.

3. **Of the two VR encodings, bit-packing is the clear winner** (~3.5x vs
   ~1.3x). Reserve the Bauer combinatorial numbering for when its
   dimension-independent global index is actually needed; for plain storage,
   bit-pack the vertex ids.

4. **Direct coboundary is both faster and structurally better than
   antitranspose.** It is per-column independent (lock-free parallel), whereas
   the antitranspose is a global scatter that needs the whole boundary matrix
   materialized first and cannot be built lock-free (filtration.h says as much).
   The single-thread numbers understate this; the gap widens in parallel.

5. **The dense "flat" map is a representation-enabled win** (a further 1.6-2.8x
   over a hash map for Freudenthal/cubical), available precisely because the
   domain lives in the filtration and the geometric uid is a small dense
   integer. VR uids are not dense, so VR keeps a hash map.
