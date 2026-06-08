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

## Findings

1. **Parallelism buys ~2-3x at 8 threads** for grid-style inputs (2.94x
   homology, 2.16x cohomology). It is sublinear because extraction is largely
   memory-bound (scanning the pivots / bitmap arrays) and two stages stay
   serial: the `_pivots -> col_to_low` inversion (pass A) and the per-dimension
   merge.

2. **taskflow has no overhead vs raw std::thread here -- it is slightly faster**
   (negative "overhead", -1% to -9%). `diagram_general_par` creates one
   `tf::Executor` and reuses its pool across both parallel passes, whereas the
   std::thread variant spawns a fresh set of threads for each pass (two
   spawns). Pool reuse wins. So taskflow is the right choice for the default
   path; there is no reason to hand-roll threads.

3. **Very large diagrams are merge-bound.** Low-dimensional VR produces a
   diagram with millions of points (7M here, comparable to the cell count), so
   the serial concatenation merge becomes the Amdahl bottleneck and speedup
   drops to ~1.65x. The obvious follow-up is to parallelize the merge
   (per-(thread,dim) prefix-sum offsets + parallel copy into a presized result);
   it was left serial here because for typical inputs |diagram| << n_cols and
   the merge is negligible. Pass A (the pivot inversion) is a smaller, similar
   serial-fraction candidate if 8+ thread scaling matters.
