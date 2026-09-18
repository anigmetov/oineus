# Selected ELZ restoration and matrix-access benchmark

Measured 2026-09-18 on macOS 26.6.1 arm64 (16 logical CPUs), Python 3.12.9,
AppleClang 21, Release build with `oin_use_jemalloc=OFF`.
Native implementation: `f6bf9891f0110fe25c210e06567620788b041c47`.

The input is a 192 x 192 random Freudenthal grid (seed 1701), with 219,651
columns. Each trial starts a fresh process and performs a four-thread reduction
with V and clearing. Results below are medians of three trials; memory columns
are maxima across those trials. Homology and cohomology, retained working
storage and materialized storage, are measured separately. Apparent pairs are
not enabled. `working` uses fused `oineus.reduce`; `materialized` uses the classic
`Decomposition(fil).reduce` path, which also retains D.

Restoration and access are timed separately. Access trials first restore the
requested columns outside the timed phase, then either copy just their R/V
supports with getters, export R/V once to cached lists, or export R/V once to
CSC and slice those exports. The returned supports are kept alive during memory
measurement. Time includes the Python binding calls and requested-support list
construction; filtration construction and reduction are excluded.

**Memory convention:** peak RSS is the process high-water mark, including setup.
Growth is the increase over the high-water mark immediately before the measured
phase. A zero growth value does not mean zero allocations: setup can mask a
smaller subsequent peak. These are OS measurements, not Python-only allocation
counts, and include native storage. Independent processes avoid contamination
by prior trials.

## Restoration

### 32 requested columns

| Mode | Storage | Threads | Time (ms) | Peak RSS (MiB) | RSS growth (MiB) |
|---|---|---:|---:|---:|---:|
| Homology | working | 1 | 0.049 | 101.6 | 0.0 |
| Homology | working | 4 | 0.071 | 101.8 | 0.1 |
| Homology | materialized | 1 | 0.041 | 127.1 | 0.0 |
| Homology | materialized | 4 | 0.072 | 127.2 | 0.1 |
| Cohomology | working | 1 | 0.113 | 102.4 | 0.0 |
| Cohomology | working | 4 | 0.141 | 102.7 | 0.1 |
| Cohomology | materialized | 1 | 0.051 | 136.2 | 0.0 |
| Cohomology | materialized | 4 | 0.094 | 136.1 | 0.1 |

### 109,825 requested columns

| Mode | Storage | Threads | Time (ms) | Peak RSS (MiB) | RSS growth (MiB) |
|---|---|---:|---:|---:|---:|
| Homology | working | 1 | 72.946 | 117.7 | 11.8 |
| Homology | working | 4 | 24.221 | 118.0 | 12.3 |
| Homology | materialized | 1 | 72.276 | 142.0 | 10.9 |
| Homology | materialized | 4 | 23.549 | 144.3 | 12.0 |
| Cohomology | working | 1 | 168.080 | 116.3 | 10.0 |
| Cohomology | working | 4 | 46.356 | 116.3 | 10.3 |
| Cohomology | materialized | 1 | 161.683 | 148.9 | 9.2 |
| Cohomology | materialized | 4 | 45.935 | 151.1 | 9.5 |

## Access after restoration

### 32 requested columns

| Mode | Storage | Access | Time (ms) | Peak RSS (MiB) | RSS growth (MiB) |
|---|---|---|---:|---:|---:|
| Homology | working | Column getters | 0.018 | 101.7 | 0.0 |
| Homology | working | Cached full lists | 28.467 | 196.1 | 94.2 |
| Homology | working | Cached full CSC | 52.708 | 160.2 | 58.3 |
| Homology | materialized | Column getters | 0.014 | 127.2 | 0.0 |
| Homology | materialized | Cached full lists | 22.624 | 197.0 | 69.8 |
| Homology | materialized | Cached full CSC | 44.693 | 161.2 | 32.2 |
| Cohomology | working | Column getters | 0.265 | 102.9 | 1.4 |
| Cohomology | working | Cached full lists | 30.729 | 206.5 | 104.6 |
| Cohomology | working | Cached full CSC | 53.202 | 166.4 | 64.6 |
| Cohomology | materialized | Column getters | 0.270 | 137.0 | 1.3 |
| Cohomology | materialized | Cached full lists | 24.662 | 214.8 | 79.3 |
| Cohomology | materialized | Cached full CSC | 46.837 | 176.2 | 39.0 |

### 109,825 requested columns

| Mode | Storage | Access | Time (ms) | Peak RSS (MiB) | RSS growth (MiB) |
|---|---|---|---:|---:|---:|
| Homology | working | Column getters | 57.031 | 191.0 | 73.3 |
| Homology | working | Cached full lists | 74.440 | 251.7 | 133.9 |
| Homology | working | Cached full CSC | 92.982 | 226.8 | 109.2 |
| Homology | materialized | Column getters | 56.072 | 215.6 | 73.3 |
| Homology | materialized | Cached full lists | 67.028 | 253.9 | 110.4 |
| Homology | materialized | Cached full CSC | 87.215 | 227.8 | 85.3 |
| Cohomology | working | Column getters | 51.541 | 192.0 | 76.5 |
| Cohomology | working | Cached full lists | 73.792 | 257.7 | 141.4 |
| Cohomology | working | Cached full CSC | 95.711 | 231.0 | 115.6 |
| Cohomology | materialized | Column getters | 51.404 | 224.7 | 75.7 |
| Cohomology | materialized | Cached full lists | 68.272 | 265.3 | 116.4 |
| Cohomology | materialized | Cached full CSC | 90.220 | 239.5 | 90.2 |

## Interpretation

For the 32-column requests, one restoration thread is faster than four: worker
startup costs more than the small amount of repair work. For 109,825 targets in
retained working storage, four threads reduce homology time from 72.946 to
24.221 ms (3.0x) and cohomology time from 168.080 to 46.356 ms (3.6x).

For sparse access, the column getters avoid tens of milliseconds and roughly
94-105 MiB of additional process high-water memory incurred by full list exports
from working storage. For the half-matrix requests, getters also won this
particular comparison, but cached exports remain useful when callers will reuse
many columns repeatedly. There is no measured need here for native Python view
objects with additional lifetime and mutation rules. A single cached full export
is supported and its cost should be paid once, after restoration.

These measurements cover one grid family and one host, not a general scaling
claim. Noncanonical parallel factors can vary across runs, so repair work and
timings can vary even with a fixed filtration seed.

## Reproduce and validate

From the repository root after building the extension:

```sh
PYTHONPATH="$PWD/build/bindings/python" .venv/bin/python benchmarks/bench_selected_elz.py --side 192 --repeats 3 --output benchmarks/selected_elz_results.json
```

The [raw results](selected_elz_results.json) contain all 120 process trials and
the exact native module path. The benchmark does not measure correctness; that
is covered separately by 211 focused Python tests and 2,191 native assertions,
including exact serial-oracle comparison, untouched targets, factorization,
cache validity, interruption, and injected apparent-resolver failure. The full
CTest suite passed 83/83 tests in 57.81 seconds. Independent review also
validated the resolver failure with AddressSanitizer.

See [the usage example](../examples/python/selected_representatives.py) and
[API guide](../docs/topics/decomposition.md).
