# Slim cells + shared geometry: the cell-representation refactor

> Production design doc. This supersedes the earlier benchmark-flavored framing.
> The work is staged (see "Phasing"); a session can stop and hand off at any
> green checkpoint. Slim cells are the committed direction (user), so this is the
> design for the *published* library, not a benchmark hack.

## 1. Why (the root cause)

`Cube<Int,D>` stores its `GridDomain<Int,D>` BY VALUE (`cube.h:93,95`;
`global_domain()` even returns by value, `cube.h:105`). A 3D cube is ~48-56 B of
which only ~8 B (the uid) is real information; the domain is copied into every one
of millions of cells. Consequences: cells are ~6x larger than necessary (cache),
every `boundary()`/`coboundary()` recomputes `anchor_vertex()` + per-dim
`domain.contains()`/`point_to_id()` against a private copy, and each call heap-
allocates a fresh `std::vector` for the result.

### Measurements that pin the cause (smooth 96^3 volume, 7M cells, 8 threads)

- `benchmarks/bench_boundary.cpp --only cube`: a packed-uid direct (co)boundary
  with a SHARED domain + flat `uid->sorted_id` is **6.37x** faster than the
  current antitranspose (276 vs 1762 ms); boundary build 5.48x.
- Decomposing: the **flat array vs the hash map is 2.4x alone** (654->276 ms);
  the rest is the light cell + buffer writes + direct-vs-antitranspose.
- In the live apparent path, swapping only the resolver's hash lookup for a flat
  array recovered just **18%** -> the bottleneck is the heavy cell + per-call
  allocation, NOT the lookup. Benchmark packed coboundary ~40 ns/cell; live
  resolver ~1.9 us/regeneration (~47x gap = cell weight + alloc + by-value
  domain).

So: slim cells speed up EVERY (co)boundary build (the common non-apparent path),
and incidentally flip apparent_opt from a memory-win-with-a-time-cost into a net
win.

## 2. Production architecture

Three design questions drove this (all settled with the user); the answers
converge on one shape: **compile-time policies for the hot core, one type-erased
wrapper at the Python boundary.**

### 2a. Parameterize on a cell/geometry POLICY; dissolve `FiltrationKind` from the hot path

`FiltrationKind` (a runtime enum) is a mild code smell: it recovers at runtime
what the type system threw away. The tell -- today **Freudenthal, VR, alpha, and
user filtrations all share one `Simplex` cell type**, so a runtime tag is needed
to tell them apart. The fix is to FINISH the parameterization: the (co)boundary +
geometry become a **compile-time policy/trait keyed on a per-regime cell type**,
so dispatch is monomorphized, not a `switch`.

Do NOT delete the enum on principle. A *derived* runtime tag (`static constexpr
Kind kind` on each policy, surfaced once at the erased boundary) is legitimate for
serialization, `repr`, and "user handed me data, choose a filtration at runtime."
Move the tag out of the hot path into traits; keep a thin derived tag at the edge.

### 2b. The packable-vs-fat taxonomy (user-defined is mandatorily fat)

A single universal "slim cell" is wrong: the encoding is per-regime, and some
cells cannot be packed at all.

| regime          | compact encoding                    | shared geometry          | packable? |
|-----------------|-------------------------------------|--------------------------|-----------|
| cubical         | `anchor<<3 \| face-bits` (the uid)  | `GridDomain`             | yes       |
| Freudenthal     | `anchor<<type_bits \| type`         | grid + (anchor,type) tbl | yes       |
| VR / alpha      | bit-packed sorted vertex tuple      | none combinatorial*      | yes**     |
| product         | `pair<A::uid, B::uid>`              | the two factor policies   | composes  |
| **user-defined**| **none** -- fat simplex stores ids  | --                       | **NO**    |

\* VR uses the distance matrix and alpha the circumradius only to compute filtration
*values* (stored on `CellWithValue`), not the combinatorial cell -- so VR and alpha
can share ONE packed-simplex policy; only value computation differs.
\** packable only while vertex-count and dim fit the packing width; fall back to fat.

Implications:
- The current single `Simplex` SPLITS into **packed-simplex** (VR/alpha, vertex
  ids bounded by the point count) and **fat-simplex** (user, arbitrary ids).
  User filtrations stay fat -- "fat" is just another policy, not a special case.
- **Products** compose: `ProductCell<A,B>` keeps `Uid = pair<A::Uid,B::Uid>` and
  its policy composes the two factor policies (already close to this today).
- The taxonomy is OPEN: a new regime is a new policy, not a new enum value.

### 2c. Shared geometry owned by the filtration

Move the geometry (the `GridDomain` for cube/Freudenthal; the (anchor,type)
tables; for VR the neighbor graph + dist source) into a store owned by the
`Filtration`, built once. Cells reference it only through the policy at build
time; they do not each carry a copy.

### 2d. Buffer-based policy (co)boundary; flat lookup for dense uids

Promote the benchmark's free functions to the live policy:
- cubical: `cube_boundary_col(uid, dom, lookup, out)` / `cube_coboundary_col(...)`
  (`bench_boundary.cpp:504,530`, already validated vs the transpose-relation).
- Freudenthal: `bd_table`/`cob_table` over `(anchor,type)`
  (`bench_boundary.cpp:359-380`), built once per filtration from
  `GridDomain::get_fr_displacements`.
- VR: bit-packed boundary is direct (drop one vertex); coboundary needs the
  neighbor graph (Ripser-style), deferred.
The policy writes row indices into a caller-provided per-dim buffer (cardinality
known for cube/Freudenthal), then sorts. Lookup `uid->sorted_id` is a **flat
direct-address array** where the uid space is dense (cube/Freudenthal: the 2.4x);
a hash map for VR/user.

### 2e. One Python type via erasure -- cheap, because the reduction core is already cell-agnostic

The load-bearing fact: **`Decomposition<Int,Real>` and `parallel_reduction`
operate on integer columns, not cells.** The only cell-typed code is (1) building
the (co)boundary matrix and (2) materializing cells for Python. So the natural
erasure seam is at the Filtration API: a virtual `FiltrationBase` whose
`reduce()`/`diagram()`/`cell(i)`/`boundary_matrix()` dispatch ONCE into the
concrete `Filtration<Policy>` (which builds the matrix with its monomorphized
policy at full speed); everything downstream is already single-typed
(`Decomposition` is one type per Int,Real). One virtual call per `reduce`, not
per cell.

Expose a single nanobind `Filtration` (and a single `Cell`, which is just the
fat-materialized view) backed by `FiltrationBase`. This is the "small wrapper" --
not surrender, but the correct seam -- and it is where the residual `kind()` tag
from 2a lives. Python users see one `Filtration` / one `Cell` type.

### 2f. Lazy "fat" cells for Python

`Filtration.cell(i)`/`.cells()`/iteration materialize a self-contained fat cell on
demand (uid + a copy/ref of the shared geometry), `keep_alive`-d to the
filtration; the slim internal cells are never exposed. The fat cell reproduces
today's surface (`.vertices`, `.boundary()`, `.coboundary()`, `.value`, `.dim`,
`.uid`, pickle, `__eq__`/`__hash__` on uid). For user-defined filtrations the
"fat" form is the stored form, so materialization is trivial.

### Synthesis

Templates/traits inside (no virtuals, no runtime kind in the column loop); one
erased wrapper at the Python boundary (one type, a runtime tag, slow path OK).

## 3. The Cell/policy concept (what each policy must provide)

Generic code requires of a cell: `dim()`, `get_uid()`/`set_uid()`,
`get_id()`/`set_id()`, `operator==` (uid-only after this change),
`operator<<`, the `Uid`/`UidHasher`/`UidSet`/`Boundary` typedefs, and `boundary()`
(`filtration.h:206,131,208`; `cell_with_value.h:43,54,59`); `coboundary()` is
optional, gated by `SupportsApparent<Cell>` (`apparent.h:34-39`). `CellWithValue`
adds value/sorted_id bookkeeping. After the refactor:
- `boundary()`/`coboundary()` are provided by the POLICY as
  `(uid, shared_geometry, lookup, out_buffer)`, used by the filtration builders,
  apparent detection (`apparent.h:222,230`), and the resolver
  (`decomposition.h:877,882`) -- the few hot call sites in scope of the geometry.
- Drop the domain from `Cube::operator==` (`cube.h:289`) and `std::hash<Cube>`
  (`cube.h:352`): uid-only is correct (all cells in one filtration share geometry)
  and required for a slim cell.

## 4. Blast radius (must update)

- Cube ctor sites needing a domain: `grid.h:202,244,315,363` (cube_filtration
  single/multi-threaded, `..._and_critical_indices`), `grid.h:530,564` (validity
  helpers), `oineus_cells.cpp:279-282,318-322` (Python ctors).
- Domain-using Cube methods -> policy: `boundary`, `coboundary`, `top_cofaces`,
  `anchor_vertex`, `get_vertices`, and `_cubes` wrappers
  (`cube.h:116,159,209,292,299,262-287`).
- Grid validity/critical-index helpers (`grid.h:508,529-531,540,563-565,571`).
- Pickle (`oineus_cells.cpp:298-303,336-341`): switch to uid-only + materialize
  fat on unpickle (or eager-materialize before pickle).
- `CellWithValue` SFINAE wrappers (`cell_with_value.h:107-128`).

## 5. Prior art (reuse vs avoid)

- `cubes_from_cadmus` (DESIGN REFERENCE): unified `Cube` + `Grid` with a
  `DataLocation { VERTEX, CELL }` enum separating `data_domain_` (values) from
  `computational_domain_` (cells), plus a GUDHI-cross-checked test suite
  (`test_cube.py`, `test_cubical_dgms.py`, parametrized over dim/values_on/
  dualize/threads). Inherit those tests. `DataLocation` (data on cells vs
  vertices) is orthogonal but rides along cleanly -- fold it into the cubical
  policy.
- `small_cells` (PATTERN ONLY): has a `SmallCube` taking the domain as a method
  parameter -- the domain-not-stored idea -- but as a dual class. Prefer the
  policy form over a second cell class.
- `benchmarks/bench_boundary.cpp` (this branch): the CURRENT proven spec for the
  packed policy bodies + the flat-vs-hash measurement. NOT superseded.
- Both branches are ~7 mo stale (base `9f318d1`, Nov 2025); use as reference,
  re-land fresh on current master.

## 6. Phasing (staged; each stage shippable + oracle-checked; hand-off-friendly)

Oracle every stage: diagrams identical to current, all kinds/dims/dualize, plus
`sanity_check` (R = D V). Benchmark each stage with `bench_boundary`/
`bench_apparent.py`. Each stage below is a natural STOP-AND-HAND-OFF point: the
tree is green and a useful increment has landed.

1. **Cubical policy + slim cube (the big one).** Introduce the cubical
   `CellPolicy` (geometry = `GridDomain`, buffer (co)boundary, flat lookup); slim
   `Cube` (uid-only; drop domain from `==`/hash); shared domain in the filtration;
   rewire the `filtration.h` builders + `apparent.h` detection + `decomposition.h`
   resolver to the policy; Python fat-cube materialization + pickle. KEEP
   `FiltrationKind` and the per-type Python classes during this stage (minimize
   churn). Lands the 6x build win + flips apparent to a net win. Fold in
   `DataLocation` here (reference has it + tests). The policy must be the SINGLE
   source of the (co)boundary column -- it replaces BOTH the filtration builder's
   construction AND `apparent_resolve_fn_` (today hand-synced duplicates that must
   stay consistent), and detection + build should share ONE `tf::Executor` (today
   two are constructed/torn-down per call). HAND-OFF POINT.

2. **Freudenthal policy.** `(anchor,type)` tables built once per filtration; a
   compact `(anchor,type)` cell. Now there are TWO packable policies -- enough to
   justify the erased wrapper in stage 3. Oracle: `test_fr_dgm_*`. HAND-OFF POINT.

3. **Erased boundary + dissolve `FiltrationKind`.** Introduce `FiltrationBase`
   (virtual reduce/diagram/cell/boundary-matrix); bind ONE Python `Filtration` +
   ONE `Cell`; move `kind()` to a derived tag on the wrapper; collapse the
   per-type Python classes. Verify no virtual leaks into the column loop. HAND-OFF
   POINT (the published-API shape is now in place).

4. **Split `Simplex`.** packed-simplex (VR/alpha, bit-packed vertices) vs
   fat-simplex (user, arbitrary ids, stored). User stays fat. Map existing
   `Filtration<Simplex>` user code to the fat policy. Oracle: VR + user-filtration
   tests. HAND-OFF POINT.

5. **VR coboundary + arena (measured).** VR coboundary via the neighbor graph
   (Ripser-style); optional per-dim arena allocation using known cardinality.
   CAUTION: the `column_arena` branch found a MONOLITHIC slab won prepare/teardown
   but regressed reduce 1.1-6.7x (allocator reuse defeated) -- adopt per-dim
   arenas only if measured to help end-to-end.

A session may stop after any stage. Stage 1 alone is a large, self-justifying win
(6x cubical build + apparent net win) and is a fine place to pass it on.

## 7. Honest costs / risks

- **Compile-time / instantiation blowup**: ~5 policies x grid dims x value/no-value.
  The erased wrapper tames the *Python* surface, not C++ build time.
- **Migrating the single `Simplex`** (stage 4) is the riskiest churn: every
  filtration constructor picks a policy; existing `Filtration<Simplex>` user code
  maps to the fat policy.
- **Erasure seam placement**: keep the virtual strictly at Filtration-level API;
  never let it reach the column loop (that stays templated/monomorphized).
- **Python lifetime/pickle**: fat cells `keep_alive` the geometry;
  standalone-pickled cells eager-materialize.
- **OINEUS_MAX_CUBE_DIM == 3** ceiling inherited (fine for current support).
- **Arena**: measure, do not assume "one big alloc is cheaper" (column_arena
  evidence).

## 8. Files

- New: `include/oineus/cell_policy.h` (per-regime policies; bodies from
  `bench_boundary.cpp`); maybe `include/oineus/filtration_base.h` (erased wrapper,
  stage 3).
- Modify: `cube.h` (slim cell; uid-only `==`/hash), `filtration.h` (shared
  geometry store; builders -> policy; flat `uid->sorted_id`), `grid.h` (ctor
  sites; validity/critical helpers; `DataLocation`), `cell_with_value.h` (SFINAE
  wrappers), `apparent.h` (detection -> policy), `decomposition.h` (resolver ->
  policy; keep core cell-agnostic), `bindings/python/oineus_cells.cpp` +
  `oineus_filtration.cpp` (fat-cell materialization; single erased `Filtration`/
  `Cell` in stage 3).
- Reference (don't edit): `benchmarks/bench_boundary.cpp`; branches
  `cubes_from_cadmus` (design + tests), `small_cells` (domain-as-param pattern).

## 9. Verification

- C++: existing `tests` (reduction, apparent) green; add a policy unit test
  (policy columns == current `boundary_matrix` / transpose).
- Python via ctest: cube/Freudenthal/VR diagrams unchanged across
  dim/values_on/dualize/threads (port `cubes_from_cadmus` `test_cube*`); full
  `ctest` green; user-defined-filtration tests still pass (fat path).
- Benchmark: `bench_boundary --only cube` shows the 6x; `bench_apparent.py` shows
  apparent ON beating OFF in wall time while keeping the -20..-31% peak-RSS win.

## 10. Current branch state (entry point)

On `claude/affectionate-lovelace-36e48d` (== `claude/youthful-leakey-2e1198`
fast-forwarded): the apparent direct-coboundary build + parallel detection is done
and green (52/52) -- a self-contained memory-win checkpoint (cohomology -31% /
homology -20% peak RSS) to commit before this refactor reworks the cube builders.
`benchmarks/bench_apparent.py` is the end-to-end apparent benchmark;
`benchmarks/bench_boundary.cpp` is the policy spec. Apparent firing is INHERENT
(cohomology fires 111K; "never fires" was wrong) and this refactor is what makes
recomputation cheap enough for apparent to be a net win.
