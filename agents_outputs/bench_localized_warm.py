"""Quick check: localized warm update (dynamic move-schedule) vs from-scratch
parallel pairing-only reduce, for a SMALL local change on a large complex."""
import time

import numpy as np

import oineus as oin


def lower_star(data):
    return oin.freudenthal_filtration(np.ascontiguousarray(data), max_dim=2)


def rp():
    p = oin.ReductionParams(); p.compute_v = True; p.n_threads = 1; p.clearing_opt = False
    return p


def rp_full(nt):
    p = oin.ReductionParams(); p.compute_v = False; p.compute_u = False
    p.clearing_opt = True; p.n_threads = nt
    return p


def nto(fo, fn):
    o = {c.uid: c.sorted_id for c in fo.cells()}
    return [o[c.uid] for c in fn.cells()]


def pairing(r):
    return frozenset((max(c), i) for i, c in enumerate(r) if c)


for side in [128, 256]:
    rng = np.random.default_rng(side)
    data = rng.random((side, side))
    f0 = lower_star(data)
    d = oin.Decomposition(f0, False)
    d.reduce(rp())
    n = len(d.d_data)
    t = time.perf_counter(); d.make_dynamic(8); t_make = time.perf_counter() - t

    # tiny change: nudge ONE pixel by a small delta -> very small displacement
    r, c = rng.integers(0, side, 2)
    data[r, c] += 0.0003 * rng.standard_normal()
    f1 = lower_star(data)
    perm = nto(f0, f1)
    moved = sum(1 for k, o in enumerate(perm) if k != o)

    st = oin.DecompositionManipStats()
    t = time.perf_counter(); d.apply_move_schedule(perm, st); t_warm = time.perf_counter() - t

    times_full = {}
    dd = None
    for k in [1, 8, 16]:
        dd = oin.Decomposition(f1, False); pp = rp_full(k); dd.reduce(pp)
        times_full[k] = pp.elapsed
    ok = pairing(d.r_data) == pairing(dd.r_data)

    print(f"side={side} n={n} | moved={moved} moves={st.n_moves} transp={st.n_transpositions} "
          f"col_scans={st.n_columns_scanned}")
    print(f"   warm total={1000*t_warm:.2f}ms  (schedule_build={1000*st.elapsed_schedule_build:.2f}ms, "
          f"transpositions={1000*st.elapsed_transpose:.2f}ms)  "
          f"full@1={1000*times_full[1]:.2f} @8={1000*times_full[8]:.2f} @16={1000*times_full[16]:.2f}ms  ok={ok}")
