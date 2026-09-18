"""Read canonical birth/death cycles and a bounding chain for one finite H1 pair."""

import numpy as np
import oineus as oin

angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
points = np.column_stack((np.cos(angles), np.sin(angles)))
filtration = oin.vr_filtration(points, max_dim=2, n_threads=1)
params = oin.ReductionParams(n_threads=4, compute_v=True, use_clearing=True)
decomposition = oin.reduce(filtration, params, dualize=False)
diagrams = decomposition.diagram(filtration, include_inf_points=False)
values = diagrams.in_dimension(1)
pairs = diagrams.index_diagram_in_dimension(1)
birth, death = map(int, pairs[np.argmax(values[:, 1] - values[:, 0])])

# Homology matrix indices are filtration sorted indices
decomposition.restore_elz_columns([birth, death], n_threads=4)
birth_cycle = decomposition.v_col(birth)
death_cycle = decomposition.r_col(death)
bounding_chain = decomposition.v_col(death)
print(f"Pair: ({birth}, {death})")
print(f"Birth cycle: {birth_cycle}")
print(f"Death cycle: {death_cycle}")
print(f"Bounding chain: {bounding_chain}")
