"""Reduction-side defaults for differentiable computations."""

from .. import _oineus


# Filtration kinds whose default reduction side is cohomology when the caller
# leaves dualize=None. Today this is only VR. When apparent-pair accelerated
# reductions land for scalar-field filtrations, add Freudenthal and Cubical here
# instead of updating every caller.
COHOMOLOGY_PREFERRED_KINDS = frozenset({
    _oineus.FiltrationKind.Vr,
})


def default_dualize_for_kind(kind) -> bool:
    return kind in COHOMOLOGY_PREFERRED_KINDS


def default_dualize_for_filtration(fil) -> bool:
    return default_dualize_for_kind(fil.kind)
