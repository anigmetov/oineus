"""Shared validation of alpha persistence output."""

import warnings

import numpy as np


def warn_invalid_alpha_diagrams(diagrams, *, stacklevel):
    """Warn about alpha points with death <= birth without changing them."""
    invalid_count = 0
    for diagram in diagrams:
        birth, death = diagram[:, 0], diagram[:, 1]
        invalid_count += int(np.count_nonzero(death <= birth))
    if invalid_count:
        warnings.warn(
            f"Alpha persistence output contains {invalid_count} point(s) "
            "with death <= birth. If built with exact=False, try "
            "exact=True. The diagram values have not been modified.",
            RuntimeWarning,
            stacklevel=stacklevel,
        )
