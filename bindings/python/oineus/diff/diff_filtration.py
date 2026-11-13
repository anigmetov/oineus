class DiffFiltration:
    """Autograd-aware wrapper pairing a C++ :class:`oineus.Filtration` with a differentiable values tensor.

    ``under_fil`` stores the concrete C++ filtration (defining cell combinatorics
    and sorted filtration order), while ``values`` is a 1D PyTorch tensor or JAX
    array of length ``under_fil.size()`` aligned to ``sorted_id`` order so that
    persistence diagrams and cochain content losses connect back to upstream
    parameters via standard reverse-mode autodiff.

    Attributes not found directly on ``DiffFiltration`` (such as ``size()``,
    ``max_dim``, ``negate``, ``cell()``, ``without_cells()``) are delegated
    transparently to ``under_fil``.

    Args:
        fil: Underlying C++ :class:`oineus.Filtration` instance.
        values: 1D differentiable tensor/array of filtration values aligned with
            the sorted cells of ``fil``.
    """

    def __init__(self, fil, values):
        self.under_fil = fil
        self.values = values

    def __len__(self):
        return len(self.under_fil)

    def __repr__(self):
        return f"DiffFil(under_fil={self.under_fil}, values={self.values})"

    def __iter__(self):
        return iter(self.under_fil)

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getattr__(self, name):
        # Fires only when normal attribute lookup fails on self, so
        # `under_fil` and `values` (set in __init__) still resolve directly.
        # Everything else is delegated to the wrapped filtration.
        #
        # During reconstruction the instance is created via __new__ without
        # __init__, so unknown delegated attributes arrive before under_fil
        # exists. Raise AttributeError rather than a bare KeyError.
        try:
            under_fil = self.__dict__["under_fil"]
        except KeyError:
            raise AttributeError(name)
        return getattr(under_fil, name)
