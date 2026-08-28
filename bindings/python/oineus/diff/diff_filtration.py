class DiffFiltration:
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
