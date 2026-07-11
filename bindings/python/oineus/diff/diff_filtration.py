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

    def __getattr__(self, name):
        # Fires only when normal attribute lookup fails on self, so
        # `under_fil` and `values` (set in __init__) still resolve directly.
        # Everything else is delegated to the wrapped filtration.
        #
        # During unpickling / copy.deepcopy the instance is created via __new__
        # without __init__, so special-method probes (__setstate__, __deepcopy__,
        # __reduce_ex__, ...) arrive before under_fil exists. Raise AttributeError
        # (not the bare KeyError from indexing __dict__) so those protocols fall
        # back to their defaults instead of crashing.
        try:
            under_fil = self.__dict__["under_fil"]
        except KeyError:
            raise AttributeError(name)
        return getattr(under_fil, name)
