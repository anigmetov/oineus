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
        return getattr(self.__dict__["under_fil"], name)
