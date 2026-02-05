import functools as ft

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

    #@ft.wraps(_oineus.Filtration.max_dim)
    def max_dim(self):
        return self.under_fil.max_dim()

    #@ft.wraps(_oineus.Filtration.size)
    def size(self):
        return self.under_fil.size()

    #@ft.wraps(_oineus.Filtration.size_in_dimension)
    def size_in_dimension(self, dim):
        return self.under_fil.size(dim)

    #@ft.wraps(_oineus.Filtration.n_vertices)
    def n_vertices(self):
        return self.under_fil.n_vertices()

    #@ft.wraps(_oineus.Filtration.cells)
    def cells(self):
        return self.under_fil.cells()

    #@ft.wraps(_oineus.Filtration.get_id_by_sorted_id)
    def id_by_sorted_id(self, sorted_id):
        return self.under_fil.id_by_sorted_id(sorted_id)

    #@ft.wraps(_oineus.Filtration.get_sorted_id_by_id)
    def sorted_id_by_id(self, id):
        return self.under_fil.sorted_id_by_id(id)

    #@ft.wraps(_oineus.Filtration.get_cell)
    def cell(self, sorted_idx):
        return self.under_fil.cell(sorted_idx)

    #@ft.wraps(_oineus.Filtration.get_simplex)
    def simplex(self, sorted_idx):
        return self.under_fil.simplex(sorted_idx)

    #@ft.wraps(_oineus.Filtration.get_sorting_permutation)
    def sorting_permutation(self):
        return self.under_fil.sorting_permutation()

    #@ft.wraps(_oineus.Filtration.get_inv_sorting_permutation)
    def get_inv_sorting_permutation(self):
        return self.under_fil.inv_sorting_permutation()

    #@ft.wraps(_oineus.Filtration.cell_by_uid)
    def cell_by_uid(self, uid):
        return self.under_fil.cell_by_uid(uid)

    #@ft.wraps(_oineus.Filtration.boundary_matrix)
    def boundary_matrix(self, uid):
        return self.under_fil.boundary_matrix(uid)

    #@ft.wraps(_oineus.Filtration.simplex_value_by_sorted_id)
    def simplex_value_by_sorted_id(self, sorted_id):
        return self.under_fil.simplex_value_by_sorted_id(sorted_id)

    #@ft.wraps(_oineus.Filtration.simplex_value_by_vertices)
    def simplex_value_by_uid(self, uid):
        return self.under_fil.simplex_value_by_uid(uid)

    def cell_value_by_uid(self, uid):
        return self.under_fil.cell_value_by_uid(uid)

    #@ft.wraps(_oineus.Filtration.get_sorted_id_by_vertices)
    def sorted_id_by_uid(self, uid):
        return self.under_fil.sorted_id_by_uid(uid)

    #@ft.wraps(_oineus.Filtration.reset_ids_to_sorted_ids)
    def reset_ids_to_sorted_ids(self):
        self.under_fil.reset_ids_to_sorted_ids()
