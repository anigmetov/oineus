#!/usr/bin/env python3


class Simplex:
    def __init__(self, vs):
        # vs must be iterable of tuples of ints (iterable of points)
        assert(type(list(vs)[0]) is tuple)
        assert(type(list(vs)[0][0]) is int)
        # all points are distinct
        assert(len(vs) == len(set(vs)))
        # all points have same dimension
        assert(len(set([len(v) for v in vs])) == 1)

        # for unit cube: all coords are 0, 1
        assert(set([x for v in vs for x in v]).issubset(set([0, 1])))

        self.vertices = tuple(sorted(vs))

        assert self.dim() <= self.ambient_dim()

    def dim(self):
        return len(self.vertices) - 1

    def ambient_dim(self):
        return len(self.vertices[0])

    def __lt__(self, other):
        return self.vertices < other.vertices

    def __hash__(self):
        return hash(self.vertices)

    def __eq__(self, other):
        return self.vertices == other.vertices

    def faces(self, dim):
        if dim < 0 or dim > self.dim():
            raise RuntimeError("Bad dimension")

        result = [Simplex(self.vertices)]

        for k in range(self.dim() - dim):
            result = frozenset().union([ sigma for tau in result for sigma in tau.facets()])

        return result

    def facets(self):
        result = []
        for i in range(len(self.vertices)):
            facet = Simplex([ vj for j, vj in enumerate(self.vertices) if j != i ])
            result.append(facet)
        return frozenset(result)


    def faces_with_vertex(self, dim, v):
        return sorted([f for f in self.faces(dim) if f.has_vertex(v)])


    def has_vertex(self, v):
        return v in self.vertices


    def last_vertex(self):
        return self.vertices[-1]


    def __repr__(self):
        return f"Simplex({self.vertices})"


    def vert_to_str_braces(self, v):
        s = "{ "
        for x in v[:-1]:
            s += f"{x}, "
        s += f"{v[-1]} }}"
        return s

    def to_str_braces(self):
        s = "{ "
        for v in self.vertices[:-1]:
            s += f"{self.vert_to_str_braces(v)}, "
        s += f"{self.vert_to_str_braces(self.vertices[-1])} }}"
        return s

    def to_str_braces_no_origin(self):
        vs = [ v for v in self.vertices if v != tuple(0 for _ in range(self.ambient_dim()) ) ]
        s = "{ "
        if vs:
            for v in vs[:-1]:
                s += f"{self.vert_to_str_braces(v)}, "
            s += f"{self.vert_to_str_braces(vs[-1])}"
        s += " }"

        return s


def zero_positions(p):
    return [ i for i, p_i in enumerate(p) if p_i == 0 ]


def flip_one_zero(p):
    return tuple([ tuple(1 if i == c else p_i for i, p_i in enumerate(p)) for c in zero_positions(p) ])


def helper(cands, cube_dim):
    if len(list(cands)[0]) != cube_dim + 1:
        return helper(tuple([ tuple(list(simplex) + [new_v]) for simplex in cands for new_v in flip_one_zero(simplex[-1])]), cube_dim)
    else:
        return cands



def top_simplices(cube_dim):
    orig_vertex = tuple(0 for _ in range(cube_dim))
    orig_simplex = tuple([orig_vertex])
    cand_seed = [orig_simplex]
    cand_seed = frozenset(tuple(cand_seed))
    return helper(cand_seed, cube_dim)


def top_fr_simplices(cube_dim):
    return [ Simplex(vs) for vs in top_simplices(cube_dim) ]


def faces_of_simplices(sigmas, d):
    result = []
    for sigma in sigmas:
        result += [tau for tau in sigma.faces(d)]
    return list(set(result))


def fr_simplices(cube_dim, dim):
    if dim < 0 or cube_dim < 0 or dim > cube_dim:
        raise RuntimeError("Bad dimension")

    tops = top_fr_simplices(cube_dim)

    return faces_of_simplices(tops, dim)


def fr_simplices_from_origin(cube_dim, dim):
    if dim < 0 or cube_dim < 0 or dim > cube_dim:
        raise RuntimeError("Bad dimension")

    origin = tuple(0 for _ in range(cube_dim))
    tops = top_fr_simplices(cube_dim)
    return [ sigma for sigma in faces_of_simplices(tops, dim) if sigma.has_vertex(origin) ]


if __name__ == "__main__":
    print("switch (cube_dim) {")
    for cube_dim in range(1, 4):
        print(f"""    case {cube_dim} :
        switch (dim) {{""")
        for dim in range(cube_dim + 1):
            sigmas = fr_simplices_from_origin(cube_dim, dim)
            print(f"            case {dim} : return {{")
            for sigma in sigmas[:-1]:
                print(f"                {sigma.to_str_braces()},")
            print(f"                {sigmas[-1].to_str_braces()} }};")
            print(f"                         break;")
        print('            default: throw std::runtime_error("Bad dimensions");')
        print('        }')
    print('    }')
