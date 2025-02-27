#!python3

import oineus as oin

def test_ind_match():
    v0 = oin.Simplex([0], 0.2)
    v1 = oin.Simplex([1], 0.1)
    v2 = oin.Simplex([2], 0.3)

    # indices of vertices should be sorted
    e1 = oin.Simplex([0, 1], 0.9)
    e2 = oin.Simplex([0, 2], 0.5)
    e3 = oin.Simplex([1, 2], 0.8)

    t = oin.Simplex([0, 1, 2], 1.0)

    simplices = [v0, v1, v2, e1, e2, e3, t]

    negate = False
    n_threads = 1

    fil_codomain = oin.Filtration(simplices, negate, n_threads)

    v0 = oin.Simplex([0], 0.5)
    v1 = oin.Simplex([1], 0.2)
    v2 = oin.Simplex([2], 0.6)

    # indices of vertices should be sorted
    e1 = oin.Simplex([0, 1], 0.9)
    e2 = oin.Simplex([0, 2], 0.7)
    e3 = oin.Simplex([1, 2], 0.91)

    t = oin.Simplex([0, 1, 2], 1.2)

    simplices = [v0, v1, v2, e1, e2, e3, t]
    fil_domain = oin.Filtration(simplices, negate, n_threads)

    m = oin.get_induced_matching(fil_domain, fil_codomain)

    assert(len(m[0]) == 3)
    assert(len(m[1]) == 1)

if __name__ == "__main__":
    test_ind_match()
