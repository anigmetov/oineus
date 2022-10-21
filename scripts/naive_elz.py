#!/usr/bin/env python3

from icecream import ic
import numpy as np


def is_zero(c):
    return np.all(c % 2 == 0)


def get_low(c):
    if is_zero(c):
        return -1
    else:
        return np.max(np.where(c % 2 == 1))


def sum_mod_two(a, b):
    return  (a + b) % 2


def sparse_to_set(a):
    result = []
    for i in range(a.shape[0]):
        if a[i] % 2 == 1:
            result.append(i)
    return set(result)


def is_reduced(a):
    lowest_ones = []
    for col_idx in range(a.shape[1]):
        if np.any(a[:, col_idx] % 2 == 1):
            lowest_ones.append(np.max(np.where(a[:, col_idx] % 2 == 1)))
    return len(lowest_ones) == len(set(lowest_ones))


def reduce_elz(d):
    n_rows, n_cols = d.shape
    pivots = np.zeros(n_rows, dtype=np.int64) - 1
    r, v = d.copy(), np.eye(n_rows, dtype=np.int64)
    for i in range(n_cols):
        while not is_zero(r[:, i]):
            low = get_low(r[:, i])
            piv = pivots[low]
            if piv == -1:
                pivots[low] = i
                assert(get_low(r[:, i]) == low)
                break
            else:
                assert(piv < i and get_low(r[:, i]) == get_low(r[:, piv]))
                r[:, i] = sum_mod_two(r[:, i], r[:, piv])
                v[:, i] = sum_mod_two(v[:, i], v[:, piv])
                assert(get_low(r[:, i]) < low)

    return r, v


def is_nested_or_disjoint(a, b):
    a, b = sparse_to_set(a), sparse_to_set(b)
    return a.isdisjoint(b) or a.issubset(b) or b.issubset(a)



if __name__ == "__main__":
    d = np.load("bm.npy")
    vals = np.load("vals.npy")
    n_rows, n_cols = d.shape
    ic(n_cols)
    r, v = reduce_elz(d)
    ic("elz done")
    r1 = d.dot(v)
    r1 %= 2
    ic(is_reduced(r))
    ic(np.all(r1 == r))
    for i in range(n_cols):
        if np.sum(d[:, i]) < 4:
            continue
        for j in range(i+1, n_cols):
            if not is_nested_or_disjoint(v[:, i], v[:, j]):
                print(f"Bad: {i}, {j}, {sparse_to_set(v[:,i])}, {sparse_to_set(v[:,j])}")

