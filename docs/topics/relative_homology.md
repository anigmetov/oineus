# Relative persistent homology

For a pair of filtrations $(K, L)$ where $L$ is a subcomplex of $K$, the
*relative* persistence diagram captures the cells of $K$ that are not in
$L$ -- equivalently, the homology of the quotient $K / L$.

## API

```{code-block} python
import oineus as oin

# fil_K = the big filtration; fil_L = the subcomplex you want to quotient by
dgms_rel = oin.compute_relative_diagrams(fil_K, fil_L, include_inf_points=True)
print(dgms_rel.in_dimension(0))
print(dgms_rel.in_dimension(1))
```

`compute_relative_diagrams` does the bookkeeping internally: it walks the
two filtrations, identifies which cells of $K$ are missing from $L$, and
returns the relative diagrams as a {py:class}`oineus.Diagrams` object
indexed by homology dimension.

## When you have two unrelated filtrations on the same vertex set

If your inputs are two independent filtrations $f_1, f_2 \colon K \to
\mathbb{R}$ on the same complex, build their pointwise minimum first:

```{code-block} python
fil_min = oin.min_filtration(fil_1, fil_2)
dgms = oin.compute_relative_diagrams(fil_1, fil_min)
```

`min_filtration` assigns each cell the smaller of its two values. The
relative diagram of $(f_1, \min(f_1, f_2))$ then captures the cells where
$f_2$ is smaller -- a common workhorse in interleaving constructions.

## Relative diagrams vs. KICR

Both relative persistence and {doc}`kicr` deal with pairs of filtrations,
but they answer different questions:

- **Relative diagrams** compute the persistence module of $H_*(K, L)$ --
  the homology of the quotient.
- **KICR** computes the kernel, image, and cokernel persistence modules
  of the map $H_*(L) \to H_*(K)$ -- three different modules tied to the
  long exact sequence of the pair.

A short exact sequence of complexes induces a long exact sequence of
persistence modules; relative diagrams give you one piece, KICR gives
you three. Pick the one that names the algebraic object you actually want.

## See also

- {doc}`kicr` -- the related kernel/image/cokernel construction.
- {doc}`mapping_cylinder` -- a different route to inclusion-style
  diagnostics when the map is not literally an inclusion.
- `tests/test_api_dgm.py`, `tests/test_kicr.py` -- runnable usage.
