# Mapping cylinders

Map-induced persistence ({doc}`kicr`) is defined for an **inclusion**
$L \hookrightarrow K$. When the simplicial map you actually care about is
not an inclusion -- collapses, merges, glueings, two different filtrations
on overlapping vertex sets -- the standard fix is to replace the map with
its mapping cylinder, then run KICR on the resulting inclusion.

## The construction

Given two filtrations on overlapping vertex sets, the mapping cylinder is

$$ \text{Cyl}(f) = \big(L \times \{0\}\big) \;\cup\; \big(K \times \{1\}\big) $$

glued by the cylinder cells. In Oineus this is realized as a *product
filtration*: each cell in the cylinder is a product of a cell from one of
the input filtrations with an auxiliary vertex (`v_top` or `v_bottom`).

## API

```{code-block} python
import oineus as oin

# fil_K: the "big" filtration. fil_L: the inclusion source.
v_top    = oin.Simplex([n_K + n_L],     0.0)   # auxiliary vertex at t=0
v_bottom = oin.Simplex([n_K + n_L + 1], 0.0)   # auxiliary vertex at t=1

fil_cyl = oin.mapping_cylinder(fil_L, fil_K, v_top, v_bottom)
```

`mapping_cylinder(fil_domain, fil_codomain, v_domain, v_codomain)` returns
a {py:class}`oineus.ProdFiltration` of product cells. The two auxiliary
vertices need IDs that don't collide with the vertices already in either
filtration. Their values default to the filtration's $-\infty$ sentinel
(so they enter the filtration before any real cell); pass
`v_domain_value=...` / `v_codomain_value=...` to override.

## When to use this

For the common case of `KICR on a cylinder`, the wrapper
{py:func}`oineus.compute_ker_cok_reduction_cyl` builds the cylinder, the
inclusion, and the KICR object in one call:

```{code-block} python
kicr_prod = oin.compute_ker_cok_reduction_cyl(fil_2, fil_3)

kicr_prod.kernel_diagrams().in_dimension(0)
kicr_prod.cokernel_diagrams().in_dimension(0)
```

It returns a {py:class}`oineus.KerImCokReducedProd` (the product-cell
analogue of {py:class}`oineus.KerImCokReduced`; same accessors).

## Building blocks

If you need to assemble a cylinder by hand:

- {py:func}`oineus.mapping_cylinder` -- the cylinder itself.
- {py:func}`oineus.multiply_filtration` -- $\text{fil} \times \{\sigma\}$,
  i.e. multiply every cell in `fil` by the auxiliary cell. This is how
  you build the inclusion-as-subcomplex of the cylinder.
- {py:func}`oineus.min_filtration` -- the pointwise minimum of two
  filtrations on a shared vertex set. Useful when the "two filtrations"
  picture starts with two unrelated functions on the same complex.

## See also

- {doc}`kicr` -- once you have the cylinder, this is where you go.
- {doc}`relative_homology` -- a different construction for $(K, L)$
  pairs; algebraically different from KICR on a cylinder.
- `examples/python/example_kernel_cyl.py`, `tests/test_example_kicr_cyl.py`
  -- runnable demos.
