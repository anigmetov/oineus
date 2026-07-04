# Mixup barcodes

A *mixup barcode* (Wagner, Arustamyan, Wheeler, Bubenik, [arXiv:2402.15058],
SoCG 2026) describes how one point cloud interacts with another: it enriches
the persistence barcode of a point cloud $A$ with the information of how much
the lifetime of each topological feature of $A$ shortens once the points of a
second cloud $B$ are mixed in.

[arXiv:2402.15058]: https://arxiv.org/abs/2402.15058

## Definition

Take the Vietoris–Rips filtrations $L = \mathrm{VR}(A)$ and
$K = \mathrm{VR}(A \cup B)$ over a common range of scales, and the image of
the induced map $H_k(L) \to H_k(K)$. For every bar $[b, d)$ of the barcode of
$H_k(L)$ there is a unique bar $[b, d')$ with the same birth (index) in the
image barcode — the induced matching of Bauer–Lesnick, computable with the
image-persistence reduction of Cohen-Steiner–Edelsbrunner–Harer–Morozov —
and since the extra boundaries of $K$ can only kill classes of $L$ earlier,
$b \le d' \le d$. The *mixup triple* $(b, d', d)$ splits the persistence bar
into

- the **image sub-bar** $[b, d')$ — the part of the feature's lifetime that
  survives the inclusion of $B$, and
- the **mixup sub-bar** $[d', d)$ — the *premature death*: the part destroyed
  by the boundaries that $B$ brings in.

In degree $0$ the mixup measures overlap (components of $A$ merging quicker
through $B$), in degree $1$ encirclement (loops of $A$ filled by $B$), in the
codimension-one degree surrounding (voids of $A$ filled by $B$).

Summary statistics, per degree, over the finite bars:

- $\mathrm{mixup}(t) = d - d'$ for a triple $t = (b, d', d)$;
- **total mixup** $= \sum_t (d - d')$, which equals the total persistence of
  $A$ minus the total image persistence;
- **mixup percentage** $\mathrm{mixup}_{\%}(t) = (d - d') / (d - b)$;
- **total / mean mixup percentage** — their sum / mean; the mean is
  scale-invariant and lies in $[0, 1]$.

Note the mixup sub-bars are *not* the kernel persistence bars — the matching
between the standard and the image barcode is what carries the information,
which is why a single coordinated computation is needed.

## Usage

```{code-block} python
import numpy as np
import oineus as oin

A = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])  # a square
B = np.array([[0.5, 0.5]])                                       # its center

mb = oin.mixup_barcodes(A, B, max_dim=1)

mb.in_dimension(1)             # (n, 3) triples (birth, image_death, death)
mb.image_sub_barcode(1)        # (n, 2) [b, d')
mb.mixup_sub_barcode(1)        # (n, 2) [d', d)
mb.persistence_barcode(1)      # (n, 2) [b, d) -- the ordinary barcode of A
mb.total_mixup(1)              # sqrt(2) - 1: the center fills the hole of A
mb.mean_mixup_percentage(1)    # 1.0: ... instantly, at the hole's birth
```

`mixup_barcodes(A, B, max_dim=1, max_diameter=None, n_threads=1)` builds both
filtrations with simplices up to dimension `max_dim + 1` and reports degrees
`0..max_dim`. The default `max_diameter` is the enclosing radius of $A$:
$\mathrm{VR}(A)$ is a cone above it, so every finite bar of $A$ — and hence
every image death — is captured, and only the essential
connected-component bar remains at infinity. Essential bars are kept in the
result (`mb.essential_in_dimension(dim)`) but excluded from the statistics,
as in the authors' reference implementation; so are zero-persistence bars.

Because the mixup of $(A, B')$ for $B' \subseteq B$ is at most the mixup of
$(A, B)$ (subsampling lemma), $B$ may be subsampled for speed at the price of
a weaker signal only.

For non-VR filtrations, `oin.mixup_barcodes_of_filtrations(K, L, ...)`
accepts any pair of filtrations with $L$ a subcomplex of $K$ (same values on
shared cells, matching cell encodings).

## Differentiable variant

`oineus.diff.mixup_barcodes(A, B, ...)` takes torch tensors or jax arrays and
returns triples and statistics that are differentiable with respect to the
coordinates of *both* $A$ and $B$: each triple is a pure gather of the
(birth, image death, death) cells' values from the differentiable VR
filtration of the union, so gradients flow through the critical-edge
distances exactly as in {py:func}`oineus.diff.kicr_diagrams`. All four
statistics (total persistence, total mixup, total and mean mixup percentage)
are smooth in the filtration values, hence differentiable wherever the
pairing is locally constant — the generic situation. Only finite triples are
represented; the statistics agree in value with the non-differentiable
variant, which also computes them over the finite bars.

```{code-block} python
import torch
import oineus.diff

A = torch.tensor(..., requires_grad=True, dtype=torch.float64)
B = torch.tensor(..., requires_grad=True, dtype=torch.float64)

dmb = oineus.diff.mixup_barcodes(A, B, max_dim=1)
loss = dmb.total_mixup(1)      # e.g. push B out of the holes of A
loss.backward()                # gradients land on A and B
```
