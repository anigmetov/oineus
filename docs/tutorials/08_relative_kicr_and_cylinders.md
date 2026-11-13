---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 8. Relative Persistence, KICR & Mapping Cylinders

Standard persistent homology tracks a single filtered space $K$. Frequently, however, we have **two spaces connected by an inclusion $L \hookrightarrow K$ or a simplicial map $f: X \to Y$**—for instance:
- a boundary or region-of-interest subcomplex $L \subset K$,
- a subsampled landmark complex mapped into a dense ambient complex,
- a projection or embedding between two geometric representations of the same dataset.

From the induced map on homology $i_*: H_*(L) \to H_*(K)$ and the long exact sequence of the pair $(K, L)$, Oineus computes four rich persistence invariants:

1. **Relative persistence** $H_*(K, L)$ via {py:func}`oineus.compute_relative_diagrams`.
2. **Kernel, Image, and Cokernel (KICR) persistence** ($\ker i_*$, $\operatorname{im} i_*$, $\operatorname{coker} i_*$) via {py:func}`oineus.compute_kernel_image_cokernel_reduction` (Cohen-Steiner, Edelsbrunner, Harer & Morozov 2009).
3. **General simplicial maps via Mapping Cylinders** ({py:func}`oineus.mapping_cylinder`, {py:func}`oineus.compute_ker_cok_reduction_cyl`).
4. **Bauer–Lesnick Induced Matchings** via {py:func}`oineus.get_induced_matching`.

---

## 1. Relative Persistent Homology $H_*(K, L)$

Given a filtered complex $K$ and a subcomplex $L \subset K$, quotienting out chains in $L$ yields relative homology $H_*(K, L)$. For example, a solid disk $D^2$ has trivial $H_2(D^2) = 0$, whereas relative to its boundary circle $S^1 = \partial D^2$, $H_2(D^2, S^1) \cong \mathbb{Z}_2$!

Let's build a triangulated disk $K$ and its boundary loop $L$, and compute {py:func}`oineus.compute_relative_diagrams`:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import oineus as oin

# Wheel triangulation of a disk: center vertex 0 + outer rim vertices 1, 2, 3, 4
K = oin.Filtration([
    ([0], 0.0), ([1], 0.0), ([2], 0.0), ([3], 0.0), ([4], 0.0),
    # Outer rim edges enter at t = 1.0
    ([1, 2], 1.0), ([2, 3], 1.0), ([3, 4], 1.0), ([1, 4], 1.0),
    # Spokes enter at t = 2.0
    ([0, 1], 2.0), ([0, 2], 2.0), ([0, 3], 2.0), ([0, 4], 2.0),
    # Disk triangles fill in at t = 3.0
    ([0, 1, 2], 3.0), ([0, 2, 3], 3.0), ([0, 3, 4], 3.0), ([0, 1, 4], 3.0),
])

# Subcomplex L: just the outer boundary circle S^1
rim_verts = {1, 2, 3, 4}
L = K.subfiltration(lambda s: s.dim <= 1 and set(s.vertices).issubset(rim_verts))

rel_dgms = oin.compute_relative_diagrams(K, L, include_inf_points=True)
print("Ordinary H2(K):   ", oin.reduce(K).diagram(K)[2])
print("Relative H2(K, L):", rel_dgms[2])  # Born at t=3.0 when the disk fills in, never dies!
```

---

## 2. Kernel, Image & Cokernel Persistence (`KerImCokReduced`)

For an inclusion $i: L \hookrightarrow K$ at every scale $t$:

- **$\operatorname{im} i_*$ (`image_diagrams()`)**: homology classes **born in $L$** that remain nontrivial in $K$ (tracks features of $L$ up to the moment they become boundaries in $K$).
- **$\ker i_*$ (`kernel_diagrams()`)**: homology classes **born in $L$** that **become trivial in $K$** while still alive in $L$! Specifically, if a loop in $L$ is filled in by $K$ at $t_{\text{fill}}$ and only filled inside $L$ later at $t_L > t_{\text{fill}}$ (or never filled in $L$), it creates a $\ker i_*$ bar $[t_{\text{fill}}, t_L)$!
- **$\operatorname{coker} i_*$ (`cokernel_diagrams()`)**: homology classes of $K$ **not** in the image of $L$.

Let's see all three on our disk-and-rim pair $(K, L)$, where the outer rim loop is born at $t = 1.0$ in $L$, never dies inside $L$ ($t_L = +\infty$), and gets capped inside $K$ at $t = 3.0$:

```{code-cell} ipython3
kicr_params = oin.KICRParams()
kicr_params.kernel = True
kicr_params.image = True
kicr_params.cokernel = True
kicr_params.codomain = True

kicr = oin.compute_kernel_image_cokernel_reduction(K, L, params=kicr_params)

print("Domain H1(L):  ", kicr.domain_diagrams()[1])     # [1.0, inf)
print("Codomain H1(K):", kicr.codomain_diagrams()[1])   # [1.0, 3.0)
print("Image H1:      ", kicr.image_diagrams()[1])      # Born in L at 1.0, dies in K at 3.0 -> [1.0, 3.0)
print("Kernel H1:     ", kicr.kernel_diagrams()[1])     # Enters ker(H1(L)->H1(K)) at 3.0, lives in L forever -> [3.0, inf)
print("Cokernel H1:   ", kicr.cokernel_diagrams()[1])   # Empty (K's H1 loop already came from L!)
```

Look how cleanly the exact sequence fits together:
- From $t = 1.0$ to $t = 3.0$, the rim loop lives in **Image** (`[1.0, 3.0)`).
- At $t = 3.0$, the triangles of $K$ cap the rim loop while $L$ has no triangles, so at $t = 3.0$ the loop transitions into **Kernel** (`[3.0, inf)`)!

---

## 3. Non-Inclusion Maps via `mapping_cylinder`

What if you have two filtrations $F_{\text{dom}}$ and $F_{\text{cod}}$ on the same vertex set where $F_{\text{dom}}$ is **not** a subfiltration of $F_{\text{cod}}$ (for instance, Vietoris–Rips complexes before and after applying a dimensionality-reduction map $\pi: \mathbb{R}^3 \to \mathbb{R}^2$ that shrinks some pairwise distances and stretches others)?

1. First form $F_{\min} = \text{min\_filtration}(F_{\text{dom}}, F_{\text{cod}})$ so the vertex identity map $F_{\text{dom}} \to F_{\min}$ is filtration-preserving, or build the **simplicial mapping cylinder** via {py:func}`oineus.mapping_cylinder` (producing a {py:class}`oineus.ProdFiltration` homotopy equivalent to the codomain into which $F_{\text{dom}} \times \{v_0\}$ embeds as a true subcomplex!).
2. Or call the one-step helper {py:func}`oineus.compute_ker_cok_reduction_cyl` directly!

```{code-cell} ipython3
rng = np.random.default_rng(9)
theta = np.linspace(0, 2 * np.pi, 18, endpoint=False)
# A 3D circle tilted in Z, projected flat onto a 1D line in X (collapsing the H1 loop!)
circle_3d = np.column_stack([np.cos(theta), 0.15 * rng.standard_normal(18), np.sin(theta)])
line_2d = np.column_stack([circle_3d[:, 0], np.zeros(18)])

fil_3d = oin.vr_filtration(circle_3d, max_dim=2, max_diameter=2.5, packed=False)
fil_proj = oin.vr_filtration(line_2d, max_dim=2, max_diameter=2.5, packed=False)
fil_target = oin.min_filtration(fil_3d, fil_proj)

# Build explicit mapping cylinder and run KICR on ProdFiltration
v_top = oin.Simplex([100])
v_bot = oin.Simplex([101])
cyl = oin.mapping_cylinder(fil_3d, fil_target, v_top, v_bot)
dom_in_cyl = oin.multiply_filtration(fil_3d, v_top)

kicr_cyl = oin.compute_kernel_image_cokernel_reduction(cyl, dom_in_cyl)
ker_h1 = kicr_cyl.kernel_diagrams()[1]
print("Mapping cylinder size:", cyl.size())
print("Kernel H1 bars (3D loop collapsed by projection):", np.round(ker_h1, 4))

# Equivalently via the convenience wrapper compute_ker_cok_reduction_cyl:
kicr_shortcut = oin.compute_ker_cok_reduction_cyl(fil_proj, fil_3d)
print("Shortcut ker_cok_reduction_cyl H1 kernel bars:", len(kicr_shortcut.kernel_diagrams()[1]))
```

---

## 4. Induced Matchings (`get_induced_matching`)

When two filtrations $F_{\text{sub}}$ and $F_{\text{sup}}$ live on the **same underlying complex** $X$ with $F_{\text{sup}}(\sigma) \le F_{\text{sub}}(\sigma)$ for every cell $\sigma$ (for instance, comparing $f$ against $\min(f, g)$ built via {py:func}`oineus.min_filtration`), sublevel sets include at every threshold: $X_t^{F_{\text{sub}}} \subseteq X_t^{F_{\text{sup}}}$. {py:func}`oineus.get_induced_matching` computes the canonical **Bauer–Lesnick algebraic induced matching** between bars of $H_*(F_{\text{sub}})$ and bars of $H_*(F_{\text{sup}})$:

```{code-cell} ipython3
# Two filtrations on the SAME triangulated square complex with F_sup <= F_sub:
F_sub = oin.Filtration([
    ([0], 0.0), ([1], 0.0), ([2], 0.0), ([3], 0.0),
    ([0, 1], 1.0), ([1, 2], 1.0), ([2, 3], 1.2), ([0, 3], 1.2), ([0, 2], 1.5),
    ([0, 1, 2], 4.0), ([0, 2, 3], 3.5),
])
F_other = oin.Filtration([
    ([0], 0.0), ([1], 0.0), ([2], 0.0), ([3], 0.0),
    ([0, 1], 1.0), ([1, 2], 1.0), ([2, 3], 1.0), ([0, 3], 1.0), ([0, 2], 1.5),
    ([0, 1, 2], 3.0), ([0, 2, 3], 2.5),
])
F_sup = oin.min_filtration(F_sub, F_other)  # cell-wise min -> F_sup(sigma) <= F_sub(sigma)

matched = oin.get_induced_matching(F_sub, F_sup, dim=1)
print("Induced H1 matching pairs (bar in F_sub -> bar in F_sup):", matched)
```
