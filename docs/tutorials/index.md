# Tutorials

Self-contained notebooks that you can run and edit. They are executed at
documentation build time, so the outputs in the rendered HTML always match
the code.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 📓 TDA for beginners
:link: 01_tda_for_beginners
:link-type: doc

Starts from "what is a filtration" and ends with a diagram-distance
comparison of two datasets. For anyone who has heard of persistent
homology but hasn't used it.
:::

:::{grid-item-card} 📈 Differentiable Wasserstein gradients
:link: 03_differentiable_wasserstein_gradients
:link-type: doc

Sample a clean and noisy circle, compute differentiable H1 diagrams, and
overlay Wasserstein and sliced Wasserstein diagram gradients.
:::
::::

For deeper, task-oriented coverage (filtrations, distances, KICR, Fréchet
means, differentiable diagrams, ...), see the {doc}`../user_guide` and the
individual {doc}`../topics/index` essays.

```{toctree}
:hidden:

01_tda_for_beginners
03_differentiable_wasserstein_gradients
```
