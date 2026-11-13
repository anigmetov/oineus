# -*- coding: utf-8 -*-
"""Sphinx configuration for Oineus documentation."""
from __future__ import annotations

import os
import re
import sys
from datetime import date

# Ensure MPLBACKEND is not forced to "Agg" in os.environ, otherwise spawned
# myst-nb ipykernel subprocesses will inherit "Agg" instead of activating
# matplotlib_inline and will silently omit inline PNG figures.
os.environ.pop("MPLBACKEND", None)

# Make the Python package importable for both Sphinx autodoc/autosummary and
# myst-nb's spawned Jupyter kernel subprocesses. Prefer the CMake build tree
# (`build/bindings/python` or `build_nanobind/bindings/python`) when present;
# otherwise fall back to an installed `oineus` package (as on ReadTheDocs).
_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
for _candidate in (
    os.path.join(_REPO, "build", "bindings", "python"),
    os.path.join(_REPO, "build_nanobind", "bindings", "python"),
):
    if os.path.isdir(_candidate):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
        _existing_pp = os.environ.get("PYTHONPATH", "")
        _pp_parts = [p for p in _existing_pp.split(os.pathsep) if p]
        if _candidate not in _pp_parts:
            os.environ["PYTHONPATH"] = os.pathsep.join([_candidate, *_pp_parts])
        break

# -- Project information ----------------------------------------------------

project = "Oineus"
author = "Arnur Nigmetov and contributors"
copyright = f"{date.today().year}, {author}"

# Map public Python facade classes (which dispatch in `__new__` to precision-
# or cell-specialized C++ classes) to their canonical `oineus._oineus` C++
# counterparts so Sphinx autodoc/autosummary documents all methods & properties.
_FACADE_TARGETS: dict[str, str] = {
    "Filtration": "_Filtration",
    "ProdFiltration": "_ProdFiltration",
    "Diagrams": "Diagrams",
    "DiagramPoint": "DiagramPoint",
    "Simplex": "Simplex",
    "ProdSimplex": "ProdSimplex",
    "Cube_1D": "Cube_1D",
    "Cube_2D": "Cube_2D",
    "Cube_3D": "Cube_3D",
    "Cube_4D": "Cube_4D",
    "Grid_1D": "Grid_1D",
    "Grid_2D": "Grid_2D",
    "Grid_3D": "Grid_3D",
    "Grid_4D": "Grid_4D",
    "TopologyOptimizer": "TopologyOptimizer",
    "TopologyOptimizerProd": "TopologyOptimizerProd",
    "TopologyOptimizerCube_1D": "TopologyOptimizerCube_1D",
    "TopologyOptimizerCube_2D": "TopologyOptimizerCube_2D",
    "TopologyOptimizerCube_3D": "TopologyOptimizerCube_3D",
    "TopologyOptimizerCube_4D": "TopologyOptimizerCube_4D",
    "KerImCokReduced": "KerImCokReduced",
    "KerImCokReducedProd": "KerImCokReducedProd",
    "IndicesValues": "IndicesValues",
    "IndicesValuesProd": "IndicesValuesProd",
}

_FACADE_INIT_SIGS: dict[str, str] = {}

_INTERNAL_FILTRATION_RE = re.compile(
    r"\boineus\._oineus(?:\._f32)?\.(?:"
    r"_Filtration"
    r"|_CubeFiltration_[1-4]D"
    r"|_FreudenthalFiltration_[1-4]D"
    r"|_PackedSimplexFiltration_(?:64|128)"
    r")\b"
)
_INTERNAL_PROD_FILTRATION_RE = re.compile(
    r"\boineus\._oineus(?:\._f32)?\._ProdFiltration\b"
)
_INTERNAL_MOD_RE = re.compile(r"\boineus\._oineus(?:\._f32)?\.")


def _clean_oineus_types(text: str) -> str:
    """Normalize internal nanobind C++ type paths to public `oineus.*` names."""
    text = _INTERNAL_PROD_FILTRATION_RE.sub("oineus.ProdFiltration", text)
    text = _INTERNAL_FILTRATION_RE.sub("oineus.Filtration", text)
    text = _INTERNAL_MOD_RE.sub("oineus.", text)
    return text


def _extract_first_init_sig(init_doc: str | None) -> str | None:
    """Extract `(arg1, ...)` from a nanobind `__init__` docstring."""
    if not init_doc:
        return None
    first_line = init_doc.strip().splitlines()[0].strip()
    if not first_line.startswith("__init__(self"):
        return None
    rest = first_line[len("__init__(self"):]
    rest = re.sub(r"\s*->\s*None\s*$", "", rest)
    if rest.startswith(", "):
        sig = "(" + rest[2:]
    elif rest.startswith(")"):
        sig = "()"
    else:
        return None
    return _clean_oineus_types(sig)


def _expose_facade_members() -> None:
    """Attach public C++ descriptors onto facade classes for Sphinx autodoc.

    Note: Never attach `__init__` onto `_Meta` facade classes, because Python
    invokes `cls.__init__` after `cls.__new__` whenever `isinstance(ret, cls)`
    is True (`_Meta.__instancecheck__`), which would double-initialize C++
    objects constructed via placement new.
    """
    import oineus as _oin

    ext = getattr(_oin, "_oineus", None)
    if ext is None:
        return
    for pub_name, ext_name in _FACADE_TARGETS.items():
        dst = getattr(_oin, pub_name, None)
        src = getattr(ext, ext_name, None)
        if dst is None or src is None or dst is src:
            continue
        init_sig = _extract_first_init_sig(
            getattr(getattr(src, "__init__", None), "__doc__", None)
        )
        if init_sig is not None:
            _FACADE_INIT_SIGS[f"oineus.{pub_name}"] = init_sig
        for attr_name, attr_val in src.__dict__.items():
            if not attr_name.startswith("_") and attr_name not in dst.__dict__:
                try:
                    setattr(dst, attr_name, attr_val)
                except (AttributeError, TypeError):
                    pass


try:
    import oineus  # noqa: F401

    release = getattr(oineus, "__version__", "0.0.0")
    _expose_facade_members()
except Exception:
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

# -- General configuration --------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
]

# MyST configuration: enable useful extensions for technical prose.
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# myst-nb: execute notebooks & executable Markdown tutorials at build time.
nb_execution_mode = "cache"
nb_execution_timeout = 300
nb_execution_allow_errors = False
nb_execution_raise_on_error = True
nb_merge_streams = True

# Autosummary generates stub files for each API member.
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

if os.environ.get("OINEUS_DOCS_OFFLINE") == "1":
    intersphinx_mapping = {}
else:
    intersphinx_mapping = {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable/", None),
        "scipy": ("https://docs.scipy.org/doc/scipy/", None),
        "matplotlib": ("https://matplotlib.org/stable/", None),
        "torch": ("https://docs.pytorch.org/docs/stable/", None),
    }

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "jupyter_execute",
    "**/.ipynb_checkpoints",
    # internal dev notes -- not part of the user-facing docs
    "u_strategy_findings.md",
]

master_doc = "index"
language = "en"
pygments_style = "sphinx"

# -- HTML output ------------------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = f"Oineus {release}"
html_show_sourcelink = False

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/anigmetov/oineus",
    "source_branch": "master",
    "source_directory": "docs/",
}

# -- Copybutton -------------------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: "
copybutton_prompt_is_regexp = True


# -- Autodoc hooks for clean nanobind / facade rendering --------------------

_OVERLOAD_ITEM_RE = re.compile(r"^(\d+)\.\s+(``.+``)\s*$")


def _deduplicate_overload_docstring(lines: list[str]) -> None:
    """Collapse identical normalized C++ template overloads in nanobind docs."""
    for i, line in enumerate(lines):
        if line.strip() == "Overloaded function.":
            header_idx = i
            break
    else:
        return

    items: list[tuple[str, list[str]]] = []
    cur_sig: str | None = None
    cur_body: list[str] = []

    for line in lines[header_idx + 1 :]:
        m = _OVERLOAD_ITEM_RE.match(line)
        if m:
            if cur_sig is not None:
                items.append((cur_sig, cur_body))
            cur_sig = m.group(2)
            cur_body = []
        elif cur_sig is not None:
            cur_body.append(line)

    if cur_sig is not None:
        items.append((cur_sig, cur_body))
    if not items:
        return

    unique_items: list[tuple[str, list[str]]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for sig, body in items:
        trimmed = list(body)
        while trimmed and not trimmed[0].strip():
            trimmed.pop(0)
        while trimmed and not trimmed[-1].strip():
            trimmed.pop()
        key = (sig, tuple(trimmed))
        if key not in seen:
            seen.add(key)
            unique_items.append((sig, trimmed))

    prefix = lines[:header_idx]
    while prefix and not prefix[-1].strip():
        prefix.pop()

    rebuilt = list(prefix)
    if rebuilt:
        rebuilt.append("")
    if len(unique_items) == 1:
        sig, body = unique_items[0]
        rebuilt.append(f"Signature: {sig}")
        if body:
            rebuilt.append("")
            rebuilt.extend(body)
    else:
        rebuilt.append("Overloaded function.")
        for idx, (sig, body) in enumerate(unique_items, start=1):
            rebuilt.append("")
            rebuilt.append(f"{idx}. {sig}")
            if body:
                rebuilt.append("")
                rebuilt.extend(body)

    lines[:] = rebuilt


def _on_process_signature(
    app, what, name, obj, options, signature, return_annotation
):
    if what == "class" and name in _FACADE_INIT_SIGS:
        if not signature or signature == "(*args, **kwargs)":
            signature = _FACADE_INIT_SIGS[name]
    if signature:
        signature = _clean_oineus_types(signature)
    if return_annotation:
        return_annotation = _clean_oineus_types(return_annotation)
    return signature, return_annotation


def _on_process_docstring(app, what, name, obj, options, lines):
    if what == "data":
        if name.endswith(".REAL_DTYPE") or name == "REAL_DTYPE":
            lines[:] = [
                "Default floating-point NumPy dtype (``numpy.dtype('float64')``) "
                "used by Oineus when inputs do not carry an explicit ``float32`` dtype."
            ]
            return
        if isinstance(obj, dict) and ".DEFAULT_" in name:
            lines[:] = [
                "Mutable default Matplotlib keyword-argument dictionary consulted by "
                "``oineus.vis`` plotting helpers (use the matching ``default_*_style()`` "
                "helper for a fresh copy)."
            ]
            return
        if isinstance(obj, str) and ".OKABE_ITO_" in name:
            lines[:] = [
                f"Hex color string (``{obj!r}``) from the colorblind-safe Okabe-Ito palette."
            ]
            return
        if isinstance(obj, str) and ".DEFAULT_" in name:
            lines[:] = [
                f"Default Matplotlib color string (``{obj!r}``) used by ``oineus.vis`` plotting helpers."
            ]
            return

    for idx, line in enumerate(lines):
        lines[idx] = _clean_oineus_types(line)
    _deduplicate_overload_docstring(lines)


def setup(app):
    app.connect("autodoc-process-signature", _on_process_signature)
    app.connect("autodoc-process-docstring", _on_process_docstring)
