# -*- coding: utf-8 -*-
"""Sphinx configuration for Oineus documentation."""
from __future__ import annotations

import os
import sys
from datetime import date

# Make the Python package importable for autosummary. Prefer the build tree
# (bindings/python under build_nanobind) if it exists; otherwise fall back to
# an installed package.
_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
for _candidate in (
    os.path.join(_REPO, "build_nanobind", "bindings", "python"),
    os.path.join(_REPO, "build", "bindings", "python"),
):
    if os.path.isdir(_candidate):
        sys.path.insert(0, _candidate)
        break

# -- Project information ----------------------------------------------------

project = "Oineus"
author = "Arnur Nigmetov and contributors"
copyright = f"{date.today().year}, {author}"

# Pulled at build time from the package itself when available.
try:
    import oineus  # noqa: F401
    release = getattr(oineus, "__version__", "0.0.0")
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

# myst-nb: execute notebooks at build time, cache for speed.
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

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
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

# source_suffix is left to myst_nb to register (it handles both .md and
# .ipynb). Explicitly setting it here fights the extension.

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
