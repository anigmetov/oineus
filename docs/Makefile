# Minimal makefile for Oineus Sphinx documentation.
#
# Usage:
#   make html         -- one-shot build into _build/html
#   make strict       -- build HTML failing on any Sphinx warning (-W --keep-going)
#   make livehtml     -- live-reload build, serves on http://127.0.0.1:8000
#   make clean        -- remove _build, autosummary stubs, and cached outputs
#   make linkcheck    -- verify external links resolve

SPHINXOPTS      ?=
SPHINXBUILD     ?= sphinx-build
SPHINXAUTOBUILD ?= sphinx-autobuild
SOURCEDIR       = .
BUILDDIR        = _build
HOST            ?= 127.0.0.1
PORT            ?= 8000

.PHONY: help html strict livehtml clean linkcheck

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo
	@echo "Build finished. Open $(BUILDDIR)/html/index.html in a browser."

strict:
	@$(SPHINXBUILD) -W --keep-going -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo
	@echo "Strict build finished with zero warnings."

livehtml:
	@$(SPHINXAUTOBUILD) --host $(HOST) --port $(PORT) \
	    --ignore "*.swp" --ignore "*.pyc" --ignore "*~" \
	    --ignore "$(BUILDDIR)/*" \
	    "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

clean:
	@rm -rf "$(BUILDDIR)" jupyter_execute api/_autosummary
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	@echo "Cleaned build artifacts."

linkcheck:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS)
