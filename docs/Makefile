# Minimal makefile for Oineus Sphinx documentation.
#
# Usage:
#   make html         -- one-shot build into _build/html
#   make livehtml     -- live-reload build, serves on http://127.0.0.1:8000
#   make figures      -- (re)generate static figures under _static/figures
#   make clean        -- remove _build and cached notebook outputs
#   make linkcheck    -- verify external links resolve

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAUTOBUILD ?= sphinx-autobuild
SOURCEDIR     = .
BUILDDIR      = _build
HOST          ?= 127.0.0.1
PORT          ?= 8000

.PHONY: help html livehtml clean figures linkcheck

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo
	@echo "Build finished. Open $(BUILDDIR)/html/index.html in a browser."

livehtml:
	@$(SPHINXAUTOBUILD) --host $(HOST) --port $(PORT) \
	    --ignore "*.swp" --ignore "*.pyc" --ignore "*~" \
	    --ignore "$(BUILDDIR)/*" \
	    "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

figures:
	@python _static/figures/generate_figures.py
	@echo "Figures written to _static/figures/"

clean:
	@rm -rf "$(BUILDDIR)" jupyter_execute api/_autosummary
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	@echo "Cleaned build artifacts."

linkcheck:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS)
