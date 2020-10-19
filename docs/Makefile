# Minimal makefile for Sphinx documentation

# Usage:
# - Call from project root via `make -f "docs/Makefile"`
# - To add further sphinx arguments, append SPHINXOPTS="some arguments" to the make call
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= python -m sphinx
SOURCEDIR     = docs/source
BUILDDIR      = docs/build

# Put it first so that "make" without argument is like "make build".
build:
	@$(SPHINXBUILD) "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

help:
	@$(SPHINXBUILD) --help

.PHONY: help Makefile

clean:
	rm -rf "$(SOURCEDIR)/apiref/generated" "$(BUILDDIR)"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: build

