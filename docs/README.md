# Documentation

This folder contains the [sources](./source) for the documentation of 
the provided functionality.
For build instructions see the [main README.md](../README.rst).  

## Folder structure
The [source](./source) sub-folder contains the sources for building the
documentation using [sphinx](https://www.sphinx-doc.org/):
- [_templates](./source/_templates): templates, esp. for sphinx `autosummary`
- [conf.py](./source/conf.py): sphinx config file
- Restructured text sources:
  + [index.rst](./source/index.rst): The entry point for the documentation.
  + [quickstart](./source/quickstart): Quick start instructions.
  + [userguide](./source/userguide): Function overview and further details for usage.
  + [apiref](./source/apiref): The entry point for the API reference;
    sphinx autosummary will auto-generate API documentation from docstrings and store
    the resulting sources in `apiref/generated/`.
- Build directory: the proposed sphinx settings will generate a HTML documentation tree
  stored under `build/`.
