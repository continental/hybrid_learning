"""Sphinx configuration file."""

#  Copyright (c) 2022 Continental Automotive GmbH

# pylint: skip-file
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# Descriptive comments are partly auto-generated by the sphinx
# quickstart tool, or cited from the sphinx documentation.
#

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join('..', '..', 'hybrid_learning')))

# -- Project information -----------------------------------------------------

project = 'hybrid_learning'
copyright = '2022, Continental Automotive GmbH'
author = 'Gesina Schwalbe, Christian Wirth'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',  # recursive API doc generation
    'sphinx_automodapi.automodapi',  # easy module overviews
    'sphinx_automodapi.smart_resolver',  # avoid automodapi resolution errors
    # 'autodocsumm',  # summaries at beginning of autodoc of classes
    'autoclasstoc',  # summaries at beginning of autodoc of classes
    'sphinx.ext.napoleon',  # for pytorch docs
    'sphinx.ext.intersphinx',  # for referencing external docs
    'sphinx_rtd_theme',  # for mobile-friendly Read-the-Docs HTML theme
    'sphinx.ext.autosectionlabel',  # automatic anchors headings
    'sphinx.ext.viewcode',  # for links to source code
    'sphinx.ext.githubpages',  # add .nojekyll to build folder for github pages
]

# True to prefix each section label with the name of the document it is in,
# followed by a colon. For example, index:Introduction for a section called
# Introduction that appears in document index.rst.
# Useful for avoiding ambiguity when the same section heading appears in
# different documents.
autosectionlabel_prefix_document = True

# This config value contains the locations and names of other projects that
# should be linked to in this documentation.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3.6', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'PIL': ('https://pillow.readthedocs.io/en/stable', None),
    # tqdm objects.inv not available as of 2020-06-28
    # (see https://github.com/tqdm/tqdm/issues/705)
    # 'tqdm': ('https://tqdm.github.io/docs', None),
    'matplotlib': ('https://matplotlib.org', None),
    'pyparsing': ('https://pyparsing-docs.readthedocs.io/en/latest/', None),
}

# The number of seconds for timeout. The default is None,
# meaning do not timeout.
# timeout is not a time limit on the entire response download;
# rather, an exception is raised if the server has not issued
# a response for timeout seconds.
intersphinx_timeout = 10

# A list of (type, target) tuples (by default empty) that should be
# ignored when generating warnings in “nitpicky mode”. Note that type
# should include the domain name if present.
# Example entries would be ('py:func', 'int') or ('envvar', 'LD_LIBRARY_PATH').
nitpick_ignore = [
    ('py:class', 'pandas.core.series.Series'),  # actually pandas.Series
    ('py:class', 'pandas.core.frame.DataFrame'),  # actually pandas.DataFrame
    ('py:class', 'torch.nn.modules.module.Module'),  # actually torch.nn.Module
    ('py:class', 'torch.utils.data.dataloader.DataLoader'),
    # actually torch.utils.data.DataLoader
    ('py:class', 'torch.optim.optimizer.Optimizer'),
    # actually torch.optim.Optimizer
    ('py:class', 'torch.utils.hooks.RemovableHandle'),  # not in objects.inv?
    ('py:class', 'torch.utils.data.dataset.Dataset'),
    # actually torch.utils.data.Dataset
    ('py:class', 'torch.utils.data.dataset.Subset'),
    ('py:class', 'torch.nn.modules.activation.Sigmoid'),
    ('py:class', 'torch.nn.modules.padding.ZeroPad2d'),
    ('py:class', 'torch.nn.modules.container.ModuleDict'),
    ('py:class', 'torch.Size'),  # currently not documented
    ('py:class', 'torch.device'),  # currently not documented
    ('py:class', 'tqdm.std.tqdm'), ('py:class', 'tqdm.tqdm'),
    # currently there is no objects.inv
    ('py:class', 'matplotlib.axes._axes.Axes'),  # actually matplotlib.axes.Axes
    ('py:class', 'COCO'),  # no objects.inv available
    ('py:class', 'Tuple'), ('py:const', 'logging.INFO'),
    # not in python objects.inv
]
# An extended version of nitpick_ignore, which instead interprets the type and target
# strings as regular expressions. Note, that the regular expression must match the
# whole string (as if the ^ and $ markers were inserted).
#nitpick_ignore_regex = []

# Boolean indicating whether to scan all found documents for
# autosummary directives, and to generate stub pages for each.
# It is disabled by default.
# The new files will be placed in the directories specified in the
# :toctree: options of the directives.
autosummary_generate = True

# This value controls the docstrings inheritance.
# If set to True the docstring for classes or methods,
# if not explicitly set, is inherited form parents.
autodoc_inherit_docstrings = True

# This value selects if automatically documented members are sorted
# alphabetical (value 'alphabetical'), by member type (value 'groupwise') or
# by source order (value 'bysource').
# The default is alphabetical.
autodoc_member_order = 'groupwise'

# This value controls how to represents typehints.
# The setting takes the following values:
# 'signature' – Show typehints as its signature (default)
# 'description' – Show typehints as content of function or method
# 'none' – Do not show typehints
autodoc_typehints = 'description'

# This value is a list of autodoc directive flags that should be
# automatically applied to all autodoc directives. The supported flags are
# 'members', 'undoc-members', 'private-members', 'special-members',
# 'inherited-members', 'show-inheritance', 'ignore-module-all', and
# 'exclude-members'.
autodoc_default_options = {
    'show-inheritance': None,  # print base classes underneath class signature
    'undoc-members': None,  # include members without docstring
    'inherited-members': False,  # include inherited members from super-class
    'special-members': None,  # include members named like __special__
    'exclude-members': (
        "__dict__, __weakref__, __new__, "
        "__module__, __abstractmethods__, __annotations__, "),
    # 'autosummary': True,  # from autodocsumm: add member summaries
}

# The default list of sections to include in class TOCs, in the order they
# should appear in the documentation. The values in the list can be either
# strings or Section classes. Strings are the same values that can be
# provided to the section options of the autoclasstoc directive,
# and must refer to the name of a section class.
# The names of any custom sections that have been defined can be used as well.
from autoclasstoc import Section


class PublicNonSpecialMethods(Section):
    key = 'public-non-special-methods'
    title = "Public Methods:"

    def predicate(self, name: str, attr, meta) -> bool:
        """Only methods with names not starting with underscore."""
        return callable(attr) and not name.startswith('_')


class SpecialMethods(Section):
    key = 'special-methods'
    title = "Special Methods:"

    def predicate(self, name: str, attr, meta) -> bool:
        """Only special methods."""
        return callable(attr) and name.startswith('__') and name.endswith('__')


autoclasstoc_sections = [
    'public-attrs',
    'public-non-special-methods',
    'special-methods',
    # 'private-attrs',
    # 'private-methods',
    # 'inner-classes'
]

# This value selects what content will be inserted into the main body
# of an autoclass directive. The possible values are:
# - "class": Only the class’ docstring is inserted.
#   This is the default. You can still document
#   __init__ as a separate method using automethod or
#   the members option to autoclass.
# - "both": Both the class’ and the __init__ method’s docstring
#   are concatenated and inserted.
# - "init": Only the __init__ method’s docstring is inserted.
autoclass_content = 'class'

# This value selects how the signature will be displayed for the class
# defined by autoclass directive. The possible values are:
# "mixed" - Display the signature with the class name.
# "separated" - Display the signature as a method.
# The default is "mixed".
autodoc_class_signature = "mixed"

# This value controls the format of typehints. The setting takes the following values:
# 'fully-qualified' – Show the module name and its name of typehints
# 'short' – Suppress the leading module names of the typehints (ex. io.StringIO -> StringIO) (default)
autodoc_typehints_format = 'short'

# If true, suppress the module name of the python reference if it can be resolved. The default is False.
# (still experimental)
python_use_unqualified_type_names = True

# A boolean flag indicating whether to document classes and functions
# imported in modules. Default is False.
# autosummary_imported_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'  # 'nature'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
