"""Parse fuzzy logic rules to truth functions in the form of dictionary transformations.
This module implements a general framework and DSL to specify and parse (fuzzy)
logical rules into a computational tree of
:py:class:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge` operations.
Besides the base classes to define rules, logical theories, and parse rules in a logic
to/from string, several standard operations are provided.

Logical formulas and theories are modelled
using the base classes
:py:class:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge` and
:py:class:`~hybrid_learning.fuzzy_logic.logic_base.logic.Logic`
as follows:

- (Logical) merge operations: These are instances of ``Merge`` operations,
  i.e. callables that accept a dictionary
  (the grounding = a map assigning values to variable names)
  and return the dictionary with the operation outputs added.
- Variables and constants: Variables are represented by string keys.
  They may be grounded (have a value assigned) by adding the key-value pair
  to the grounding dictionary.
  Constants are treated like grounded variables.
- Formula = tree of operations: Merge operations may depend directly on
  values given by keys (names of variables or constants) in the grounding dictionary,
  or they may have children operations on the output of which they depend.
  An example would be the operation ``AND("a", OR("b", "c"))` depending
  on the variable ``"a"`` and the child operation ``OR("b", "c")``.
  The children operations are applied to the grounding dictionary
  before the parent operation, adding their outputs.
  Such a computational tree of operations constitutes a logical formula.
- Typical logical operations are:

  - (Fuzzy) logical connectives: AND, OR, NOT, IMPLIES;
    These receive truth values (either Boolean or fuzzy) and return a truth value.
    Standard ones are those defined by the standard t-norm fuzzy logics.
  - Predicates: e.g. IsPartOfA, GreaterThan;
    These receive an arbitrary domain value, e.g. a pixel position, and return a truth value.
  - Quantifiers: EXISTS, FORALL;
    These reduce a domain of values, e.g. a set of image samples,
    to a single truth value, using a child predicate (the body formula)
    and a truth value reduction.

- Logic: A ``Logic`` is here defined by a collection of operation types (operation builders)
  for building formulas.
  To allow parsing of infix notation, these are associated with an operator precedence
  (their order in the collection).
- Parsing: A logic allows parsing of infix notation string representations of formulas using a
  :py:class:`~hybrid_learning.fuzzy_logic.logic_base.parsing.FormulaParser`.
  Calling ``str`` on the resulting formula returns an equivalent parsable string
  representation of the formula.


Note that naming is adopted from the case of Boolean mask operations
on segmentation mask annotations
(hence ``Merge`` operations accepting ``annotations`` dictionaries).
"""
#  Copyright (c) 2022 Continental Automotive GmbH

import typing as _typing

from . import logic_base
from . import predicates
from . import quantifiers
from .logic_base import FormulaParser, Logic, Merge, MergeBuilder
from .predicates import arithmetic
from .tnorm_connectives import boolean, goedel, lukasiewicz, product
from .tnorm_connectives.boolean import BooleanLogic
from .tnorm_connectives.goedel import GoedelLogic
from .tnorm_connectives.lukasiewicz import LukasiewiczLogic
from .tnorm_connectives.product import ProductLogic

_PARSERS: _typing.Dict[str, Logic.__class__] = dict(
    lukasiewicz=LukasiewiczLogic,
    goedel=GoedelLogic,
    product=ProductLogic,
    boolean=BooleanLogic,
)
"""Map from string keys to fuzzy logic handles as used in :py:func:`logic_by_name`."""


def logic_by_name(fuzzy_logic_key: str) -> Logic:
    """Select a fuzzy logic handle by string identifier.
    It can be used to parse formulas or access the logical operators
    of that logic."""
    if fuzzy_logic_key not in _PARSERS:
        raise KeyError("Unknown fuzzy_logic_key {}; choose from {}"
                       .format(fuzzy_logic_key, list(_PARSERS.keys())))
    return _PARSERS[fuzzy_logic_key]()
