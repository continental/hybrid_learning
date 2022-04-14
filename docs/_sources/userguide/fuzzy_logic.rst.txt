Fuzzy Logic Operations
======================

One core feature of the provided libraries are the operations defined in
:py:mod:`hybrid_learning.fuzzy_logic`.
These allow to

- define logical formulas as truth functions
- define logics as a collection of such logical operations
- parse logical formulas to and from string infix representation for a given logic


.. contents::
    :depth: 2
    :local:
    :backlinks: top


Modelling Approach
------------------

A logic in mathematics is a collection of logical symbols that may be used
to formulate formulas. Symbols are variables and constants, functions, and
logical operations, i.e.
logical connectives (e.g. AND :math:`\wedge`, OR :math:`\vee`, NOT :math:`\neg`),
quantifiers (e.g. :math:`\exists`, :math:`\forall`), and
logical predicates (e.g. GreaterThan :math:`>`).

Basic Ideas
...........

The core data exchange format between operations is a dictionary (the ``annotations``).
This holds the groundings of variables and constants (i.e. mapping their names to values),
and outputs of previous operations, e.g. truth values.
This common storage for domain values and truth values enables easy reuse of prior operation outputs.
Operations (functions, logical operations) are then callables that accept such an annotations dictionary,
and add their output as further key-value pair.
Operations are defined by specifying the keys of values that should be used from the annotations dictionary.

>>> from hybrid_learning.fuzzy_logic.tnorm_connectives.boolean import AND
>>> grounding = {'a': True, 'b': False}
>>> operation = AND("a", "b")
>>> operation(grounding)
{'a': True, 'b': False, 'a&&b': 0}

Also, for efficient computation, a family of (truth) values may be stored as a vector under one key.
Operations will then take care of broadcasting.

>>> import numpy as np
>>> grounding = {'a': np.array([True, False]), 'b': True}
>>> AND("a", "b")(grounding)
{'a': array([ True, False]), 'b': True, 'a&&b': array([1, 0])}

An operation may also output a vector of values. Mathematically, this then is a family of operations
with each operation producing one of the vector entries as output.
This allows to handle a DNN that produces a mask tensor as output as a family of operations.


Base Classes
............

In this framework, logical formulas and theories are modelled
using the base classes
:py:class:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge` and
:py:class:`~hybrid_learning.fuzzy_logic.logic_base.logic.Logic`
as follows.

.. rubric:: Variables and Constants

Variables are represented by string keys.
They may be grounded (have a value assigned) by adding the key-value pair
to the annotations dictionary.
Constants are treated like grounded variables.


.. rubric:: Functions and Logical Operations

These are instances of
:py:class:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge`.
Provided base classes and implementations of typical logical operations are:

- *Predicates* (e.g. ``IsPartOfA``, ``GreaterThan``):
  These receive an arbitrary domain value, e.g. an image, and return a truth value.
  An example would be a DNN-based binary classifier (image in, truth value out).
  Some useful predicates are defined in :py:mod:`hybrid_learning.fuzzy_logic.predicates`.
- *(Fuzzy) logical connectives* (e.g. ``AND``, ``OR``, ``NOT``, ``IMPLIES``):
  These receive truth values (either Boolean or fuzzy), e.g. outputs from previous
  operations, and return a truth value.
  Base classes are defined in :py:mod:`hybrid_learning.fuzzy_logic.logic_base.connectives`.
  Standard connectives from Boolean and
  `t-norm fuzzy logics <https://en.wikipedia.org/wiki/T-norm_fuzzy_logics>`_.
  are defined in :py:mod:`hybrid_learning.fuzzy_logic.tnorm_connectives`.
- *Quantifiers* (e.g. ``EXISTS``, ``FORALL``):
  These reduce a domain of values, e.g. a set of image samples,
  to a single truth value, using a child predicate (the body formula)
  and a truth value reduction.
  Base classes and some standard quantifiers are defined in
  :py:mod:`hybrid_learning.fuzzy_logic.quantifiers`.


.. rubric:: Formulas

Fromulas are computational trees of operations.
As placeholder keys, Merge operations may use string keys of annotations values,
or directly specify a child operation that should produce the value to use.

>>> from hybrid_learning.fuzzy_logic.tnorm_connectives.boolean import AND, OR
>>> operation = AND("a", OR("b", "c"), keep_keys=['b||c'])
>>> operation({'a': True, 'b': False, 'c': True})
{'a': True, 'b': False, 'c': True, 'b||c': True, '(b||c)&&a': 1}
>>> operation({'a': True, 'b': False, 'c': True})
{'a': True, 'b': False, 'c': True, 'b||c': True, '(b||c)&&a': 1}

The children operations are applied to the grounding dictionary
before the parent operation, adding their outputs.
In case the output of the root operation is a truth value, such a computational tree
of operations constitutes a logical formula.

.. rubric:: Logics

A logic is here defined by a collection of operation builders for building formulas.
To allow parsing of infix notation, these are associated with an operator precedence
(their order in the collection).
The base class for this is
:py:class:`~hybrid_learning.fuzzy_logic.logic_base.logic.Logic`,
and some logics with standard connectives are defined in
:py:class:`~hybrid_learning.fuzzy_logic.tnorm_connectives`.
For building a standard logic including some predicates one can use
:py:func:`~hybrid_learning.experimentation.fuzzy_exp.fuzzy_exp_helpers.get_logic`.
To derive custom operation builders from
:py:class:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge` types,
one can use a :py:class:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.MergeBuilder`.


.. rubric:: Parsing

A logic allows parsing of infix notation string representations of formulas using a
:py:class:`~hybrid_learning.fuzzy_logic.logic_base.parsing.FormulaParser`.
Calling ``str`` on the resulting formula returns an equivalent parsable string
representation of the formula.


Variadic Operations
-------------------

At the heart of a merge operation lies the
:py:meth:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge.operation`
function that receives the domain or truth values referenced by the operation settings,
and returns the actual operation output.
The class method :py:meth:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge.variadic_`
will return a callable that accepts a list of values (or a dict of values, ignoring the keys),
and directly return the output of the operation.
For further details and examples see
:py:meth:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge.variadic_`.


Parsing Formulas
----------------

For parsing a string specification of a formula into an operation object,
one requires a list of operation builders, e.g. operation classes or
:py:class:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.MergeBuilder`
instances, that are ordered by precedence (higher index = higher precedence).
Instances of the default :py:class:`~hybrid_learning.fuzzy_logic.logic_base.parsing.FormulaParser`
can be called on an infix notation of a formula and will return the parsed object.
The symbols for representing the operations in the infix formula are assumed to be the
:py:class:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge.SYMB` attributes
of the operation builders.
The ``str`` to the formula object is the inverse to this default parsing operation.

>>> from hybrid_learning.fuzzy_logic import FormulaParser
>>> from hybrid_learning.fuzzy_logic.tnorm_connectives.boolean import AND, OR
>>> formula_spec: str = "a&&b||c&&d"
>>> FormulaParser([AND, OR])(formula_spec)
AND('a', OR('b', 'c'), 'd')
>>> FormulaParser([OR, AND])(formula_spec)
OR(AND('a', 'b'), AND('c', 'd'))
>>> str(FormulaParser([AND, OR])(formula_spec))
'(b||c)&&a&&d'


Defining Operator Builders and Logics
-------------------------------------

A logic is a sequence of operator builders for feeding a parser.
Besides acting as a collection, the :py:class:`~hybrid_learning.fuzzy_logic.logic_base.logic.Logic`
base class offers some convenience methods for easy parsing and operation building.
See the :py:class:`~hybrid_learning.fuzzy_logic.logic_base.logic.Logic` documentation
and the :py:meth:`~hybrid_learning.fuzzy_logic.logic_base.logic.Logic.parser` methods for details.

Operator builders must be a callable that returns an operation,
provide a ``SYMB`` attribute and a ``variadic_`` function, which is fulfilled,
e.g., by a :py:class:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge` subclass
or an :py:class:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.MergeBuilder` object.
Using :py:class:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.MergeBuilder`,
one can specify additional keyword arguments for the init call to a
:py:class:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge` class,
e.g., change the default ``SYMB``.
A merge builder can be created using the standard init, or using the
class method :py:meth:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge.with_`
of a :py:class:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge` class.


>>> from hybrid_learning.fuzzy_logic import MergeBuilder
>>> from hybrid_learning.fuzzy_logic.quantifiers import ALL
>>> ALL('mask', dim=(-2, -1), symb='AllPixels')
ALL('mask', dim=(-2, -1), symb='AllPixels')
>>> builder = MergeBuilder(ALL, symb='AllPixels', additional_args={'dim': (-2, -1)})
>>> builder('mask')
ALL('mask', dim=(-2, -1), symb='AllPixels')
>>> # Using functional interface integrated with Merge classes
>>> builder = ALL.with_(dim=(-2, -1)).symb_('AllPixels')
>>> builder('mask')
ALL('mask', dim=(-2, -1), symb='AllPixels')

The builders also support variadic instance generation:

>>> ALL.variadic_(dim=(-2, -1), symb='AllPixels')
ALL(_variadic=True, dim=(-2, -1), symb='AllPixels')
>>> ALL.with_(dim=(-2, -1)).symb_('AllPixels').variadic_()
ALL(_variadic=True, dim=(-2, -1), symb='AllPixels')

Such operator builders can now be used to enrich one of the standard logics,
as used in the standard logics builder
:py:func:`~hybrid_learning.experimentation.fuzzy_exp.fuzzy_exp_helpers.get_logic`:

>>> from hybrid_learning.fuzzy_logic import Logic, logic_by_name
>>> logic: Logic = logic_by_name('product')
>>> logic
ProductLogic()
>>> logic.operators
[...IMPLIEDBY..., ...IMPLIES..., ...AND..., ...OR..., ...NOT...]
>>> logic.insert(-1, ALL.with_(dim=(-2, -1)).symb_('AllPixels'))
>>> logic.operators
[..., ...OR..., MergeBuilder(hybrid_learning.fuzzy_logic.quantifiers.ALL, symb='AllPixels', additional_args={'dim': (-2, -1)}), ...NOT...]
>>> logic.parser()('AllPixels mask || b')
OR(ALL('mask', dim=(-2, -1), symb='AllPixels'), 'b')


