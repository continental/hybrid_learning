Framework for Fuzzy Logic Rule Definition
=========================================

The module :py:mod:`hybrid_learning.fuzzy_logic` defines a framework to

- define (logical) operations as functions that accept a grounding and provide the rule truth value
- define (fuzzy) logic rules as computational trees of operations
- parse rules to and from string representation
- define logics as collections of operation builders

For details have a look at the :ref:`apiref/index:API Reference`
and the :ref:`user guide <userguide/fuzzy_logic:Fuzzy Logic Operations>`.

.. contents::
    :depth: 2
    :local:
    :backlinks: top


Base Classes
------------

.. py:currentmodule:: hybrid_learning.fuzzy_logic.logic_base
.. autosummary::
    :nosignatures:

    ~merge_operation.Merge
    ~merge_operation.TorchOrNumpyOperation
    ~logic.Logic
    ~parsing.FormulaParser


Standard Fuzzy Logics
---------------------
The following (fuzzy) logics with standard connectives from Boolean or t-norms fuzzy logic
are defined in :py:mod:`hybrid_learning.fuzzy_logic.tnorm_connectives`.
Select one of them by key using :py:func:`hybrid_learning.fuzzy_logic.logic_by_name`.


.. py:currentmodule:: hybrid_learning.fuzzy_logic.tnorm_connectives
.. autosummary::
    :nosignatures:

    boolean.BooleanLogic
    product.ProductLogic
    lukasiewicz.LukasiewiczLogic
    goedel.GoedelLogic


Parsing
-------
See :py:class:`~hybrid_learning.fuzzy_logic.logic_base.parsing.FormulaParser`.


Standard Connectives
--------------------

.. py:currentmodule:: hybrid_learning.fuzzy_logic.tnorm_connectives
.. autosummary::
    :nosignatures:

    fuzzy_common.NOT
    boolean.AND
    boolean.OR
    boolean.IMPLIES
    boolean.IMPLIEDBY
    product.AND
    product.OR
    product.IMPLIES
    product.IMPLIEDBY
    lukasiewicz.AND
    lukasiewicz.OR
    lukasiewicz.IMPLIES
    lukasiewicz.IMPLIEDBY
    goedel.AND
    goedel.OR
    goedel.IMPLIES
    goedel.IMPLIEDBY


Standard Quantifiers
--------------------

.. automodsumm:: hybrid_learning.fuzzy_logic.quantifiers
    :skip: Any, Callable, Dict, Tuple, Optional, Iterable, Mapping, List, Sequence, Set, Union, Merge, TorchOperation, TorchOrNumpyOperation, AbstractQuantifier, AbstractTorchOrNumpyQuantifier
    :classes-only:
    :nosignatures:


Predicates
----------

Arithmetic Predicates
.....................

.. automodsumm:: hybrid_learning.fuzzy_logic.predicates.arithmetic
    :skip: Any, Callable, Dict, Tuple, Optional, Iterable, Mapping, List, Sequence, Set, Union, TorchOrNumpyOperation
    :classes-only:
    :nosignatures:

Custom Operations
.................

.. automodsumm:: hybrid_learning.fuzzy_logic.predicates.custom_ops
    :skip: Any, Callable, Dict, Tuple, Optional, Iterable, Mapping, List, Sequence, Set, Union, TorchOperation, AbstractFuzzyIntersect, Merge
    :classes-only:
    :nosignatures:
