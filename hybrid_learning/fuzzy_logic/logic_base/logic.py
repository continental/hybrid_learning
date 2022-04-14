#  Copyright (c) 2022 Continental Automotive GmbH
"""Base classes for defining a logic. For details see :py:class:`Logic`."""
import collections
import collections.abc
import inspect
from typing import Union, Sequence, List, Iterable, Callable

import numpy as np
import torch

from .merge_operation import Merge, _OpBuilder
from .parsing import FormulaParser


class Logic(collections.abc.MutableSequence):
    """Basic definition of a logic.
    A logic must have operator builders for some basic connectives
    (by default ``AND``, ``OR``, ``NOT``, see :py:attr:`DEFAULT_CONNECTIVES`)
    and holds a default operator builder precedence for parsing.
    New operator builders (e.g. for predicates or functions) can be added.
    The logic allows to iterate over its :py:attr:`operators` (the operator builders)
    using usual ``__getitem__`` notation.
    Operator builders must
    (cf. :py:class:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge`
    sub-classes and
    :py:class:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.MergeBuilder`
    for examples of valid operator builders):

    - be callable with string variable names, e.g. ``OR("a", "b")``, or child operators,
      e.g. ``OR(NOT("a"), "b")``, returning an operator,
    - return operators that are callable on dicts matching the variable names to instantiations,
    - provide a ``variadic_`` class method that returns a callable which
      accepts tensors/arrays/floats/Booleans and returns
      the result of the connective operation upon those inputs,
    - provide a ``SYMB`` attribute that specifies the (unique) parsing symbol
      within this logic.

    **Operator Access**

    Obtain primitive variadic instances (i.e. callables on logic values)
    of the operations via :py:meth:`logical_`.
    The basic connectives specified in :py:attr:`DEFAULT_CONNECTIVES` or
    (as overrides) during init can be accessed by their given common name,
    i.e. the used key. All other operations can be accessed by their parsing
    ``SYMB``. See also :py:meth:`op` for this access.

    **Sub-classing**

    To provide your own Logic, either create a :py:class:`Logic` instance
    and override all mandatory (i.e. ``None`` valued) :py:attr:`DEFAULT_CONNECTIVES`
    values, or define a sub-class that defines its own :py:attr:`DEFAULT_CONNECTIVES`.
    """
    DEFAULT_CONNECTIVES: collections.OrderedDict = collections.OrderedDict(
        IMPLIEDBY=False, IMPLIES=False, AND=None, OR=None, NOT=None, )
    """Default connectives by common names.
    Set ``None`` as default to force users to specify it during init.
    Set ``False`` to just mark the precedence position but not force
    users to specify an implementation during init.
    The order also determines the default order of precedence (higher index = higher precedence)."""

    def __init__(self, operators: Iterable[_OpBuilder] = None,
                 **default_overrides: _OpBuilder):
        """Init.

        :param operators: the list of operators this logic holds;
            extended by the :py:attr:`DEFAULT_CONNECTIVES` and the ``default_overrides``;
            accessible via their symbol
        :param default_overrides: overrides and additions in the form ``common_name=op_builder``
            for the :py:attr:`DEFAULT_CONNECTIVES`;
            added to the :py:attr:`operators` in the order given first by :py:attr:`DEFAULT_CONNECTIVES`
            then by additionally added connectives;
            later also accessible via their common name;
            set ``common_name=False`` to explicitly discard this standard operator from the logic.
        :raises: :py:class`TypeError` in case not all mandatory basic connectives
            (such with ``None`` entries in :py:attr:`DEFAULT_CONNECTIVES`) are overridden
        """
        self.operators: List[_OpBuilder] = list(operators) if operators is not None else []
        """Logical operators in their default order of precedence during parsing."""
        self._basic_connectives: collections.OrderedDict = self.DEFAULT_CONNECTIVES.copy()
        self._basic_connectives.update(default_overrides)
        """Non-deletable basic operators ordered by precedence."""

        # Add required operators as defaults up front (respecting precedence):
        for key, op in reversed(list(self._basic_connectives.items())):
            if op is None:
                raise TypeError("Please provide required operator {} as keyword argument"
                                .format(key, self.__class__))
            if op and op not in self.operators:
                self.operators.insert(0, op)
        self._basic_connectives = collections.OrderedDict({k: v for k, v in self._basic_connectives.items() if v})

        if not len(set(b.SYMB for b in self.operators)) == len(self.operators):
            raise ValueError("Got duplicate parsing symbols in operator builder list (class, symbol): {}"
                             .format([(b.__name__, b.SYMB) for b in self.operators]))

    def op(self, symb: str) -> _OpBuilder:
        """Get the operator builder for the given symbol.
        The symbol may be the common name, or, in second precedence, the parsing ``SYMB``.

        :raises: :py:class:`KeyError` in case the symbol is not found or not unique.
        """
        op = self._basic_connectives.get(symb, None)
        if op is not None:
            return op
        op_builders: List[_OpBuilder] = [op for op in self.operators if op.SYMB == symb]
        if len(op_builders) > 1:
            raise KeyError("Duplicate symbol {} in operator list! Found operator builders: {}"
                           .format(symb, op_builders))
        if len(op_builders) == 0:
            raise KeyError("No operator builder registered for symbol {}.".format(symb))
        return op_builders[0]

    def logical_(self, symb: str) -> Union[Merge, Callable[[Sequence[Union[float, np.ndarray, torch.Tensor, bool]]],
                                                           Union[float, np.ndarray, torch.Tensor]]]:
        """Get a variadic instance of the logical operation specified by ``symb``."""
        return self.op(symb).variadic_()

    def parser(self) -> 'FormulaParser':
        """Return a default parser for this logic."""
        return FormulaParser(self)

    def is_pure(self, op: Merge) -> bool:
        """Whether the formula ``op`` purely consists of operators built from this logic.
        See method
        :py:meth:`~hybrid_learning.fuzzy_logic.logic_base.parsing.FormulaParser.is_pure`
        of :py:class:`~hybrid_learning.fuzzy_logic.logic_base.parsing.FormulaParser`."""
        return self.parser().is_pure(op)

    def insert(self, precedence: int, op_builder: _OpBuilder) -> None:
        """Add an operator builder with precedence ``precedence``."""
        if any(b.SYMB == op_builder.SYMB for b in self.operators):
            raise ValueError("Cannot insert operator builder {} with symbol {}: Symbol occupied by operator {}"
                             .format(op_builder, op_builder.SYMB,
                                     {b.SYMB: b for b in self.operators}[op_builder.SYMB]))
        self.operators.insert(precedence, op_builder)

    def __setitem__(self, precedence: int, op_builder: _OpBuilder) -> None:
        if 0 < precedence >= len(self):
            raise IndexError("Index {} out of range for operator builder list of length {}"
                             .format(precedence, len(self)))
        self.insert(precedence, op_builder)

    def __getitem__(self, i: int):
        return self.operators[i]

    def __delitem__(self, i: int) -> None:
        if any(b == self.operators[i] for b in self._basic_connectives.values()):
            raise ValueError("Cannot remove required operator {} from logic.".format(self.operators[i]))

    def __len__(self):
        return len(self.operators)

    def __add__(self, other: Sequence[Merge]):
        other_basic_connectives = other._basic_connectives if isinstance(other, Logic) else {}
        return self.__class__([*self.operators, *other], **{**self._basic_connectives, **other_basic_connectives})

    def __radd__(self, other: Sequence[Merge]):
        other_basic_connectives = other._basic_connectives if isinstance(other, Logic) else {}
        return self.__class__([*other, *self.operators], **{**other_basic_connectives, **self._basic_connectives})

    def __repr__(self):
        # Operators list
        operators = self.operators
        if list(self.operators[:len(self._basic_connectives)]) == list(self._basic_connectives.values()):
            operators = operators[len(self._basic_connectives):]
        op_str = "[" + ", ".join([f"{s.__module__}.{s.__name__}" if inspect.isclass(s) else repr(s)
                                  for s in operators]) + "]" if len(operators) else ""
        # Default overrides
        setts = {}
        for common_name in self._basic_connectives:
            if self._basic_connectives[common_name] != self.DEFAULT_CONNECTIVES.get(common_name, None):
                setts[common_name] = self._basic_connectives[common_name]
        setts_str = ", ".join(f'{arg_n}={f"{s.__module__}.{s.__name__}" if inspect.isclass(s) else repr(s)}'
                              for arg_n, s in setts.items())
        return self.__class__.__name__ + "(" + ", ".join(s for s in [op_str, setts_str] if s) + ")"
