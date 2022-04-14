"""Basic functionality for parsing string representations of logical formulas.
See :py:class:`FormulaParser` for further information.
"""

#  Copyright (c) 2022 Continental Automotive GmbH
import collections
from typing import Union, Sequence, Dict, Any, Set, TYPE_CHECKING, Tuple, List, Iterable

import pyparsing

from .merge_operation import Merge, _OpBuilder

if TYPE_CHECKING:
    from .logic import Logic


class FormulaParser:
    """Base class to parse string formula specifiers in infix notation to merge operations.
    Furthermore, it assists in validating logical formulas and
    producing nice string representations (see e.g. :py:meth:`to_str`).
    This basic parser is an inverse to
    :py:meth:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge.to_infix_notation`
    of :py:class:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge`:

    >>> from hybrid_learning.fuzzy_logic import FormulaParser, boolean
    >>> logic = boolean.Logic()
    >>> parser = FormulaParser(logic)
    >>> formula_obj = parser("a || ~(b && c)")
    >>> formula_obj
    OR('a', NOT(AND('b', 'c')))
    >>> parser(formula_obj.to_str()) == formula_obj
    True

    Also, the :py:meth:`to_str` method matches
    :py:meth:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge.to_str`
    of :py:class:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge`:

    >>> f = "c || ~(b && a)"
    >>> parser.to_str(f)
    'c || ~(a && b)'
    >>> parser.to_str(f) == parser(f).to_str(precedence=logic, use_whitespace=True)
    True

    **Parsing**

    Calling a parser (same as calling :py:meth:`parse`) will parse a string to a logical operation.
    Parsing assumes infix notation of operators, and prefix for unary operators,
    with brackets allowed.
    A list of operator builders, e.g.
    :py:class:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge`
    sub-classes, defines the precedence
    of the operators (list must be ordered by increasing precedence).
    The operator builders each must provide a ``SYMB`` string attribute
    that specifies the symbol representing the operator.
    The implementation uses :py:func:`pyparsing.infixNotation`.

    **Functional Interface**

    The ``functional_*`` class methods provide a functional interface to parsing operations
    that will accept the logic (a list of allowed operators by precedence) as argument.
    """

    def __init__(self, logic: Union['Logic', Sequence[_OpBuilder]],
                 brackets: Tuple[str, str] = ('(', ')'),
                 allowed_chars: Union[str, List[str]] = pyparsing.alphanums + '-' + '_'):
        self.logic: Union['Logic', Sequence[_OpBuilder]] = logic
        """The list of operators that are allowed and used for parsing ordered by increasing precedence."""
        self.brackets: Tuple[str, str] = brackets
        """The symbols to interpret as left and right bracket in strings."""
        self.allowed_chars: Union[str, List[str]] = allowed_chars
        """The characters allowed within variable names in formulas.
        Brackets and characters occurring in operation keys are excluded automatically."""

    def __call__(self, specifier: str, **init_args) -> Union[str, 'Merge']:
        """Call parse on the specifier."""
        return self.parse(specifier, **init_args)

    def parse(self, specifier: str, **init_args
              ) -> Union[str, 'Merge']:
        """Parse a specifier to a merge operation.
        Will return the original string if it does not contain operation
        specifiers.
        Specifiers are any of the ``SYMB`` symbols of the classes
        listed in the operator builder list ``logic``.
        The sorting of the list specifies the operator precedence.

        :param specifier: the string specifier to parse
        :param init_args: any keyword arguments for init of the generated
            operations (parent and all children);
            ``out_key`` is only applied to the parent operation
        :return: the parsing outcome, a string (for a single variable) or a Merge operation
        :raises: :py:class:`pyparsing.ParseException` in case parsing fails
            (e.g. if a non-allowed char is used for variables)
        """
        return self.functional_parse(specifier, self.logic,
                                     brackets=self.brackets,
                                     allowed_chars=self.allowed_chars,
                                     **init_args)

    def to_str(self, specifier: str, for_filenames: bool = False, **kwargs):
        """Parse the specifier and return an easy-to-read string representation.
        If ``for_filenames`` is given, it the output string is suitable for filenames
        (no whitespace, pretty symbols)."""
        return self.functional_to_str(specifier, self.logic, for_filenames=for_filenames,
                                      **{'brackets': self.brackets, **kwargs})

    def is_pure(self, op: Merge) -> bool:
        """Whether the formula ``op`` purely consists of operators built from the logic."""
        return self.functional_is_pure(op, self.logic)

    def is_normal_form(self, parent_op: Merge) -> bool:
        """Checks whether the current formula respects the operator precedence.
        Uses introspection and assumes that all operators are objects of
        operator builders that are classes."""
        return self.functional_is_normal_form(parent_op, self.logic)

    def apply(self, specifier: Union[str, Merge], annotations: Dict[str, Any],
              **init_args) -> Dict[str, Any]:
        """Parse ``specifier`` and return its result on annotations
        if it's an operation.

        :param specifier: specifier for :py:meth:`parse`
        :param annotations: dictionary to which to apply the parsed operation
        :param init_args: further keyword arguments to init the parent
            operation while parsing
        """
        return self.functional_apply(specifier, annotations, self.logic, **init_args)

    @classmethod
    def _logic_to_symbol_class_map(cls, logic) -> collections.OrderedDict:
        """For a logic return a mapping ``{op_builder.SYMB: op_builder}``.

        :raises: :py:class:`ValueError` if a non-unique symbol is found
        """
        symbol_classes: collections.OrderedDict[str, _OpBuilder] = collections.OrderedDict(
            (op_cls.SYMB, op_cls) for op_cls in logic)
        if len(symbol_classes) != len(logic):
            raise ValueError("Found duplicate symbols in logic: {}".format(logic))
        return symbol_classes

    @classmethod
    def functional_is_pure(cls, op: Merge, logic: Sequence[_OpBuilder]) -> bool:
        """Whether the formula ``op`` purely consists of operators built from ``logic``.
        See :py:meth:`~FormulaParser.is_pure`."""
        return (op.__class__ in logic and
                all(cls.functional_is_pure(child, logic) for child in op.children))

    @classmethod
    def functional_is_normal_form(cls, parent_op: Merge, logic: Sequence[_OpBuilder]) -> bool:
        """Checks whether the current formula respects the logic operator precedence.
        See :py:meth:`~FormulaParser.is_normal_form`."""
        if any(not cls.functional_is_normal_form(child, logic) for child in parent_op.children):
            return False

        # Any child operator has higher precedence?
        children_cls: Set[type] = set((op.__class__ for op in parent_op.children))
        parent_cls: type = parent_op.__class__
        if parent_cls not in logic:
            raise ValueError("Cannot match operation of type {} to a builder: {}\nAllowed types: {}"
                             .format(parent_cls, parent_op, logic))
        if any(logic.index(child_cls) <= logic.index(parent_cls)
               for child_cls in children_cls):
            return False
        return True

    @classmethod
    def functional_to_str(cls, specifier: str, logic: Sequence[_OpBuilder],
                          brackets: Tuple[str, str] = None,
                          allowed_chars: Union[str, Iterable[str]] = None,
                          for_filenames: bool = False, **kwargs):
        """Parse the specifier and return a unique string representation.
        See :py:meth:`~FormulaParser.to_str`."""
        preset = dict(precedence=logic,
                      use_whitespace=not for_filenames,
                      use_pretty_op_symb=for_filenames,
                      brackets=brackets)
        return cls.functional_parse(specifier, logic=logic, brackets=brackets, allowed_chars=allowed_chars
                                    ).to_str(**{**preset, **kwargs})

    @classmethod
    def functional_apply(cls, specifier: Union[str, Merge], annotations: Dict[str, Any],
                         logic: Sequence[_OpBuilder],
                         brackets: Tuple[str, str] = None,
                         allowed_chars: Union[str, Iterable[str]] = None,
                         **init_args) -> Dict[str, Any]:
        """Parse ``specifier`` and return its result on annotations
        if it's an operation. See :py:meth:`~FormulaParser.apply`
        """
        return cls.functional_parse(specifier, logic=logic, brackets=brackets, allowed_chars=allowed_chars,
                                    **init_args)(annotations)

    @staticmethod
    def _pyparsing_op_def(op_builder: _OpBuilder, init_args: Dict[str, Any] = None,
                          op_symb: str = None, unary: bool = None):
        """For a ``unary`` or non-unary operation return a pyparsing parse action function.
        Defaults for ``op_symb`` and ``unary`` are taken from ``op_builder``
        attributes ``SYMB`` and ``ARITY``."""
        op_symb: str = getattr(op_builder, 'SYMB', None) if op_symb is None else op_symb
        assert op_symb is not None
        unary: bool = (getattr(op_builder, 'ARITY', -1) == 1) if unary is None else unary
        arity: int = 1 if unary else 2
        associativity = pyparsing.opAssoc.RIGHT if unary else pyparsing.opAssoc.LEFT

        def tokens_to_op(tokens):
            """Build operator instance from tokens."""
            try:
                if unary:
                    op = op_builder(*tokens[0][1:], **init_args)
                else:
                    op = op_builder(*[t for t in tokens[0] if t != op_symb], **init_args)
            except TypeError as t:
                t.args = (*t.args, f"\n{'unary ' if unary else ''}operation builder called incorrectly: {op_builder}")
                raise t
            return op

        return op_symb, arity, associativity, tokens_to_op

    @classmethod
    def functional_parse(cls, specifier: str, logic: Sequence[_OpBuilder],
                         brackets: Tuple[str, str] = None,
                         allowed_chars: Union[str, List[str]] = None,
                         **init_args
                         ) -> Union[str, 'Merge']:
        """Parse a specifier to a merge operation given a ``logic``.
        See :py:meth:`~FormulaParser.parse`."""
        brackets = brackets or ('(', ')')
        allowed_chars = allowed_chars or pyparsing.alphanums + '-' + '_'
        common_init_args = {k: v for k, v in init_args.items() if k != 'out_key'}
        out_key = init_args.get('out_key', None)
        symbol_classes: Dict[str, _OpBuilder] = \
            cls._logic_to_symbol_class_map(logic)

        # Define parser
        variable: pyparsing.ParserElement = pyparsing.Word(''.join(allowed_chars))
        op_list = [cls._pyparsing_op_def(op, common_init_args) for op in symbol_classes.values()]
        logic_expr: pyparsing.Forward = pyparsing.infixNotation(
            baseExpr=variable,
            opList=reversed(op_list),
            lpar=pyparsing.Suppress(brackets[0]), rpar=pyparsing.Suppress(brackets[1]),
        )
        # Parse
        try:
            parsed = logic_expr.parseString(specifier, parseAll=True)[0]
        except pyparsing.ParseException as p:
            p.msg += f"\nExpression to parse: '{str(specifier)}'"
            raise p

        # Adjust out_key if requested
        if out_key is not None and isinstance(parsed, Merge):
            parsed.out_key = out_key
        return parsed
