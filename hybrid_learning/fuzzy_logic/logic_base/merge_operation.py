#  Copyright (c) 2022 Continental Automotive GmbH
"""Base classes and helper functions for defining logical operations.
Main base classes are:

- :py:class:`Merge`: Base class for operating on arrays/tensors and booleans,
  and for building computational trees of such operations
- :py:class:`TorchOrNumpyOperation`: Base :py:class:`Merge` class for operating
  on numpy or pytorch tensors
- :py:class:`MergeBuilder`: A convenience builder class that allows to
  define custom constructors for a merge class;
  of interest for easily setting defaults

The logical merging operations derived from :py:class:`Merge` allow for
concatenation of operations. Using them, any operation involving
intersection (``AND``),
union (``OR``), and
inversion (``NOT``)
of masks can be modelled. Scalar values in this case are
treated as all-same-valued masks when mixed with mask tensors.
For further information have a look at the :py:class:`Merge` documentation.
"""

import abc
import inspect
from typing import Type, Union, Sequence, Optional, Any, Dict, Callable, Tuple, List, Mapping, Iterable, Set, MutableMapping, \
    Collection, Literal

import numpy as np
import torch

from ...datasets.transforms.dict_transforms import DictTransform
from ...datasets.transforms.image_transforms import ToTensor


def stack_tensors(*inputs: torch.Tensor) -> torch.Tensor:
    """Broadcast and stack the inputs in dim 0 to enable pixel-wise operations."""
    return torch.stack(torch.broadcast_tensors(*inputs)
                       if len(inputs) else torch.tensor(inputs))


class Merge(DictTransform, abc.ABC):
    """Base class for operations and operation trees on dictionary inputs.
    Merge the masks or scalars values of the dict input according to the
    operation (tree) definition and store them under the specified output key.
    The merge operation may recursively have child merge operations as
    :py:attr:`in_keys`, which are evaluated on the given dictionary before
    the parent is.

    **Operation**

    The actual operation is hidden in the :py:meth:`apply_to` method:
    It is given a dictionary of annotations of the form ``{ID: value}`` and
    will return the dict with the merged mask added as ``{out_key: value}``.
    The intermediate outputs of child operations are by default only used
    for caching (see :py:attr:`cache_duplicates`) and then discarded.
    To include them into the final output, use the ``keep_keys`` argument
    to the operation call (see :py:meth:`apply_to`).
    The benefit of caching duplicates is that results may be reused
    amongst different operations.

    **Initialization**

    During init, all non-keyword arguments serve as :py:attr:`in_keys`.
    These are used when the merge operation is called on a dict:
    The dict must provide items with these :py:attr:`in_keys`, and the values of
    these items are fed to the actual operation.
    Settings must be given as keyword arguments.
    To set default keyword arguments for the init call, use a :py:class:`MergeBuilder`.
    See :py:meth:`with_` for creating a :py:class:`MergeBuilder` from a :py:class:`Merge`
    class.

    **Example: Boolean Logic**

    To get all heads, noses, and mouths (binary masks)
    of real persons (binary masks)
    in bathrooms (boolean labels),
    call:

    >>> from hybrid_learning.fuzzy_logic.tnorm_connectives.boolean import AND, OR, NOT, BooleanLogic
    >>> op = AND("person", OR("head", "nose", "mouth"), NOT("bathroom"))
    >>> op == BooleanLogic().parser()("person&&head||nose||mouth&&~bathroom")
    True
    >>> # Example with 1 pixel of a person mouth not in a bathroom:
    >>> result = op({"person": 1, "head": 0, "nose": 0, "mouth": 1, "bathroom": False})
    >>> result[op.out_key] == 1
    True
    >>> result
    {'person': 1, 'head': 0, 'nose': 0, 'mouth': 1, 'bathroom': False,
    '(head||mouth||nose)&&(~bathroom)&&person': 1}

    To also inspect the intermediate output, use the ``keep_keys`` option:

    >>> op({"person": 1, "head": 0, "nose": 0, "mouth": 1, "bathroom": False},
    ...    keep_keys=op.all_out_keys)
    {'person': 1, 'head': 0, 'nose': 0, 'mouth': 1, 'bathroom': False,
    'head||mouth||nose': True,
    '~bathroom': True,
    '(head||mouth||nose)&&(~bathroom)&&person': 1}

    Note that the input dict must feature all ``in_keys`` of operations in the formula.

    **Subclassing**

    To implement your own merge operation

    - implement the :py:meth:`operation`
    - specify your own :py:attr:`SYMB` (this must be unique within the logic you are using)
    - extend the :py:attr:`settings` and :py:attr:`setting_defaults` properties by new items if necessary

    **Format and String Parsing**

    The (recursive) merge operation best is specified in conjunctive normal
    form for uniqueness (thus comparability) and parsing compatibility.
    This is the form

    .. code::

        AND(..., [NOT(...), ...], [OR(..., [NOT(..), ...])])

    (see https://en.wikipedia.org/wiki/Conjunctive_normal_form).
    Exemplary available operations are the Boolean ones
    :py:class:`~hybrid_learning.fuzzy_logic.tnorm_connectives.boolean.AND` (intersection),
    :py:class:`~hybrid_learning.fuzzy_logic.tnorm_connectives.boolean.OR` (union), and
    :py:class:`~hybrid_learning.fuzzy_logic.tnorm_connectives.boolean.NOT` (inversion)
    that all operate pixel-wise.
    Boolean classification labels are treated as all-one-masks.

    One can use a :py:class:`~hybrid_learning.fuzzy_logic.logic_base.parsing.FormulaParser`
    implementation to parse a string representation of an operation tree.
    Check the corresponding implementation for the operator precedence and examples.
    For parsing, used connectors of the logic must be encoded by their :py:attr:`SYMB`
    attribute, e.g. for the examples above:

    - ``AND``: a&&b
    - ``OR``: a||b
    - ``NOT`` (unary operation): ~a
    """
    SYMB: str = None
    """The string symbol of this class (override for sub-classes)."""
    ARITY: int = -1
    """The arity of the operation. -1 means unlimited number of arguments possible."""
    IS_COMMUTATIVE: bool = False
    """Whether instances are equivalent to ones with permuted :py:attr:`in_keys`."""

    @classmethod
    def variadic_(cls, **kwargs):
        """Return an instance with variadic __call__.
        It's __call__ will accept maps or iterables of arbitrary length
        (for :py:attr:`ARITY` = -1) respectively of length matching :py:attr:`ARITY`.
        All values/elements are passed through to the :py:meth:`operation`,
        and the plain output of :py:meth:`operation` is returned
        (see also :py:meth:`variadic_apply_to`).
        Use this e.g. to wrap the :py:meth:`operation` into an object in a
        multiprocessing-safe manner.
        Example:

        >>> from hybrid_learning.fuzzy_logic.tnorm_connectives.boolean import AND
        >>> primitive_and = AND.variadic_()
        >>> primitive_and({"a": 1, "b": True})
        1
        >>> primitive_and([1, True, 1.])
        1

        No :py:attr:`in_keys` may be given, and :py:attr:`out_key` is obsolete.
        The returned instance may not be used as child element of a formula."""
        return cls(**{**dict(_variadic=True), **kwargs})

    @property
    def is_variadic(self) -> bool:
        """Whether the instance is variadic.
        See :py:meth:`variadic_`."""
        return self._variadic

    def __init__(self, *in_keys: Union[str, 'Merge'], out_key: str = None,
                 overwrite: bool = True, skip_none: bool = True,
                 replace_none=None, symb: str = None, cache_duplicates: bool = True,
                 keep_keys: Collection[str] = None,
                 _variadic: bool = False):
        """Init.

        Hand over input keys either as str or as a Merge operation of str.

        :param in_keys: sequence of either
            :py:class:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge` operation
            instances or strings with placeholders for the input keys
        :param out_key: key for the output of this operation; used to
            init :py:attr:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge.out_key`
        :param overwrite: on call, whether to overwrite the value at
            :py:attr:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge.out_key`
            in the given dict if the key already exists;
            raise if key exists and ``overwrite`` is true; saved in
            :py:attr:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge.overwrite`.
        :param replace_none: if not ``None``, the value to replace any
            ``None`` values with; see
            :py:attr:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge.replace_none`
        :param symb: override the
            :py:attr:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge.SYMB`
            for this instance
        :param keep_keys: intermediate output keys to add to call output;
            see :py:attr:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge.keep_keys`
        :param cache_duplicates: whether outputs of children with identical
            keys should be cached and reused; see
            :py:attr:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge.cache_duplicates`
        :param _variadic: the preferred way to specify this argument is
            :py:meth:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge.variadic_`;
            see there for details
        """
        # region Value checks
        if not _variadic and len(in_keys) <= 0:
            raise TypeError("Got empty list of in_keys for non-variadic operator!")
        if not _variadic and 0 < self.ARITY > len(in_keys):
            raise TypeError("Got too few in_keys ({}) for operation of class {} with arity {}: {}"
                            .format(len(in_keys), self.__class__.__name__, self.ARITY, in_keys))
        if not _variadic and 0 < self.ARITY < len(in_keys):
            raise TypeError("Got too many in_keys ({}) for operation of class {} with arity {}: {}"
                            .format(len(in_keys), self.__class__.__name__, self.ARITY, in_keys))
        if _variadic and len(in_keys) != 0:
            raise TypeError("Variadic instances do not accept in_keys. Either set variadic=True or give in_keys.")
        for child in [c for c in in_keys if isinstance(c, Merge) and c.is_variadic]:
            raise ValueError("Children operations of a formula may not be variadic, "
                             "but found variadic child operation {}.".format(repr(child)))
        # endregion

        if symb is not None:
            self.SYMB = symb
        if self.SYMB is None:
            raise ValueError("SYMB attribute is None for object of class {}!".format(self.__class__) +
                             " Either set class attribute or specify during init via symb parameter.")

        self._variadic: bool = _variadic
        """See :py:meth:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge.is_variadic`."""

        self.in_keys: Sequence[Union[str, 'Merge']] = in_keys
        """The keys of segmentation masks to unite in given order.
        Keys are either constant strings or a merge operation."""

        self.out_key: str = out_key or str(self)
        """The key to use to store the merge output in the annotations dict.
        Take care to not accidentally overwrite existing keys
        (cf. :py:attr:`overwrite`)."""

        self.keep_keys: Optional[Collection[str]] = keep_keys
        """The keys of intermediate outputs in :py:attr:`all_out_keys` which should
        be added to the return of a call.
        Default (``None`` or empty collection): duplicate children outputs are cached
        but not returned to save memory."""

        self.overwrite: Union[bool, Literal['noop']] = overwrite
        """Whether to overwrite a value in the input dictionary when
        applying this operation.
        The operation is defined in :py:meth:`operation`.
        The key that may be overwritten is stored in :py:attr:`out_key`.
        An exception is raised if this is ``False`` and the key exists.
        If set to ``'noop'`` and :py:attr:`out_key` is in the given
        annotations dict, it is returned unchanged."""

        self.skip_none: bool = skip_none
        """If set to ``True``, when a None input value is encountered simply
        ``None`` is returned. If ``False``, an error is raised."""

        self.replace_none: Optional[Any] = replace_none
        """If not ``None``, any received ``None`` value is replaced by the
        given value. This is done only for computation, the ``None`` value in the
        received dict is left unchanged. Key-value pairs with ``None`` value may
        come from the input or from child operations."""

        self.cache_duplicates: bool = cache_duplicates
        """Whether to cache duplicate child operation outputs with duplicate out_key.
        If set to false, all children and children children are evaluated and the
        values of duplicate ``out_keys`` are evaluated several times and overwritten,
        possibly leading to more computational time while using less memory.
        Note that the order of children execution is determined by their order in
        :py:attr:`in_keys`, depth first for nested operations."""

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance.
        (Mind that in_keys must be expanded! For direct reproduction use copy.)"""
        return dict(in_keys=self.in_keys,
                    out_key=self.out_key,
                    overwrite=self.overwrite,
                    skip_none=self.skip_none,
                    replace_none=self.replace_none,
                    cache_duplicates=self.cache_duplicates,
                    keep_keys=self.keep_keys,
                    _variadic=self.is_variadic,
                    symb=self.SYMB)

    @property
    def setting_defaults(self):
        """Defaults used for :py:attr:`settings`."""
        return dict(out_key=str(self), overwrite=True, skip_none=True, replace_none=None,
                    cache_duplicates=True, keep_keys=None, symb=self.__class__.SYMB,
                    _variadic=False)

    @property
    def pretty_op_symb(self) -> str:
        """Name of the operation symbol suitable for filenames etc."""
        return self.__class__.__name__

    def to_infix_notation(self, sort_key: Callable = None,
                          use_whitespace: bool = False,
                          use_pretty_op_symb: bool = False,
                          precedence: Sequence['Merge'] = None,
                          brackets: Tuple[str, str] = ('(', ')')) -> str:
        """Return an infix str encoding equal for differently sorted operations.
        To define a custom sorting for children of commutative operations, hand over
        the ``sort_key`` argument for the builtin ``sorted``.
        If no ``precedence`` is given, brackets are set around all child operations.

        :param sort_key: sort child operations by the given ``sort_key``
            if the parent operation :py:attr:`IS_COMMUTATIVE`; defaults to alphabetical sorting
        :param use_whitespace: separate infix operation symbols from their arguments by whitespace
        :param use_pretty_op_symb: use the :py:attr:`pretty_op_symb` instead of :py:attr:`SYMB` for
            representation of this operation instance
        :param precedence: apply brackets according to the given ``precedence``;
            if not given, assume this operation is in normal form (no brackets)
            must be a list of :py:class:`Merge` operation classes or instances in
            order of increasing precedence; their ``SYMB`` attribute is used to access
            the operation symbol
        :param brackets: tuple of the left and right bracket symbols to use if needed
        """
        # Get pairs of (symbol, string_repr):
        symbs_and_reprs = [
            (key.SYMB, key.to_infix_notation(sort_key=sort_key, precedence=precedence,
                                             use_whitespace=use_whitespace, use_pretty_op_symb=use_pretty_op_symb))
            if isinstance(key, Merge) else (None, str(key))
            for key in self.in_keys]
        normalized_in_keys = self._set_brackets(symbs_and_reprs,
                                                reference_symb=self.SYMB, precedence=precedence,
                                                brackets=brackets)
        if self.IS_COMMUTATIVE:
            normalized_in_keys = sorted(normalized_in_keys, key=sort_key)
        symb: str = self.pretty_op_symb if use_pretty_op_symb else self.SYMB
        if len(normalized_in_keys) == 0:
            return symb
        if len(normalized_in_keys) == 1:
            return f"{symb}{normalized_in_keys[0]}"
        return (f' {symb} ' if use_whitespace else symb).join(normalized_in_keys)

    def to_str(self, **infix_notation_kwargs) -> str:
        """Alias for :py:meth:`to_infix_notation`."""
        return self.to_infix_notation(**infix_notation_kwargs)

    def to_pretty_str(self, **infix_notation_kwargs) -> str:
        """Same as :py:meth:`to_str` but using pretty operation names suitable for
        filenames etc."""
        return self.to_str(**{**infix_notation_kwargs, **dict(use_pretty_op_symb=True)})

    @staticmethod
    def _set_brackets(symbs_and_str: Sequence[Tuple[str, str]],
                      reference_symb: str = None,
                      precedence: Sequence['Merge'] = None,
                      brackets: Tuple[str, str] = ('(', ')')) -> List[str]:
        """Join the strings from the symbol-string-tuples with brackets where needed wrt ``precedence``.
        If no ``precedence`` is given or none is available for ``reference_symb``,
        brackets are set around all strings with non-``None`` symbols (i.e. all but variables).

        :param symbs_and_str: list tuples of the form ``(operation_symbol, operation_string_representation)``;
            the string representations are to be joined, enclosing those in brackets that
            have a lower precedence than ``reference_precedence``
        :param reference_symb: set all operation strings in brackets that have a lower
            precedence than that associated with the operation with ``reference_symb``;
            should be set to the common parent operation symbol
        :param precedence: list of Merge operation classes in order of increasing precedence;
            their ``SYMB`` class or instance attribute is used to access the operation symbol
        """
        symb_to_prec: Mapping[str, int] = {} if precedence is None else \
            dict((precedence[i].SYMB, i) for i in range(len(precedence)))
        if reference_symb not in symb_to_prec:
            return [s if symb is None else f"{brackets[0]}{s}{brackets[1]}" for symb, s in symbs_and_str]
        reference_prec: int = symb_to_prec[reference_symb]
        # Apply brackets to string_repr for higher precedence symbols:
        bracketed_str = [f"{brackets[0]}{p_str}{brackets[1]}"
                         if (symb is not None and symb_to_prec.get(symb, -1) <= reference_prec) else p_str
                         for symb, p_str in symbs_and_str]
        return bracketed_str

    def __str__(self):
        return self.to_str()

    def to_repr(self, settings: Dict[str, Any] = None, defaults: Dict[str, Any] = None,
                sort_key: Callable = None, use_module_names: bool = False,
                indent: str = None, indent_level: Optional[int]=None, indent_str: str = '    ',
                indent_first_child: bool = None, _prepend_indent: bool = True) -> str:
        """Return str representation which can be used to reproduce and compare the instance.

        .. warning::
            Tautologies in the form of duplicate children are not filtered for now, e.g.
            >>> from hybrid_learning.fuzzy_logic.tnorm_connectives.boolean import AND
            >>> AND("a") == AND("a", "a")
            False

        Examples:

        >>> from hybrid_learning.fuzzy_logic.tnorm_connectives.boolean import AND, OR
        >>> obj = OR(AND("b", "c"), "a", symb="CustomAND", overwrite=False,)
        >>> print(obj.to_repr())
        OR('a', AND('b', 'c'), overwrite=False, symb='CustomAND')
        >>> print(obj.to_repr(indent=True))
        OR('a',
           AND('b',
               'c'),
           overwrite=False, symb='CustomAND')
        >>> print(obj.to_repr(indent_first_child=True))
        OR(
           'a',
           AND(
               'b',
               'c'),
           overwrite=False, symb='CustomAND')
        >>> print(obj.to_repr(indent_level=1, indent_str='--'))
        --OR('a',
        ----AND('b',
        ------'c'),
        ----overwrite=False, symb='CustomAND')

        :param settings: the settings dict to include as key-value pairs;
            defaults to :py:attr:`settings` (set e.g. to overwrite this method)
        :param defaults: updates to :py:attr:`setting_defaults`; if a default for a key is given and
            the value equals the default, it is excluded from printing
        :param sort_key: sort child operations by the given ``sort_key``
            if the parent operation :py:attr:`IS_COMMUTATIVE`; defaults to alphabetical sorting
        :param use_module_names: whether to use both module plus class names or just the class names
        :param indent: if not ``None``, print a tree-like view by putting each ``in_keys``
            item in a new line with indent matching the class name length;
            takes precedence over ``indent_level`` and ``indent_str`` arguments
        :param indent_level: if not ``None`` and indent is ``None``, print a tree-like view
            by putting each ``in_keys`` item in a new line with indent of ``indent_str``;
            if >0, ``indent_level*indent_str`` is prepended to every printed line.
        :param indent_str: the (whitespace) string representing one indentation level
        :param indent_first_child: whether to already indent the first child or not
        :param _prepend_indent: whether to prepend the given indent to the output string (default: yes)
        """
        # Class name
        class_name = self.__class__.__name__
        if use_module_names:
            class_name = f"{self.__class__.__module__}.{class_name}"

        # Indentation settings
        if (indent_first_child is not None and indent_level is None and indent is None) or indent is True: indent = ''
        do_indent: bool = indent is not None or indent_level is not None
        base_indent = indent if indent is not None else (indent_level * indent_str if indent_level else '')
        key_indent = base_indent + (' ' * (len(class_name)+1) if indent is not None else (indent_str if indent_level is not None else ''))
        child_sep = ',\n' + key_indent if do_indent else ', '
        setting_first_sep = ',\n' + key_indent if do_indent else ', '
        child_first_sep = '\n' + key_indent if (do_indent and indent_first_child) else ''
        last_sep = '' # '\n' + base_indent

        # Keys
        key_reprs = [key.to_repr(sort_key=sort_key, use_module_names=use_module_names,
                                 indent=key_indent if indent else None, _prepend_indent=False,
                                 indent_first_child=indent_first_child,
                                 indent_level=None if indent_level is None else indent_level+1, indent_str=indent_str)
                     if isinstance(key, Merge) else repr(key)
                     for key in self.in_keys]
        if self.IS_COMMUTATIVE:
            key_reprs = sorted(key_reprs, key=sort_key)

        # Settings
        defaults = {**self.setting_defaults, **(defaults or {})}
        settings = settings or self.settings
        setting_reprs = [f"{key}={f'{val.__module__}.{val.__name__}' if inspect.isclass(val) else repr(val)}"
                         for key, val in sorted(settings.items())
                         if key != 'in_keys' and (key not in defaults or defaults[key] != val)]

        return ((base_indent if _prepend_indent else '') + class_name + "("
                + (child_first_sep + child_sep.join(key_reprs) if len(key_reprs) else '')
                + ((setting_first_sep if len(key_reprs) else '')
                   + f", ".join(setting_reprs) if len(setting_reprs) else '')
                + last_sep + f")")

    def __repr__(self) -> str:
        """Call :py:meth:`to_repr` without sorting."""
        return self.to_repr(sort_key=lambda _: 1)

    def __eq__(self, other: 'Merge') -> bool:
        """Two merge operations are considered equal, if their
        normalized representations coincide. (See :py:meth:`to_repr`).
        This means, they recursively have the same children up to commutation.
        """
        if not isinstance(other, Merge):
            return NotImplemented
        return self.to_repr() == other.to_repr()

    def __copy__(self) -> 'Merge':
        """Return a deep copy of self using settings."""
        setts = self.settings
        in_keys = [k.__copy__() if isinstance(k, Merge) else str(k)
                   for k in setts.pop('in_keys')]
        return self.__class__(*in_keys, **setts)

    def treerecurse_replace_keys(self, **replace_map: Dict[str, str]) -> 'Merge':
        """Return a new formula with all occurences of variables in ``replace_map`` replaced
        and else identical settings.
        The children of the new formula instance are new instances as well.
        
        :param replace_map: mapping ``{old_var_name: new_var_name}``
        """
        setts = self.settings
        in_keys = [k.treerecurse_replace_keys(**replace_map) if isinstance(k, Merge)
                   else replace_map.get(str(k), str(k))
                   for k in setts.pop('in_keys')]
        return self.__class__(*in_keys, **setts)

    def treerecurse(self, fun: Callable[[Union['Merge', str]], Optional['Merge']]) -> 'Merge':
        """Apply the given function recursively to this and all children instances.
        If ``fun`` returns ``None``, the operation is assumed to have been inline.
        A non-``None`` return replaces the original root respectively ``in_keys`` item.
        Acting root before children and depth first."""
        fun_out: Optional[Merge] = fun(self)
        curr_root = self if fun_out is None else fun_out
        if isinstance(curr_root, Merge):
            curr_root.in_keys = [k.treerecurse(fun) if isinstance(k, Merge) else fun(str(k))
                                 for k in curr_root.in_keys]
        return curr_root
            
    def __call__(self, annotations: Union[Mapping[str, Any], Iterable],
                 keep_keys: Collection[str] = None
                 ) -> Union[Mapping[str, Any], Any]:
        """Call method modifying a given dictionary."""
        return self.apply_to(annotations, keep_keys=keep_keys)

    def apply_to(self, annotations: Union[MutableMapping[str, Any], Iterable],
                 keep_keys: Collection[str] = None,
                 ) -> Union[Mapping[str, Any], Any]:
        """Apply this operation to the ``annotations`` dict.
        In case of a :py:meth:`variadic_` instance, also a plain iterable may be given,
        see :py:meth:`variadic_apply_to` which is called in that case.
        The operation of this instance is defined in :py:attr:`operation`.
        First apply all child operations to the dict.
        Hereby try to overwrite a value of annotations if its key correspond
        to an :py:attr:`out_key` of a child operation, but do not create the
        value of a key twice. Then apply :py:attr:`operation` on the
        originally given and generated values now stored in ``annotations``
        and store the result also in ``annotations``.

        .. warning::
            Annotations is inline updated.
            Especially, the :py:attr:`out_key` and ``keep_keys`` items are added,
            and children may apply inline operations to values!

        :param annotations: dict to modify by adding values for :py:attr:`out_key` and ``keep_keys``
        :param keep_keys: the output keys in :py:attr:`all_out_keys` for which
            values shall be added to ``annotations`` in addition to :py:attr:`keep_keys`
        :return: modified ``annotations`` dict, extended by the keys from
            :py:attr:`all_out_keys` with the recursively generated values;
            variadic instances return the plain output of :py:meth:`operation`
        """
        if self.is_variadic:
            return self.variadic_apply_to(annotations)
        keep_keys: List[str] = [*(keep_keys or []), *(self.keep_keys or [])]
        # region value checks
        if not isinstance(annotations, MutableMapping):
            raise TypeError(("Non-variadic instances of class {} only accept mutable mappings "
                             "as input to __call__, but got input of type {}")
                            .format(self.__class__, type(annotations)))
        # About to overwrite a value without permission?
        if self.out_key in annotations.keys():
            if not self.overwrite:
                raise KeyError(("out_key {} exists as key in given dict {}, and "
                                "overwrite is False")
                               .format(self.out_key, annotations))
            elif self.overwrite == 'noop':
                return annotations

        # Any needed in_keys missing from annotations?
        missing_keys: Set[str] = self.all_in_keys - annotations.keys()
        if len(missing_keys) > 0:
            raise ValueError(("Input keys {} for operation {} missing from "
                              "annotation keys {}")
                             .format(missing_keys, repr(self),
                                     annotations.keys()))
        # endregion

        # region get and add children outputs
        # collect from children besides direct output: keys needed for caching, keys in keep_keys
        _seen = []
        children_keep_keys: Sequence[str] = list({
            *([key for key in self._all_out_keys_with_duplicates
               if key in _seen or _seen.append(key)] if self.cache_duplicates else []),
            *keep_keys})
        # get children outputs needed for operation
        children_results = dict(annotations)
        keys_to_overwrite: Set[str] = self.all_out_keys.intersection(
            children_results.keys())
        for child_op in self.children:
            # Output not yet created/existent?
            if not self.cache_duplicates \
                    or child_op.out_key not in children_results.keys() \
                    or child_op.out_key in keys_to_overwrite:
                children_results = child_op(children_results, keep_keys=children_keep_keys)
            # Mark output as created.
            if self.cache_duplicates and child_op.out_key in keys_to_overwrite:
                keys_to_overwrite.remove(child_op.out_key)

        # add children outputs marked for keeping
        annotations.update({key: v for key, v in children_results.items()
                            if key in [*annotations.keys(), *keep_keys]})
        # endregion

        # region skip or fill None
        # Any needed input is None?
        if any(children_results[k] is None for k in self.operation_keys):
            if self.skip_none:  # Fill output with None
                annotations[self.out_key] = None
                return annotations
            if self.replace_none is None:
                raise ValueError("Received None values for keys {}"
                                 .format([k for k in set(self.operation_keys) if annotations[k] is None]))
        # endregion

        # Finally execute operation:
        op_inputs = (children_results[k] for k in self.operation_keys)
        annotations[self.out_key] = self.operation([self.replace_none if v is None else v for v in op_inputs])
        return annotations

    def variadic_apply_to(self, annotations: Union[Mapping[str, Any], Iterable]) -> Any:
        """Return the result of operation on the values/items of a mapping or sequence
        of arbitrary length. Performs ``None`` check/replacement and :py:attr:`ARITY` check.
        In case of a :py:attr:`ARITY` of -1 and empty annotations list, or an annotations
        list length not matching the arity, an :py:class:`IndexError` is raised."""
        if isinstance(annotations, Mapping):
            annotations: Sequence = list(annotations.values())
        elif not isinstance(annotations, Sequence):
            annotations: Sequence = list(annotations)
        if self.ARITY != -1 and len(annotations) != self.ARITY:
            raise IndexError("Length of the given annotations ({}) does not match ARITY ({})!"
                             .format(len(annotations), self.ARITY))
        elif len(annotations) == 0:
            raise IndexError("Empty annotations list provided!")
        # region skip or replace None
        if any(v is None for v in annotations):
            if self.skip_none:
                return None
            if self.replace_none:
                annotations = [self.replace_none if v is None else v for v in annotations]
            else:
                raise ValueError("Received None values in variadic input {}".format(annotations))
        # endregion
        return self.operation(annotations)

    @property
    def children(self) -> List['Merge']:
        """The input keys which are child operations.
        Input keys are stored in :py:attr:`in_keys`"""
        return [key for key in self.in_keys if isinstance(key, Merge)]

    @property
    def all_children(self) -> List['Merge']:
        """All children operations in the flattened computational tree, sorted depth first.
        See :py:attr:`children` for getting only the direct children."""
        direct_children: List['Merge'] = self.children
        return [child for dchild in direct_children
                for child in [dchild, *dchild.all_children]]

    @property
    def consts(self) -> Set[str]:
        """The constant string keys in the input keys.
        The :py:attr:`in_keys` contains both the constant keys which are to be
        directly found in a given annotations dictionary, and child
        operations whose output is used. For getting the child operations
        stored in :py:attr:`in_keys` refer to :py:attr:`children`.
        Should preserve the order in which children occur in :py:attr:`in_keys`.
        """
        return {key for key in self.in_keys if not isinstance(key, Merge)}

    @property
    def operation_keys(self) -> List[str]:
        """The list of keys used for this parent operation in original order
        (constants and children output keys).
        These are all :py:attr:`consts` and the :py:attr:`out_key` of all
        :py:attr:`children` operations.
        Keys may be duplicate as e.g. in

        >>> from hybrid_learning.fuzzy_logic.tnorm_connectives.boolean import OR, NOT
        >>> OR("a", NOT("b"), "a", NOT("c", out_key="not_c")).operation_keys
        ['a', '~b', 'a', 'not_c']
        """
        return [key.out_key if isinstance(key, Merge) else key for key in self.in_keys]

    @property
    def all_in_keys(self) -> Set[str]:
        """All string input keys both of self and of all child operations.
        (See :py:attr:`in_keys`.)
        These are the keys that must be present in an annotation when called
        on it. Should preserve the order in which keys and children occur in
        :py:attr:`in_keys`.
        """
        base_key_lists: List[List[str]] = \
            [key.all_in_keys if isinstance(key, Merge) else [key]
             for key in self.in_keys]
        return {k for base_key_list in base_key_lists for k in base_key_list}

    @property
    def all_out_keys(self) -> Set[str]:
        """Output keys of self and all child operations.
        (See :py:attr:`children`).
        Should preserve the order in which children occur in :py:attr:`in_keys`.
        """
        return set(self._all_out_keys_with_duplicates)

    @property
    def _all_out_keys_with_duplicates(self) -> List[str]:
        """Output keys of self and all child operations with duplicates.
        (See :py:attr:`all_out_keys`).
        """
        out_key_lists: List[Set[str]] = [c.all_out_keys for c in
                                         self.children] + [{self.out_key}]
        return [k for out_key_list in out_key_lists for k in out_key_list]

    def operation(self, annotation_vals: Sequence) -> Any:
        """Actual merge operation on values of the input keys
        in annotations. See :py:attr:`in_keys`.
        The ``annotation_vals`` must not contain ``None`` values,
        and their length must match the :py:attr:`ARITY` of this operation."""
        raise NotImplementedError()

    @classmethod
    def with_(cls, **additional_args) -> 'MergeBuilder':
        """Return a :py:class:`MergeBuilder` with the same symbol but additional init args.
        Example usage (with changed symbol):

        >>> from hybrid_learning.fuzzy_logic.tnorm_connectives.boolean import AND
        >>> builder = AND.with_(skip_none=False, replace_none=0).symb_('&n&')
        >>> builder.SYMB
        '&n&'
        >>> builder("a", "b")
        AND('a', 'b', replace_none=0, skip_none=False, symb='&n&')

        """
        return MergeBuilder(cls, symb=cls.SYMB, additional_args=additional_args)


_OpBuilder = Type[Merge]
_TensorType = Union[torch.Tensor, np.ndarray]
_NumericType = Union[bool, int, float, _TensorType]


class MergeBuilder:
    """Return a :py:class:`Merge` operation of specified class with additional settings upon call.
    Common additional init arguments can be specified and a new :py:attr:`SYMB`,
    overwriting the :py:attr:`Merge.SYMB` (or attaching a ``SYMB`` attribute to another builder).
    Attribute access is passed over to the :py:attr:`merge_class` specified.
    For easy instantiation see also :py:meth:`Merge.with_`."""

    def __init__(self, merge_class: Union[_OpBuilder, Callable[..., Merge]],
                 symb: str = None, additional_args: Dict[str, Any] = None):
        self.merge_class: Union[_OpBuilder, Callable[..., Merge]] = merge_class
        """The class or builder to use upon call."""
        self._additional_args: Dict[str, Any] = additional_args or {}
        """See :py:attr:`additional_args`."""
        self.SYMB: str = symb or merge_class.SYMB
        """The symbol representing the wrapped class."""

    @property
    def additional_args(self) -> Dict[str, Any]:
        """The additional arguments to and over to the :py:class:`Merge` class on each call."""
        return {**self._additional_args,
                **(dict(symb=self.SYMB) if self.SYMB != self.merge_class.SYMB else {})}

    def symb_(self, symb: str) -> 'MergeBuilder':
        """Set SYMB and return self.
        Can be used in chain assignments."""
        self.SYMB = symb
        return self

    def with_(self, **additional_args: Dict[str, Any]):
        """Update the additional arguments."""
        self._additional_args.update(additional_args)
        return self

    def variadic_(self, *args, **kwargs) -> Merge:
        """Return a variadic instance of the wrapped ``merge_class``.
        Calls the ``variadic_`` function of :py:attr:`merge_class`."""
        try:
            self.merge_class: Type[Merge]
            return self.merge_class.variadic_(*args, **{**self.additional_args, **kwargs})
        except TypeError as t:
            t.args = (*t.args, "Building variadic instance of {} of arity {} with {} arguments failed: {}(*{}, **{})"
                      .format(self.merge_class, getattr(self.merge_class, 'ARITY', 'unknown'),
                              len(args), self.merge_class.__name__, args, {**self.additional_args, **kwargs}))
            raise t

    def __call__(self, *args, **kwargs) -> Merge:
        """Build an instance of the specified Merge class with the additional args.
        The given ``kwargs`` will overwrite arguments from :py:attr:`additional_args`."""
        try:
            return self.merge_class(*args, **{**self.additional_args, **kwargs})
        except TypeError as t:
            t.args = (*t.args, "The following init call to {} of arity {} with {} arguments failed: {}(*{}, **{})"
                      .format(self.merge_class, getattr(self.merge_class, 'ARITY', 'unknown'),
                              len(args), self.merge_class.__name__, args, {**self.additional_args, **kwargs}))
            raise t

    def __getattr__(self, k):
        """Pass attribute requests over to Merge class."""
        if 'merge_class' not in vars(self):
            raise AttributeError()
        return getattr(self.merge_class, k)

    def __repr__(self):
        merge_class_repr = f"{self.merge_class.__module__}.{self.merge_class.__name__}" \
            if inspect.isclass(self.merge_class) else repr(self.merge_class)
        setts = {}
        if self.SYMB != self.merge_class.SYMB:
            setts['symb'] = repr(self.SYMB)
        if self._additional_args:
            setts['additional_args'] = repr(dict(sorted(self._additional_args.items())))
        return (self.__class__.__name__ + "(" + merge_class_repr + ", " +
                ", ".join([f'{key}={val}' for key, val in setts.items()])
                + ")")


class TorchOperation(Merge, abc.ABC):
    """Generic merge operation on torch tensors."""

    @staticmethod
    @abc.abstractmethod
    def torch_operation(*inputs: torch.Tensor) -> torch.Tensor:
        """Operation on pytorch tensors.
        If possible, the operation should support broadcasting."""
        raise NotImplementedError()

    def operation(self, annotation_vals: Sequence) -> torch.Tensor:
        """Calculate the predicate output.
        Non-tensor inputs are transformed to tensors.
        See :py:meth:`torch_operation`."""
        if len(annotation_vals) < self.ARITY or (self.ARITY > 0 and len(annotation_vals) > self.ARITY):
            raise TypeError("Operation {} of type {} and arity {} was called with {} inputs:\n{}"
                            .format(self, type(self), self.ARITY, len(annotation_vals), annotation_vals))
        masks = annotation_vals[:(self.ARITY if self.ARITY >= 0 else len(annotation_vals))]
        masks: List[torch.Tensor] = [torch.as_tensor(mask) for mask in masks]
        return self.torch_operation(*masks)


class TorchOrNumpyOperation(TorchOperation, abc.ABC):
    """Generic merge operation allowing to define both a torch and a numpy operation.
    Which one is selected depends on the types of the provided annotations:
    If any is a torch tensor, the torch operation is used and a torch tensor
    returned, otherwise the numpy operation.
    """

    @staticmethod
    @abc.abstractmethod
    def numpy_operation(*inputs: np.ndarray) -> np.ndarray:
        """Operation on Booleans, numpy arrays and numbers.
        If possible, the operation should support broadcasting."""
        raise NotImplementedError()

    def operation(self, annotation_vals: Sequence) -> _NumericType:
        """Operation on either torch tensors or Booleans, numpy arrays and numbers."""
        if any(isinstance(inp, torch.Tensor) for inp in annotation_vals):
            return self.torch_operation(*[ToTensor.to_tens(inp) for inp in annotation_vals])
        else:
            return self.numpy_operation(*annotation_vals)
