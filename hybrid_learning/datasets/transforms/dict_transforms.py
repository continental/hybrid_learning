"""Transformations on dict annotations containing scalars and same-sized masks.

The annotations that can be processed are supposed to be a dict with string
keys and values of floats or :py:class:`numpy.ndarray` (binary) masks.

The logical merging operations derived from :py:class:`Merge` allow for
concatenation. Using them, any operation involving intersection, union,
and inversion of masks can be modelled. Scalar values in this case are
treated as all-same-valued masks when mixed with mask tensors.
For further information have a look at the :py:class:`Merge` documentation.
"""
#  Copyright (c) 2020 Continental Automotive GmbH

import abc
from typing import Dict, Any, Union, Sequence, Set, List, Iterable, Optional

import numpy as np


class DictTransform(abc.ABC):
    """Basic transformation for dicts.
     This means a callable yielding a dict of a single value."""

    @abc.abstractmethod
    def __call__(self, annotations: Dict[str, Any]
                 ) -> Union[Dict[str, Any], Any]:
        """Call method modifying a given dictionary."""
        raise NotImplementedError()


class Merge(DictTransform, abc.ABC):
    """Merge the binary masks / scalars of the input keys and store them under
    the specified output key.
    The merge operation may recursively have other child merge operation as
    :py:attr:`in_keys`, which are evaluated on the given dictionary before
    the parent is.

    **Operation**

    The actual operation is hidden in the :py:meth:`__call__` method:
    It is given a dictionary of annotations of the form ``{ID: value}`` and
    will return the dict with the merged mask added as ``{out_key: value}``.
    The intermediate outputs of all child operations are also added, so make
    sure to remove them afterwards if they are not needed anymore. The
    benefit of this is that results may be reused amongst different operations.

    **Format**

    The (recursive) merge operation must be specified in conjunctive normal
    form, i.e. of the form

    .. code::

        AND(..., [NOT(...), ...], [OR(..., [NOT(..), ...])])

    (see https://en.wikipedia.org/wiki/Conjunctive_normal_form).
    Available operations are
    :py:class:`AND` (intersection),
    :py:class:`OR` (union), and
    :py:class:`NOT` (inversion).
    Boolean classification labels are treated as all-one-masks.
    One can use :py:meth:`Merge.parse` to parse a string representation of an
    operation tree. There, again, the formula must be in conjunctive normal
    form and connectors must be encoded as follows:

    - :py:class:`AND`: a&&b
    - :py:class:`OR`: a||b
    - :py:class:`NOT`: ~a

    **Examples**

    To get all heads, noses, and mouths of real persons in bathrooms, call:

    >>> a = AND("person", OR("head", "nose", "mouth"), NOT("bathroom"))
    >>> a == Merge.parse("person&&head||nose||mouth&&~bathroom")
    True
    """
    SYMB: str = None
    """The string symbol of this class (override for sub-classes)."""

    def __init__(self, *in_keys: Union[str, 'Merge'], out_key: str = None,
                 overwrite: bool = True, skip_none: bool = True,
                 replace_none=None):
        """Init.

        Hand over input keys either as str or as a Merge operation of str.

        :param in_keys: sequence of either :py:class:`Merge` operation
            instances or strings with placeholders for the input keys
        :param out_key: key for the output of this operation; used to
            init :py:attr:`~Merge.out_key`
        :param overwrite: on call, whether to overwrite the value at
            :py:attr:`~Merge.out_key` in the given dict if the key already
            exists; raise if key exists and ``overwrite`` is true;
            saved in :py:attr:`~Merge.overwrite`.
        :param replace_none: if not ``None``, the value to replace any
            ``None`` values with; see :py:attr:`~Merge.replace_none`
        """
        if len(in_keys) <= 0:
            raise ValueError("Got empty list of in_keys!")
        if not isinstance(self, NOT) and len(in_keys) < 2:
            raise ValueError("Too few in_keys: expected >=2, got 1 ({})"
                             .format(in_keys))

        self.in_keys: Sequence[Union[str, 'Merge']] = in_keys
        """The keys of segmentation masks to unite (either string or a merge
        operation)."""
        if not self.is_conjunctive_normal_form():
            raise ValueError(("Specified in_keys are not in conjunctive "
                              "normal form! Given: {}")
                             .format(self.normalized_repr()))

        self.out_key: str = out_key or str(self)
        """The key to use to store the merge output in the annotations dict.
        Take care to not accidentally overwrite existing keys
        (cf. :py:attr:`overwrite`)."""

        self.overwrite: bool = overwrite
        """Whether to overwrite a value in the input dictionary when
        applying this operation.
        The operation is defined in :py:meth:`operation`.
        The key that may be overwritten is stored in :py:attr:`out_key`.
        An exception is raised if this is ``False`` and the key exists."""

        self.skip_none: bool = skip_none
        """If set to ``True``, when a None input value is encountered simply
        ``None`` is returned. If ``False``, an error is raised."""

        self.replace_none: Optional[Any] = replace_none
        """If not ``None``, any received ``None`` value is replaced by the
        given value. Key-value pairs with ``None`` value may come from the
        input or from child operations."""

    def normalized_repr(self) -> str:
        """Return a str encoding equal for differently sorted operations."""
        return self.SYMB.join(
            sorted([key.normalized_repr() if isinstance(key, Merge) else key
                    for key in self.in_keys]))

    def __repr__(self) -> str:
        """Return str representation which can be used to reproduce instance."""
        return (self.__class__.__name__ + "(" +
                ", ".join([repr(key) for key in self.in_keys]) +
                (", out_key='" + self.out_key + "'" if self.out_key != str(
                    self) else "") +
                ")")

    def __str__(self) -> str:
        """Return str yielding an equal instance with same
        :py:attr:`~Merge.out_key` when parsed."""
        return self.SYMB.join([str(key) for key in self.in_keys])

    def __eq__(self, other: 'Merge') -> bool:
        """Two merge operations are considered equals, if their
        (normalized) representations coincide. (See :py:meth:`normalized_repr`).
        This means, they recursively have the same children up to commutation.

        .. warning::
            Duplicate children are not filtered for now,
            i.e. ``AND("a") != AND("a", "a")``
        """
        return self.normalized_repr() == other.normalized_repr()

    def __call__(self, annotations: Dict[str, Any]) -> Dict[str, Any]:
        """Apply this operation to the ``annotations`` dict.
        The operation of this instance is defined in :py:attr:`operation`.
        First apply all child operations to the dict.
        Hereby try to overwrite a value of annotations if its key correspond
        to an :py:attr:`out_key` of a child operation, but do not create the
        value of a key twice. Then apply :py:attr:`operation` on the
        originally given and generated values now stored in ``annotations``
        and store the result also in ``annotations``.

        :param annotations: dict to modify by adding values for
            :py:attr:`all_out_keys`
        :return: modified ``annotations`` dict, extended by the keys from
            :py:attr:`all_out_keys` with the recursively generated values
        """
        # About to overwrite a value without permission?
        if not self.overwrite and self.out_key in annotations.keys():
            raise KeyError(("out_key {} exists as key in given dict {}, and "
                            "overwrite is False")
                           .format(self.out_key, annotations))

        # Any needed in_keys missing from annotations?
        missing_keys: Set[str] = self.all_in_keys - annotations.keys()
        if len(missing_keys) > 0:
            raise ValueError(("Input keys {} for operation {} missing from "
                              "annotation keys {}")
                             .format(missing_keys, repr(self),
                                     annotations.keys()))

        # Add children outputs:
        keys_to_overwrite: Set[str] = self.all_out_keys.intersection(
            annotations.keys())
        for child_op in self.children:
            # Output not yet created/existent?
            if child_op.out_key in keys_to_overwrite or \
                    child_op.out_key not in annotations.keys():
                annotations = child_op(annotations)
            # Mark output as created.
            if child_op.out_key in keys_to_overwrite:
                keys_to_overwrite.remove(child_op.out_key)

        # Any needed input is None?
        none_keys: List[str] = [k for k in self.operation_keys if
                                annotations[k] is None]
        if len(none_keys) > 0:
            if self.skip_none:  # Fill output with None
                annotations[self.out_key] = None
                return annotations
            if self.replace_none is not None:
                for k in none_keys:
                    annotations[k] = self.replace_none
            else:
                raise ValueError("Received None values for keys {}"
                                 .format(none_keys))

        # Finally execute operation:
        annotations[self.out_key] = self.operation(annotations)
        return annotations

    @property
    def children(self) -> List['Merge']:
        """The input keys which are child operations.
        Input keys are stored in :py:attr:`in_keys`"""
        return [key for key in self.in_keys if isinstance(key, Merge)]

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
    def operation_keys(self) -> Set[str]:
        """The keys used for this parent operation (constants and children
        output keys).
        All :py:attr:`consts` and the :py:attr:`out_key` of all
        :py:attr:`children` operations."""
        return {*self.consts, *[c.out_key for c in self.children]}

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
        out_key_lists: List[Set[str]] = [c.all_out_keys for c in
                                         self.children] + [{self.out_key}]
        return {k for out_key_list in out_key_lists for k in out_key_list}

    def operation(self, annotations: Dict[str, Any]) -> Any:
        """Actual merge operation on values of the input keys
        in annotations.
        See :py:attr:`in_keys`."""
        raise NotImplementedError()

    @staticmethod
    def parse(specifier: str, **init_args) -> Union[str, 'Merge']:
        """Parse a merge operation in conjunctive normal form.
        Will return the original string if it does not contain operations
        specifiers.

        :param specifier: the string specifier to parse
        :param init_args: any keyword arguments for init of the generated
            operations (parent and all children);
            ``out_key`` is only applied to the parent operation
        """
        # Validate
        if specifier.startswith(AND.SYMB) or specifier.endswith(AND.SYMB):
            raise ValueError("Wrong usage of AND symbol {}: {}"
                             .format(AND.SYMB, specifier))
        if specifier.startswith(OR.SYMB) or specifier.endswith(OR.SYMB):
            raise ValueError("Wrong usage of OR symbol {}: {}"
                             .format(OR.SYMB, specifier))

        symbol_classes = {AND.SYMB: AND, OR.SYMB: OR, NOT.SYMB: NOT}
        for symb, cls in symbol_classes.items():
            # Validate specifier: Are there double occurrences of symb?
            if '' in specifier.split(symb)[1:]:
                raise ValueError(
                    ("Invalid specifier for {} (symbol {}): {} "
                     "but expected {}"
                     ).format(cls.__name__, symb, specifier,
                              symb.join([s for s in specifier.split(symb) if
                                         s != ''])))
            if symb in specifier:
                return cls(*[Merge.parse(spec, **{**init_args, 'out_key': None})
                             for spec in specifier.split(symb) if spec != ''],
                           **init_args)
        return specifier

    @staticmethod
    def apply(specifier: Union[str, 'Merge'], annotations: Dict[str, Any],
              **init_args
              ) -> Dict[str, Any]:
        """Parse ``specifier`` and return its result on annotations
        if it's an operation.

        :param specifier: specifier for :py:meth:`parse`
        :param annotations: dictionary to which to apply the parsed operation
        :param init_args: further keyword arguments to init the parent
            operation while parsing
        """
        return Merge.parse(specifier, **init_args)(annotations)

    def is_conjunctive_normal_form(self) -> bool:
        """Checks whether the current formula is in conjunctive normal form."""
        children = [op for op in self.in_keys if isinstance(op, Merge)]
        children_cls: Set[type] = set((op.__class__ for op in children))
        if AND in children_cls:
            return False
        if isinstance(self, OR) and OR in children_cls:
            return False
        if isinstance(self, NOT) \
                and (OR in children_cls or NOT in children_cls):
            return False
        return all([child.is_conjunctive_normal_form() for child in children])


class AND(Merge):
    """Intersection/AND operation on binary masks and scalars."""
    SYMB = "&&"

    def operation(self, annotations: Dict[str, Any]) -> Any:
        """Store intersection of :py:attr:`~Merge.in_keys` masks as
        :py:attr:`~Merge.out_key`."""
        return np.prod([annotations[key] for key in self.operation_keys],
                       axis=0)


class OR(Merge):
    """Union/OR operation on binary masks and scalars."""
    SYMB = "||"

    def operation(self, annotations: Dict[str, Any]) -> Any:
        """Store union of :py:attr:`~Merge.in_keys` masks as
        :py:attr:`~Merge.out_key`."""
        return np.sum([annotations[key] for key in self.operation_keys],
                      axis=0) > 0


class NOT(Merge):
    """Inversion/NOT operation on binary masks and scalars."""
    SYMB = "~"

    def __init__(self, in_key: str, **init_args):
        """Init.

        Note that there currently may only be one input key
        (which already determines the output key).

        :param in_key: the only input key (must be a string currently)
        :param init_args: further keyword arguments to
            :py:class:`super init <Merge>`
        """
        super(NOT, self).__init__(in_key, **init_args)

    @property
    def in_key(self) -> str:
        """The only operational input key."""
        return self.in_keys[0]

    def operation(self, annotations: Dict[str, Any]) -> Any:
        """Store inverted version of :py:attr:`~Merge.in_keys` as
        :py:attr:`~Merge.out_key`."""
        return np.logical_not(annotations[self.in_key])

    def normalized_repr(self) -> str:
        """Special case of :py:meth:`Merge.normalized_repr`
        (:py:class:`NOT` cannot have any children at the moment)."""
        return self.SYMB + self.in_key

    def __str__(self) -> str:
        return self.SYMB + str(self.in_key)


class DropAnn(DictTransform):
    """Drop the annotation with given key from the annotations dict."""

    def __init__(self, drop_key: str):
        """Init.

        :param drop_key: see :py:attr:`drop_key`
        """
        self.drop_key = drop_key
        """Dict key to drop on call."""

    def __call__(self, annotations: Dict[str, Any]) -> Dict[str, Any]:
        """Drop the item at configured key from ``annotations``.
        The configured key is stored in :py:attr:`drop_key`."""
        annotations.pop(self.drop_key)
        return annotations


class RestrictDict(DictTransform):
    """Restrict the annotation dictionary to the annotation items with
    featuring one of the selected keys."""

    def __init__(self, selected_keys: Iterable[str]):
        """Init.

        :param selected_keys: see :py:attr:`selected_keys`
        """
        self.selected_keys: Iterable[str] = selected_keys
        """The keys to restrict the dict to."""

    def __call__(self, annotations: Dict[str, Any]) -> Dict[str, Any]:
        """Restrict the annotation dict to the selected keys.
        Selected keys are stored in :py:attr:`selected_keys`."""
        return {key: annotations[key] for key in self.selected_keys}


class FlattenDict(DictTransform):
    """Return the value of the annotations dict at selected key."""

    def __init__(self, selected_key: str):
        """Init.

        :param selected_key: see :py:attr:`selected_key`
        """
        self.selected_key: str = selected_key
        """The key of the annotation value to return."""

    def __call__(self, annotations: Dict[str, Any]) -> Any:
        """Flatten ``annotations`` dict to its value at the configured key.
        The configured key is stored in :py:attr:`selected_key`.
        """
        return annotations[self.selected_key]
