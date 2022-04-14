"""Simple utility functions common to different types of transformations,
and trafo builders.
"""
#  Copyright (c) 2022 Continental Automotive GmbH
import abc
from typing import Dict, Callable, Any, Union, List, Optional, Sequence, Tuple


def settings_to_repr(obj, settings: Dict) -> str:
    """Given an object and a dict of its settings, return a representation str.
    The object is just used to derive the class name.

    :meta private:
    """
    return "{}({})".format(str(obj.__class__.__name__),
                           ', '.join(['='.join([str(k), repr(v)])
                                      for k, v in settings.items()]))


def lazy_format(string, **formatting_args):
    """Replace all keys from formatting_args that can be found in string."""
    formatting = {key: val for key, val in formatting_args.items()
                  if "{"+key+"}" in string}
    if len(formatting) > 0:
        string = string.format(**formatting)
    return string


def general_add(first, second, *,
                composition_class,
                identity_class: Union[type, List[type]] = None,
                type_check: Callable[[Any], bool] = callable):
    # pylint: disable=line-too-long
    """Return a flat, associative composition of first and second.
    Elements of ``identity_class`` and ``None`` are treated as neutral element,
    and all summands must either pass the ``type_check`` or be neutral.
    In case both summands are non-neutral, a ``composition_class`` instance
    of both is returned. If any already is of type ``composition_class``,
    this one is unpacked to ensure flatness of the returned composition.
    It is worked on copies of the summands if these provide a ``__copy__``
    method.
    The ``composition_class`` must be iterable and accept as init argument
    a sequence of the summands.

    .. note::
        This operation is in general not commutative.
        Associativity is ensured via the flatness.

    :param first: summand
    :param second: summand
    :param composition_class: the class to use for generating the flat
        composition
    :param identity_class: optionally a class that is treated as neutral element
    :param type_check: a callable that accepts any of ``first`` or ``second``
        and returns a bool whether they have an appropriate type or not
    :return: one of the summands in case the other is neutral,
        else a ``composition_class`` instance;
        ``NotImplemented`` is returned if any non-``None`` summand
        doesn't pass the ``type_check``
    """
    # pylint: enable=line-too-long
    # Ensure using copied instances if possible
    first = first.__copy__() if hasattr(first, '__copy__') else first
    second = second.__copy__() if hasattr(second, '__copy__') else second
    is_neutral = lambda el: \
        el is None or (isinstance(el, identity_class)
                       if identity_class is not None else False)

    # wrong type combination
    if not all(is_neutral(el) or type_check(el)
               or isinstance(el, composition_class) for el in (first, second)):
        return NotImplemented

    # no-op:
    if is_neutral(second):
        return first if first is not None else second
    if is_neutral(first):
        return second if second is not None else first

    # addition with Compose instances:
    if isinstance(first, composition_class) and \
            isinstance(second, composition_class):
        composition = [*first, *second]
    elif isinstance(first, composition_class):
        composition = [*first, second]
    elif isinstance(second, composition_class):
        composition = [first, *second]
    else:  # default
        composition = [first, second]
    return composition_class(composition)


class Transform(abc.ABC):
    """Base class for transformations."""

    IDENTITY_CLASS: Optional[Union[type, List[type]]] = None
    """The identity class or classes for composition / addition.
    See :py:func:`general_add`."""

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        return {}

    def __repr__(self):
        return settings_to_repr(self, self.settings)

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __copy__(self):
        """Return a shallow copy of self using settings."""
        # noinspection PyArgumentList
        return self.__class__(**self.settings)

    def __add__(self, other: Optional[Union[Callable, 'Transform']]
                ) -> 'Transform':
        # pylint: disable=line-too-long
        """Return a flat composition of ``self`` with ``other``.
        In case ``other`` is a no-op transforms, return a copy of ``self``.

        :return: one of the summands in case the other is a no-op, else a
            :py:class:`hybrid_learning.datasets.transforms.common.Compose`
            transforms
        """
        # pylint: enable=line-too-long
        return general_add(self, other, composition_class=Compose,
                           identity_class=self.IDENTITY_CLASS)

    def __radd__(self, other: Optional[Union[Callable, 'Transform']]
                 ) -> 'Transform':
        """Return a flat composition of ``other`` and ``self``.
        See :py:meth:`__add__`."""
        return general_add(other, self, composition_class=Compose,
                           identity_class=self.IDENTITY_CLASS)

    def __call__(self, *inps):
        """General call to the transformation."""
        return self.apply_to(*inps)

    @abc.abstractmethod
    def apply_to(self, *inps):
        """Application of transformation."""
        raise NotImplementedError()


class Compose(Transform):
    """Compose several transforms by sequentially executing them."""

    def __init__(self, transforms: Sequence[Callable]):
        self.transforms: List[Callable] = list(transforms)
        """List of tuple transformations to apply in order."""

    def append(self, trafo: Callable):
        """Append transformation to the processing chain."""
        self.transforms.append(trafo)

    def insert(self, i: int, trafo: Callable):
        """Insert ``trafo`` at index ``i`` into the processing chain."""
        self.transforms.insert(i, trafo)

    def __getitem__(self, i: int):
        """Get item from transforms member."""
        return self.transforms[i]

    def __len__(self):
        """Get length of transforms member."""
        return len(self.transforms)

    @property
    def settings(self):
        """Settings to reproduce the instance."""
        return dict(transforms=self.transforms)

    def apply_to(self, *inps) -> Tuple:
        """Apply all transformations in order.
        Transformations are taken from :py:attr:`transforms`."""
        inps = tuple(inps)
        for trafo in self.transforms:
            try:
                inps = trafo(*inps) if isinstance(inps, tuple) else trafo(inps)
            except TypeError as t:
                raise TypeError(("{}"
                                 "\nContext:\n called trafo: {}\n in trafo chain: {}\n on values: {}"
                                 ).format(str(t), repr(trafo), repr(self), inps))
        return inps


class Lambda(Transform):
    """Generic lambda transformation that applies the given function.

    .. warning::
        Note that lambda functions by default cannot be pickled and
        thus will cause errors when used in a multiprocessing context!
        It is strongly recommended to provide a callable object instead.
    """

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        return dict(fun=self.fun)

    def __repr__(self) -> str:
        return settings_to_repr(self, dict(
            fun=(self.fun.__name__
                 if hasattr(self.fun, "name") else repr(self.fun))))

    def __init__(self, fun: Callable):
        """Init.

        :param fun: the function to apply on call
        """
        self.fun: Callable = fun
        """The function to apply on call."""

    def apply_to(self, *inp):
        """Application of the lambda."""
        return self.fun(*inp)
