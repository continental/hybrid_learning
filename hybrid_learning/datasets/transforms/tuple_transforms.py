"""Transformations that modify tuples of items on call.

The tuple transformations are all based on :py:class:`TupleTransforms`.
:py:class:`ReduceTuple` can be used to merge the output tuple of a
:py:class:`TupleTransforms` into one output value.
The :py:class:`Identity` transformation can be used as dummy transformation
where necessary.
"""
#  Copyright (c) 2022 Continental Automotive GmbH

# TODO: tests for tuple transforms

import abc
import collections, collections.abc
from typing import Dict, Any, Tuple, Callable, Optional, Iterable, Sequence, List, Union, Set

import torch

from .common import Transform, general_add, Compose
from .image_transforms import resize


# TYPES
TwoTuple = Tuple[Any, Any]
TensorTwoTuple = Tuple[torch.Tensor, torch.Tensor]
TensorThreeTuple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class UnfoldTuple(Transform):
    """Convenience trafo that takes an iterable and returns it as tuple.
    Needed e.g. when unravelling a nested tuple for a composition of
    :py:class:`TupleTransforms`:

    >>> from hybrid_learning.datasets.transforms import UnfoldTuple, ToTensor
    >>> c = UnfoldTuple() + OnTarget(UnfoldTuple() + OnTarget(ToTensor()))
    >>> nested_tuple = ('desc', ('inp', 1))
    >>> nested_tuple
    ('desc', ('inp', 1))
    >>> c(nested_tuple)
    ('desc', ('inp', tensor(1)))
    """

    def apply_to(self, tup: Iterable) -> Tuple:
        """Return the given iterable as tuple."""
        return tuple(tup)


class TupleTransforms(Transform):
    """Base class of tuple transformations that accept the
    same amount of inputs as they provide outputs.
    To inherit, override
    :py:attr:`~hybrid_learning.datasets.transforms.common.Transform.settings`
    and :py:meth:`apply_to`.
    """

    def __init__(self):
        """Init. Set the identity transformation for this class."""
        self.IDENTITY_CLASS = Identity  # pylint: disable=invalid-name

    @abc.abstractmethod
    def apply_to(self, *inputs) -> Tuple:
        """Apply the transformation to the tuple ``(inp, target)``;
        the output again is a tuple."""
        raise NotImplementedError()

    def __call__(self, *inputs) -> Tuple:
        """Apply the transformation to the tuple ``(inp, target)``;
        the output again is a tuple."""
        return self.apply_to(*inputs)


class _IndexedTupleTrafo(TupleTransforms, abc.ABC):
    """Any tuple operation working on certain indices of the input tuple."""

    @property
    def settings(self) -> Dict[str, Any]:
        return dict(indices=self.indices, **super().settings)

    def __init__(self, indices: Union[int, Sequence[int]]):
        self.indices: Sequence[int] = [indices] if isinstance(indices, int) else indices
        """The indices of tuple items to which to apply ``trafo``.
        Trafo is applied to items in the order of given indices."""
        if not isinstance(self.indices, collections.abc.Iterable) or \
            not all(isinstance(idx, int) for idx in self.indices):
            raise ValueError("Got invalid (non-int) tuple index specification {}".format(indices))

    def unique_pos_indices_for(self, inputs: Tuple) -> List[int]:
        """For an input tuple of certain length return the unique positive indices to work on.
        The order is the same as in :py:attr:`indices`, with later duplicates removed."""
        indices: List[int] = [idx if idx >= 0 else len(inputs) + idx for idx in self.indices]
        unique_indices = list(dict.fromkeys(indices))
        if not all(len(inputs) > idx >= 0 for idx in unique_indices):
            raise IndexError("Cannot apply trafo to indices {} of tuple of length {}"
                             .format(self.indices, len(inputs)))
        return unique_indices


class _PartialTupleTrafo(TupleTransforms, abc.ABC):
    """Base class for tuple transforms that apply a child transformation
    to some tuple items."""

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        return dict(trafo=self.trafo)
    
    def __init__(self, trafo: Callable):
        """Init.

        :param trafo: the transformation to apply
        """
        super().__init__()
        self.trafo: Callable = trafo
        """The transformation to apply to just one of the items."""


class SubsetTuple(_IndexedTupleTrafo):
    """Return a tuple only containing the elements at given indices of input tuple.
    Indices may be given as positive or negative index.
    Elements are not """

    def apply_to(self, *inputs) -> Tuple:
        return tuple([inputs[i] for i in self.unique_pos_indices_for(inputs)])


class OnIndex(_IndexedTupleTrafo, _PartialTupleTrafo):
    """Apply a given transformation to tuple items at given indices."""

    @property
    def settings(self) -> Dict[str, Any]:
        return {'indices': self.indices, 'trafo': self.trafo,
                **super().settings}

    def __init__(self, indices: Union[int, Sequence[int]],  trafo: Callable):
        _PartialTupleTrafo.__init__(self, trafo)
        _IndexedTupleTrafo.__init__(self, indices)
        
    def apply_to(self, *inputs) -> Tuple:
        outputs = list(inputs)
        # list.insert behaves differently wrt negative indices than list.pop -> make all >=0
        for index in self.unique_pos_indices_for(inputs):
            outputs.insert(index, self.trafo(outputs.pop(index)))
        return tuple(outputs)


class OnAll(_PartialTupleTrafo):
    """Apply a given transformation to all tuple items.
    Variadic version of ``OnIndex``."""

    def apply_to(self, *inputs) -> Tuple:
        """Apply ``trafo`` to all items in given ``inputs`` tuple."""
        return tuple([self.trafo(item) for item in inputs])


class TwoTupleTransforms(TupleTransforms):
    """Version of TupleTransforms for operating on two-tuples only."""

    @abc.abstractmethod
    def apply_to(self, inp, target) -> TwoTuple:
        """Apply the transformation to the tuple ``(inp, target)``;
        the output again is a tuple."""
        raise NotImplementedError()

    def __call__(self, inp, target) -> TwoTuple:
        """Apply the transformation to the tuple ``(inp, target)``;
        the output again is a tuple."""
        return self.apply_to(inp, target)


class OnBothSides(_PartialTupleTrafo, TwoTupleTransforms):
    """Apply a given transformation to both input and target of a tuple in
    parallel.
    Shortcut for ``OnAll(trafo)`` with tuple length enforcement."""

    def apply_to(self, inp, target) -> TwoTuple:
        """Apply transformation in parallel to both ``inp`` and ``target``."""
        return self.trafo(inp), self.trafo(target)


class OnInput(_PartialTupleTrafo, TwoTupleTransforms):
    """Apply a given one-value transformation only to the input of a two-tuple.
    Shortcut for ``OnIndices([0], trafo)`` with tuple length enforcement."""

    def apply_to(self, inp, target) -> TwoTuple:
        """Application of the transformation to the input."""
        return self.trafo(inp), target


class OnTarget(_PartialTupleTrafo, TwoTupleTransforms):
    """Apply a given one-value transformation only to the target of a two-tuple.
    Shortcut for ``OnIndices([1], trafo)`` with tuple length enforcement."""

    def apply_to(self, inp, target) -> TwoTuple:
        """Application of the transformation to the target."""
        return inp, self.trafo(target)


class SameSize(TupleTransforms):
    # pylint: disable=line-too-long
    """Given a tuple of input and target image, resize the target to the
    size of the input.
    Both input and target image must be given as :py:class:`torch.Tensor`.

    Since the center points of pixels are considered, upscaling may
    lead to edge pixel values that exceed the previous maximum values.
    Thus, make sure to clamp the output in case e.g. sigmoid output
    is considered:

    >>> from hybrid_learning.datasets.transforms import Lambda, SameSize
    >>> (SameSize(resize_target=False)
    ...  + OnInput(Lambda(lambda t: t.clamp(0, 1))))
    Compose(transforms=[SameSize(...), OnInput(trafo=Lambda(...))])
    """

    # pylint: enable=line-too-long

    def __init__(self, interpolation="bilinear", resize_target: bool = False,
                 only_two_tuples: bool = None,
                 resize_to_index: int = None):
        """Init.

        .. note::
            Mind that other interpolation methods than ``'nearest'`` will
            produce non-binary outputs.

        :param resize_target: if set to true, shortcut for
            ``resize_to_index=0, only_two_tuples=True``,
            else provides default ``resize_to_index=-1, only_two_tuples=False``
        :param only_two_tuples: check that only two-tuples are provided
        :param resize_to_index: the index of a mask to which to resize;
            see also ``resize_target``
        :param interpolation: the interpolation method to use, parameter to
            :py:class:`torch.nn.Upsample`;
        """
        super().__init__()
        self.only_two_tuples: bool = only_two_tuples or resize_target
        """If ``True`` check that only two-tuples of masks are given."""
        self.resize_to_index: int = resize_to_index if resize_to_index is not None \
            else 0 if resize_target else -1
        """Resize all other masks to the mask at this index."""
        self.interpolation: str = interpolation

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        return dict(interpolation=self.interpolation,
                    only_two_tuples=self.only_two_tuples,
                    resize_to_index=self.resize_to_index)

    def apply_to(self, *masks: torch.Tensor):
        """Application of size adaptation."""
        if self.only_two_tuples and len(masks) != 2:
            raise IndexError("only_two_tuples set to true, but got {} masks to resize"
                             .format(len(masks)))
        non_changing_index = self.resize_to_index if self.resize_to_index >= 0 \
            else len(masks) + self.resize_to_index
        if non_changing_index < 0 or len(masks) < (non_changing_index + 1):
            raise IndexError("Got {} masks to resize but resize_to_index={}"
                             .format(len(masks), self.resize_to_index))
        if len(masks) == 1: return masks[0]  # No-op

        changing: List[torch.Tensor] = list(masks)
        non_changing = changing.pop(non_changing_index)
        new_size = (non_changing.size()[-2], non_changing.size()[-1])

        def trafo(tens: torch.Tensor) -> torch.Tensor:
            """The interpolation transformation to apply."""
            return resize(tens, size=new_size, mode=self.interpolation)

        # Resize the correct one
        result = [trafo(mask) for mask in changing]
        result.insert(non_changing_index, non_changing)
        return tuple(result)


class Identity(TupleTransforms):
    """Simple identity transformation for example for defaults."""

    def apply_to(self, *inputs):
        """Identity operation, nothing changed."""
        return inputs

    def __add__(self, other: Optional[TupleTransforms]) -> TupleTransforms:
        if other is None:
            return self
        if isinstance(other, TupleTransforms):
            return other.__copy__()
        return NotImplemented

    def __radd__(self, other: Optional[TupleTransforms]) -> TupleTransforms:
        return self + other


class ReduceTuple(Transform):
    """Transform an unpacked input tuple and then reduce it to one value
    using a reduction func.
    This is essentially the concatenation of two functions, but with
    type hints assuming a single tensor output, and with better
    One example would reduction by a loss function.

    .. note::
        Attribute access is handed over to :py:attr:`reduction`
        if the attribute cannot be found in this instance.
        This is intended to ease wrapping of metric functions.
    """

    def __getattr__(self, k):
        """Pass attribute requests over to reduction."""
        if 'reduction' not in vars(self):
            raise AttributeError()
        return getattr(self.reduction, k)

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        return dict(trafo=self.trafo, reduction=self.reduction)

    def __init__(self, trafo: Callable, reduction: Callable):
        """Init.

        :param trafo: a tuple transform that accepts an unpacked tuple of inputs
            and returns a same-length tuple of outputs
        :param reduction: the loss to apply after application of trafo
        """
        self.trafo: Callable = trafo
        """The tuple transformation to apply before reducing the tuple."""
        self.reduction: Callable[..., torch.Tensor] = reduction
        """Reduction function to reduce a tuple to a single value."""

    def apply_to(self, *inps: torch.Tensor) -> torch.Tensor:
        """Apply the transformation and the the reduction."""
        return self.reduction(*self.trafo(*inps))

    def __radd__(self, other: Optional[TupleTransforms]) -> 'ReduceTuple':
        """Make it possible to prepend trafos via left addition.

        :return: new :py:class:`ReduceTuple` instance with the new
            trafo prepended to the old one
        """
        new_trafo = general_add(other, self.trafo,
                                composition_class=Compose,
                                identity_class=self.IDENTITY_CLASS)
        return ReduceTuple(trafo=new_trafo, reduction=self.reduction)


class _SingleIndexTupleTrafo(Transform):
    """Transformation operating on a single index of a tuple."""

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        return dict(selected_index=self.selected_index, **super().settings)

    def __init__(self, selected_index: int = 0):
        """Init.

        :param selected_index: see :py:attr:`selected_index`
        """
        super().__init__()
        self.selected_index: int = selected_index
        """The index of the sequence value to operate on."""


class FlattenTuple(_SingleIndexTupleTrafo):
    """Return the value of a sequence at selected index."""

    def apply_to(self, *sequence) -> Any:
        """Flatten ``annotations`` dict to its value at the configured key.
        The configured key is stored in :py:attr:`selected_index`.
        """
        return sequence[self.selected_index]


class PackMask(_SingleIndexTupleTrafo):
    """Given a 3-tuple of prediction, target, mask tensors merge mask and target and return a two-tuple.
    This is helpful, if some later operation works on two-tuples only."""

    def apply_to(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.BoolTensor
                 ) -> TensorTwoTuple:
        mask = mask.to(target.dtype)
        return prediction, torch.stack([target, mask.to(target.device)], dim=self.selected_index)


class UnpackMask(_SingleIndexTupleTrafo):
    """Undo a ``PackMask`` operation on the target.
    Accepts a two-tuple of tensors and returns a three-tuple of ``(pred, target, bool_mask)``."""

    def apply_to(self, prediction: torch.Tensor, target_and_mask: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        tm: torch.Tensor = target_and_mask.movedim(self.selected_index, 0)
        assert tm.shape[0] == 2, \
                "Tensor of shape {} should have dimension {} of size {}, but was {}".format(
                    target_and_mask.shape, self.selected_index, 2, tm.shape[0])
        return prediction, tm[0], tm[1].bool()


class ApplyMask(Transform):
    """Given a 3-tuple of input, target, mask apply the mask to the first two.
    Returns a two-tuple of flat tensors."""
    def apply_to(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
                 ) -> TensorTwoTuple:
        mask: torch.BoolTensor = mask.bool()
        return torch.masked_select(prediction, mask.to(prediction.device)), \
            torch.masked_select(target, mask.to(prediction.device))

