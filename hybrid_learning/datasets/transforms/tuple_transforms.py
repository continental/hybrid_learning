"""Transformations that modify tuples of ``(input, output)`` on call
(or use such trafos).

The tuple transformations are all based on :py:class:`TupleTransforms`.
:py:class:`ReduceTuple` can be used to merge the output tuple of a
:py:class:`TupleTransforms` into one output value.
The :py:class:`Identity` transformation can be used as dummy transformation
where necessary.
"""
#  Copyright (c) 2020 Continental Automotive GmbH

# TODO: tests for tuple transforms

import abc
from typing import Dict, Any, Tuple, Sequence, Callable, List

import torch

from .utils import settings_to_repr


class TupleTransforms(abc.ABC):
    """Base class of tuple transformations.
    To inherit, override :py:attr:`settings` and :py:meth:`__call__`.
    """

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        return {}

    @abc.abstractmethod
    def __call__(self, inp, target) -> Tuple:
        """Apply the transformation to the tuple ``(inp, target)``;
        the output again is a tuple."""
        raise NotImplementedError()

    def __repr__(self):
        return settings_to_repr(self, self.settings)


class Compose(TupleTransforms):
    """Compose several tuple transforms by sequentially executing them."""

    def __init__(self, transforms: Sequence[TupleTransforms]):
        self.transforms: List[TupleTransforms] = list(transforms)
        """List of tuple transformations to apply in order."""

    def append(self, trafo: TupleTransforms):
        """Append transformation to the processing chain.
        """
        self.transforms.append(trafo)

    def insert(self, i: int, trafo: TupleTransforms):
        """Insert ``trafo`` at index ``i`` into the processing chain."""
        self.transforms.insert(i, trafo)

    @property
    def settings(self):
        """Settings to reproduce the instance."""
        return dict(transforms=self.transforms)

    def __call__(self, inp, target) -> Tuple:
        """Apply all transformations in order.
        Transformations are taken from :py:attr:`transforms`."""
        for trafo in self.transforms:
            inp, target = trafo(inp, target)
        return inp, target


class OneSided(TupleTransforms, abc.ABC):
    """Transformation that applies only to one element of a given tuple."""

    def __init__(self, trafo: Callable):
        """Init.

        :param trafo: the transformation to apply
        """
        self.trafo: Callable = trafo
        """The transformation to apply to just one of the items."""

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        return dict(trafo=self.trafo)

    def __call__(self, inp, target) -> Tuple[Any, Any]:
        raise NotImplementedError()


class OnBothSides(TupleTransforms):
    """Apply a given transformation to both input and target of a tuple in
    parallel."""

    def __init__(self, trafo: Callable):
        """Init.

        :param trafo: the transformation to apply
        """
        self.trafo: Callable = trafo
        """The transformation to apply to both sides of a tuple."""

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        return dict(trafo=self.trafo)

    def __call__(self, inp, target) -> Tuple[Any, Any]:
        """Apply transformation in parallel to both ``inp`` and ``target``."""
        return self.trafo(inp), self.trafo(target)


class OnInput(OneSided):
    """Apply a given one-value transformation only to the input of a two-tuple.
    """

    def __call__(self, inp, target) -> Tuple[Any, Any]:
        return self.trafo(inp), target


class OnTarget(OneSided):
    """Apply a given one-value transformation only to the target of a two-tuple.
    """

    def __call__(self, inp, target) -> Tuple[Any, Any]:
        return inp, self.trafo(target)


class SameSize(TupleTransforms):
    """Given a tuple of input and target image, resize the target to the
    size of the input.
    Both input and target image must be given as :py:class:`torch.Tensor`.
    """

    def __init__(self, interpolation="bilinear", resize_target: bool = True):
        """Init.

        :param interpolation: the interpolation method to use, parameter to
            :py:class:`torch.nn.Upsample`;

            .. note::
                Mind that other methods than ``'nearest'`` will produce
                non-binary outputs.
        """
        self.resize_target: bool = resize_target
        self.interpolation: str = interpolation

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        return dict(interpolation=self.interpolation,
                    resize_target=self.resize_target)

    def __call__(self, inp: torch.Tensor, target: torch.Tensor):
        non_changing = inp if self.resize_target else target
        new_size = (non_changing.size()[-1], non_changing.size()[-2])

        def trafo(tens: torch.Tensor) -> torch.Tensor:
            """The interpolation transformation to apply.
            If necessary, unsqueeze and later squeeze the batch and channel
            dimensions."""
            unsqueeze_dims = max(0, 4 - tens.dim())
            for _ in range(unsqueeze_dims):
                tens = tens.unsqueeze(0)
            interp_x = torch.nn.functional.interpolate(
                tens, size=new_size, mode=self.interpolation,
                align_corners=False
            )
            for _ in range(unsqueeze_dims):
                interp_x = interp_x.squeeze(0)
            return interp_x

        # Resize the correct one
        if self.resize_target:
            target = trafo(target)
        else:
            inp = trafo(inp)
        return inp, target


class Identity(TupleTransforms):
    """Simple identity transformation for example for defaults."""

    def __call__(self, inp, target):
        """Identity operation, nothing changed."""
        return inp, target


class ReduceTuple:
    """Transform ``(input, target)`` tuple and then reduce it to one value
    using a reduction func.
    One example would reduction by a loss function.
    """

    def __init__(self, trafo: TupleTransforms,
                 reduction: Callable[[Any, Any], Any]):
        """Init.

        :param trafo: a tuple transform that accepts a batch input and a batch
            target tensor and yields transformed batch input and batch target
            tensors
        :param reduction: the loss to apply after transformation
        """
        self.trafo = trafo
        """The tuple transformation to apply before reducing the tuple."""
        self.reduction: Callable[[Any, Any], Any] = reduction
        """Reduction function to reduce a tuple to a single value."""

    def __call__(self, inps, targets):
        """Apply the transformation to ``(inps, targets)`` and return loss
        of transformed tuple."""
        return self.reduction(*self.trafo(inps, targets))

    def __repr__(self):
        return "{}(\n    trafo={},\n    reduction={}\n)".format(
            self.__class__.__name__,
            repr(self.trafo),
            repr(self.reduction))
