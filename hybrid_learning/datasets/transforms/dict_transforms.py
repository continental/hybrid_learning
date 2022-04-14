"""Transformations on dict annotations containing scalars and masks.

The annotations that can be processed are supposed to be a dict with string
keys and values of floats or :py:class:`numpy.ndarray` (binary) masks.
"""
#  Copyright (c) 2022 Continental Automotive GmbH

import abc
from typing import Dict, Any, Union, Iterable, Callable, Mapping, Literal, MutableMapping, List

import torch.nn.functional

from .common import Transform
from .image_transforms import resize


class DictTransform(Transform):
    """Basic transformation for dicts.
     This means a callable yielding a dict of a single value."""

    @abc.abstractmethod
    def apply_to(self, annotations: Mapping[str, Any]
                 ) -> Union[Mapping[str, Any], Any]:
        """Call method modifying a given dictionary."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def settings(self):
        """Settings to reproduce the instance."""
        raise NotImplementedError()

    def __call__(self, annotations: Mapping[str, Any]
                 ) -> Union[Mapping[str, Any], Any]:
        """Call method modifying a given dictionary."""
        return self.apply_to(annotations)


class DropAnn(DictTransform):
    """Drop the annotation with given key from the annotations dict."""

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        return dict(drop_key=self.drop_key)

    def __init__(self, drop_key: str):
        """Init.

        :param drop_key: see :py:attr:`drop_key`
        """
        self.drop_key = drop_key
        """Dict key to drop on call."""

    def apply_to(self, annotations: Dict[str, Any]) -> Dict[str, Any]:
        """Drop the item at configured key from ``annotations``.
        The configured key is stored in :py:attr:`drop_key`."""
        annotations.pop(self.drop_key)
        return annotations


class RestrictDict(DictTransform):
    """Restrict the annotation dictionary to the annotation items with
    featuring one of the selected keys."""

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        return dict(selected_keys=self.selected_keys)

    def __init__(self, selected_keys: Iterable[str]):
        """Init.

        :param selected_keys: see :py:attr:`selected_keys`
        """
        self.selected_keys: Iterable[str] = selected_keys
        """The keys to restrict the dict to."""

    def apply_to(self, annotations: Dict[str, Any]) -> Dict[str, Any]:
        """Restrict the annotation dict to the selected keys.
        Selected keys are stored in :py:attr:`selected_keys`."""
        return {key: annotations[key] for key in self.selected_keys}


class FlattenDict(DictTransform):
    """Return the value of the annotations dict at selected key."""

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        return dict(selected_key=self.selected_key)

    def __init__(self, selected_key: str):
        """Init.

        :param selected_key: see :py:attr:`selected_key`
        """
        self.selected_key: str = selected_key
        """The key of the annotation value to return."""

    def apply_to(self, annotations: Dict[str, Any]) -> Any:
        """Flatten ``annotations`` dict to its value at the configured key.
        The configured key is stored in :py:attr:`selected_key`.
        """
        return annotations[self.selected_key]


class OnValues(DictTransform):
    """Perform a transformation on all values of a dict."""

    def __init__(self, trafo: Callable[[Dict[str, Any]], Dict[str, Any]]):
        self.trafo: Callable[[Dict[str, Any]], Dict[str, Any]] = trafo
        """Transformation to be applied to all values of a dict."""

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        return dict(trafo=self.trafo)

    def apply_to(self, annotations: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transformation in to all values of the ``annotations`` dict."""
        return {k: self.trafo(v) for k, v in annotations.items()}


class SameSizeTensorValues(DictTransform):
    """Up- or down-scale the tensor mask values of a dictionary to all have the same size.
    The :py:attr:`mode` determines whether and how it is up- or downscaled to the
    largest/smallest occurring size.
    Mask tensor sizes are interpreted as ``([batch_size[, num_channels],] height, width)``.
    In case the mask has more than one channel, each channel is resized separately.
    """

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        return dict(mode=self.mode)

    def __init__(self, mode: Literal['up_bilinear', 'down_max'] = 'up_bilinear'):
        super().__init__()
        self.mode: Literal['up_bilinear', 'down_max'] = mode or 'up_bilinear'
        """Up- or downscaling mode.
        Allowed values:

        - ``up_{interpolation}``: upscaling to largest size using ``{interpolation}``;
            the interpolation mode must be one of the options for :py:func:`torch.nn.functional.interpolate`;
            equivalent to ``OnValues(Resize(max_size))``
        - ``down_max``: down-scaling via max-pooling
        """

    def apply_to(self, annotations: MutableMapping[str, Any]
                 ) -> Union[Mapping[str, Any], Any]:
        """Up- or downscale the values of annotations to the same size according to ``mode``.
        Note that for efficiency reasons annotations is modified in-place!"""
        to_resize: Dict[str, torch.Tensor] = {key: tens for key, tens in annotations.items()
                                              if isinstance(tens, torch.Tensor)}
        if len(to_resize) <= 1:  # nothing to do?
            return annotations

        # Get the target size
        all_sizes = sorted([mask.size()[-2:] for mask in to_resize.values()], key=lambda hw: hw[0] * hw[1])
        target_size = all_sizes[0] if self.mode.startswith('down') else all_sizes[-1]
        # Mapping of list of masks of 1 channel
        to_resize: Dict[str, List[torch.Tensor]] = {
            key: ([tens] if len(tens.size()) != 4 or tens.size()[1] == 1
                  else [tens[:, i].unsqueeze(1) for i in range(tens.size()[1])])  # unravelled channels
            for key, tens in to_resize.items()
            if list(tens.size()[-2:]) != list(target_size)}

        # Do the resizing
        for key, masks in to_resize.items():
            if self.mode.startswith('up_'):
                resized_masks = [resize(mask, target_size, mode=self.mode.split('up_', maxsplit=1)[1])
                                 for mask in masks]
            elif self.mode == 'down_max':
                if masks[0].size()[-2] % target_size[0] != 0 or masks[0].size()[-1] % target_size[1] != 0:
                    raise ValueError(("Found incompatible mask shapes in annotations: "
                                      "Cannot max_pool size {} to target size {}.\nAnnotations:{}"
                                      ).format(masks[0].size(), target_size, annotations))
                resized_masks = [torch.nn.functional.max_pool2d(mask, kernel_size=[mask.size()[-2] // target_size[0],
                                                                                   mask.size()[-1] // target_size[1]])
                                 for mask in masks]
            else:
                raise ValueError("Unknown mode {}".format(self.mode))
            annotations[key] = resized_masks[0] if len(resized_masks) == 1 else \
                torch.stack(resized_masks, dim=1).squeeze(dim=2)
        return annotations
