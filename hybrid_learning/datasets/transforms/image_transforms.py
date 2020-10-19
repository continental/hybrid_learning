"""Transformations to images.
The images are assumed to be a :py:class:`torch.Tensor` of a
:py:class:`PIL.Image.Image`.
Use :py:class:`torchvision.transforms.ToTensor` to transform
:py:class:`PIL.Image.Image` instances appropriately.
"""
#  Copyright (c) 2020 Continental Automotive GmbH

import abc
from typing import Tuple, Callable, Dict, Any, Optional

import PIL.Image
import numpy as np
import torch
import torchvision as tv

from .utils import settings_to_repr


def pad_to_square(img_t: torch.Tensor, pad_value: float = 0
                  ) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """Symmetrically pad image with constant ``pad_value`` to obtain square
    image.

    :param img_t: 2D pytorch tensor
    :param pad_value: constant value to use for padding area
    """
    if len(img_t.shape) == 3:
        _, height, width = img_t.shape
    elif len(img_t.shape) == 2:
        height, width = img_t.shape
    else:
        raise ValueError("Wrong image shape ({}); expected 2 or 3 dimensions"
                         .format(img_t.shape))
    dim_diff: int = np.abs(height - width)
    # (upper / left) padding and (lower / right) padding
    pad1: int = dim_diff // 2
    pad2: int = dim_diff - dim_diff // 2
    # padding put together:
    pad: Tuple[int, int, int, int] = (0, 0, pad1, pad2) \
        if height <= width else (pad1, pad2, 0, 0)
    # Add padding to image
    img_t = torch.nn.functional.pad(img_t, list(pad), value=pad_value)

    return img_t, pad


class ImageTransform(abc.ABC):
    """Transformations that can be applied to images.
    Images should be given as :py:class:`torch.Tensor` version of a
    :py:class:`PIL.Image.Image` instance."""

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        return {}

    def __repr__(self):
        return settings_to_repr(self, self.settings)

    @abc.abstractmethod
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Application of transformation."""
        raise NotImplementedError()


class PadAndResize(ImageTransform):
    """Transformation that pads an image to square dimensions and then
    resizes it to fixed size.

    The padding value is black.
    Internally torchvision transformations are used.
    """

    @property
    def img_size(self) -> Tuple[int, int]:
        """Image target size as :py:class:`PIL.Image.Image` size
        ``(width, height)``."""
        return self._img_size

    @img_size.setter
    def img_size(self, size: Tuple[int, int]):
        self._img_size = size
        self._trafo: Callable[[torch.Tensor], torch.Tensor] = \
            self._create_trafo()

    @property
    def interpolation(self) -> int:
        """Interpolation to use for the resizing.
        Must be one of the :py:mod:`PIL.Image` interpolation constants."""
        return self._interpolation

    @interpolation.setter
    def interpolation(self, interpolation: int):
        self._interpolation = interpolation
        self._trafo: Callable[[torch.Tensor], torch.Tensor] = \
            self._create_trafo()

    # noinspection PyTypeChecker
    def _create_trafo(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Create the transformation to use in :py:attr:`__call__` from
        instance attributes."""
        return tv.transforms.Compose([
            tv.transforms.Lambda(lambda i: pad_to_square(i)[0]),  # add padding
            tv.transforms.ToPILImage(),
            tv.transforms.Resize(self.img_size,
                                 interpolation=self.interpolation),
            tv.transforms.ToTensor()
        ])

    @property
    def settings(self):
        """Settings to reproduce the instance."""
        return dict(img_size=self.img_size, interpolation=self.interpolation)

    def __init__(self, img_size: Tuple[int, int],
                 interpolation: int = PIL.Image.BILINEAR):
        """Init.

        :param img_size: see :py:attr:`img_size`
        :param interpolation: see :py:attr:`interpolation`
        """
        self._img_size: Tuple[int, int] = img_size
        """Internal storage of :py:attr:`img_size`."""
        self._interpolation: int = interpolation
        """Internal storage of :py:attr:`interpolation`."""
        self._trafo: Callable[[torch.Tensor], torch.Tensor] = \
            self._create_trafo()
        """The transformation to apply derived from the instance attributes."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Pad ``img`` to square, then resize it to the
        configured image size.
        See also :py:attr:`img_size`."""
        return self._trafo(img)


class Binarize(ImageTransform):
    """Simple class for binarizing tensors into high and low class values.
    The operation is:

    .. code-block: python

        x = val_low_class if x <= post_target_thresh else val_high_class

    .. note::
        :py:attr:`val_low_class` needs *not* to be lower than
        :py:attr:`val_high_class`, so one can also invert binary masks with
        this.
    """

    def __init__(self, threshold: float = 0.5,
                 val_low_class: float = 0., val_high_class: float = 1.):
        """Init.

        :param threshold: the threshold that defines the border between low
            and high class
        :param val_high_class: the value to which to set entries from high class
        :param val_low_class: the value to which to set entries from low class
        """
        super(Binarize, self).__init__()
        self.threshold = threshold
        """Threshold by which to decide the class;
        low class if ``x<=post_target_thresh``, else high"""
        self.val_low_class = val_low_class
        """Value to set the low class to."""
        self.val_high_class = val_high_class
        """Value to set the high class to."""

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce instance."""
        settings = dict(threshold=self.threshold)
        if self.val_low_class != 0.:
            settings['val_low_class'] = self.val_low_class
        if self.val_high_class != 1.:
            settings['val_high_class'] = self.val_high_class
        return settings

    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Binarize ``input_tensor`` according to the settings."""
        if not isinstance(input_tensor, torch.Tensor):
            raise ValueError(("input_tensor must be of type torch.Tensor, but "
                              "was {}").format(type(input_tensor)))
        low_class: torch.tensor = (input_tensor <= self.threshold).float()
        high_class: torch.tensor = (input_tensor > self.threshold).float()
        return (low_class * self.val_low_class +
                high_class * self.val_high_class)


class BinarizeByQuantile(ImageTransform):
    """Set all but the given highest number of pixels / q-th quantile
    in an image to zero, rest to 1.
    Mind for RGB images: A pixel here means a pixel in one channel."""

    def __init__(self, quantile: float = None, num_pixels: int = None):
        """Init.

        :param quantile: quantile of pixels to set to 1, rest is set to 0;
            overridden by ``num_pixels``
        :param num_pixels: number of pixels with highest value to set to one,
            rest is set to 0
        """
        if num_pixels is None and quantile is None:
            raise ValueError("either num_pixels or quantile must be given")
        self.num_pixels: float = num_pixels
        """Number of pixels with highest values to set to one."""
        self.quantile: float = quantile
        """Quantile of pixels to set to one, rest is set to 0;
        overridden by :py:attr:`num_pixels`"""

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        if self.num_pixels is not None:
            return dict(num_pixels=self.num_pixels)
        return dict(quantile=self.quantile)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Binarize ``img`` by setting a quantile or number of pixels to one,
        the rest to 0.
        See :py:attr:`quantile` respectively :py:attr:`num_pixels`.

        :param img: target tensor to binarize
        """
        img_np: np.ndarray = img.detach().cpu().numpy()
        quantile: float = min(self.num_pixels / img_np.size, 1) \
            if self.num_pixels is not None else self.quantile
        thresh: float = np.quantile(img_np, 1 - quantile)
        img = (img > thresh).float()
        return img


class WithThresh(ImageTransform):
    # pylint: disable=line-too-long
    """Wrap a batch transformation with binarizing (and unsqueezing) before
    and after.

    The transformation should accept a tensor holding a masks (respectively a
    batch of masks if
    :py:attr:`~hybrid_learning.datasets.transforms.image_transforms.WithThresh.batch_wise`
    is ``True``) and return a transformed batch.
    If given, ``pre_thresh`` is applied before, and
    ``post_thresh`` after the transformation.
    The transformation is assumed to require a batch of masks, so if
    :py:attr:`~hybrid_learning.datasets.transforms.image_transforms.WithThresh.batch_wise`
    is ``False``, the missing batch dimension is handled.
    Thus, this wrapper can also be used to turn a batch operation
    into one on single masks.
    """
    # pylint: enable=line-too-long

    def __init__(self,
                 trafo: Callable[[torch.Tensor], torch.Tensor],
                 pre_thresh: Optional[float] = None,
                 post_thresh: Optional[float] = None,
                 batch_wise: bool = False,
                 pre_low_class: float = 0., pre_high_class: float = 1.,
                 post_low_class: float = 0., post_high_class: float = 1.,
                 ):
        # pylint: disable=line-too-long
        """Init.

        :param trafo: the transformation instance to wrap
        :param pre_thresh: if not ``None``, the tensors to be modified are
            binarized to 0 and 1 values with threshold ``pre_thresh`` before
            modification
        :param post_thresh: if not ``None``, the tensors to be modified are
            binarized to 0 and 1 values with threshold ``post_thresh`` after
            modification
        :param batch_wise: see
            :py:attr:`~hybrid_learning.datasets.transforms.image_transforms.WithThresh.batch_wise`
        :param pre_high_class: value to set items to that exceed ``pre_thresh``
        :param pre_low_class: value to set items to that are below
            ``pre_thresh``
        :param post_high_class: value to set items to that exceed
            ``post_thresh``
        :param post_low_class: value to set items to that are below
            ``post_thresh``
        """
        # pylint: enable=line-too-long
        # Value checks:
        if not callable(trafo):
            raise ValueError("trafo is not callable, but of type {}"
                             .format(type(trafo)))

        super(WithThresh, self).__init__()

        self.batch_wise: bool = batch_wise
        """Whether to assume a batch of masks is given (``True``) or a
        single mask (``False``)."""
        self.trafo: Callable[[torch.Tensor], torch.Tensor] = trafo
        """Modifier (en- or decoder) module that is used for modifications."""
        self.pre_thresholder: Optional[Binarize] = None \
            if pre_thresh is None else \
            Binarize(threshold=pre_thresh,
                     val_low_class=pre_low_class,
                     val_high_class=pre_high_class)
        """Binarizing transformation applied to targets before IoU encoding
        if not ``None``."""
        self.post_thresholder: Optional[Binarize] = None \
            if post_thresh is None else \
            Binarize(threshold=post_thresh,
                     val_low_class=post_low_class,
                     val_high_class=post_high_class)
        """Binarizing transformation applied to targets after IoU encoding if
        not ``None``."""

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce instance."""
        settings = {}
        if self.pre_thresholder is not None:
            settings['pre_thresh'] = self.pre_thresholder.threshold
            if self.pre_thresholder.val_low_class != 0.:
                settings['pre_val_low'] = self.pre_thresholder.val_low_class
            if self.pre_thresholder.val_high_class != 1.:
                settings['pre_val_high'] = self.pre_thresholder.val_high_class

        if self.post_thresholder is not None:
            settings['post_thresh'] = self.post_thresholder.threshold
            if self.post_thresholder.val_low_class != 0.:
                settings['post_val_low'] = self.post_thresholder.val_low_class
            if self.post_thresholder.val_high_class != 1.:
                settings['post_val_high'] = self.post_thresholder.val_high_class

        if self.batch_wise:
            settings['batch_wise'] = self.batch_wise

        return settings

    def __call__(self, masks: torch.Tensor) -> torch.Tensor:
        """Forward method in which to apply the trafo and thresholding.

        Pre-threshold, modify, and post-threshold given mask(s).
        The thresholding is applied, if the corresponding
        :py:attr:`pre_thresholder` / :py:attr:`post_thresholder`
        is not ``None``.
        If :py:attr:`batch_wise` is ``False``, it is assumed a single mask
        was given (no batch dimension).

        :param masks: :py:class:`torch.Tensor` of shape
            ``([batch_size,] 1, width, height)`` holding masks for one batch.
        :return: the modified and thresholded masks
        """

        if self.pre_thresholder is not None:
            masks = self.pre_thresholder(masks)

        if not self.batch_wise:
            masks: torch.Tensor = masks.unsqueeze(0)
        modified_masks: torch.Tensor = self.trafo(masks)
        if not self.batch_wise:
            modified_masks = masks.squeeze(0)

        if self.post_thresholder is not None:
            modified_masks = self.post_thresholder(modified_masks)

        return modified_masks


class ToDevice(ImageTransform):
    """Move tensors to given device."""

    def __init__(self, device: torch.device):
        self.device: torch.device = device
        """The device to move tensors to."""

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings."""
        return dict(device=self.device)

    def __call__(self, tens: torch.Tensor):
        """Move ``tens`` tensor to the configured device.
        See :py:attr:`device`."""
        return tens.to(self.device)
