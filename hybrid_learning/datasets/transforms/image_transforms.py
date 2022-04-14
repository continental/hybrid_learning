"""Transformations to images.
The images are assumed to be a :py:class:`torch.Tensor` of a
:py:class:`PIL.Image.Image`.
Use :py:class:`torchvision.transforms.ToTensor` to transform
:py:class:`PIL.Image.Image` instances appropriately.
"""
#  Copyright (c) 2022 Continental Automotive GmbH

import abc
from typing import Tuple, Callable, Dict, Any, Optional, Union, Sequence, \
    Mapping, List

import PIL.Image
import numpy as np
import torch
import torch.nn.functional
import torchvision as tv
import torchvision.transforms.functional

from .common import settings_to_repr, Transform
from .encoder import BatchConvOp, BatchIntersectEncode2D, BatchIoUEncode2D, \
    BatchIntersectDecode2D, BatchBoxBloat


def pad_to_ratio(img_t: torch.Tensor, ratio: float = 1.,
                 pad_value: float = 0,
                 ) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """Pad image with constant ``pad_value`` to obtain given
    image size ``ratio``.

    :param img_t: 2D pytorch tensor
    :param ratio: the desired ratio ``(width / height)``
    :param pad_value: constant value to use for padding area
    :return: tensor representing padded 2D image (batch)
    """
    if len(img_t.shape) == 3:
        _, height, width = img_t.shape
    elif len(img_t.shape) == 2:
        height, width = img_t.shape
    else:
        raise ValueError("Wrong image shape ({}); expected 2 or 3 dimensions"
                         .format(img_t.shape))
    # Add padding to image
    pad = padding_for_ratio((height, width), ratio)
    img_t = torch.nn.functional.pad(img_t, list(pad), value=pad_value)

    return img_t, pad


def padding_for_ratio(from_size: Tuple[int, int], to_ratio: float
                      ) -> Tuple[int, int, int, int]:
    """Return the int padding for an image of size ``(height, width)`` to get
    a ``(width / height)`` ratio of ``ratio``.
    Output can be used for :py:func:`torch.nn.functional.pad` pad argument.

    :return: padding as ``(left, right, top, bottom)``"""
    height, width = from_size
    dim_diff_w: int = max(0, int((height * to_ratio) - width))
    dim_diff_h: int = max(0, int((width / to_ratio) - height))
    # (upper / left) padding and (lower / right) padding
    pad_h_l: int = dim_diff_h // 2
    pad_h_r: int = dim_diff_h - pad_h_l
    pad_w_l: int = dim_diff_w // 2
    pad_w_r: int = dim_diff_w - pad_w_l
    # padding put together:
    return pad_w_l, pad_w_r, pad_h_l, pad_h_r


def resize(tens: torch.Tensor, size: Tuple[int, int], mode: str = "bilinear"
           ) -> torch.Tensor:
    """Resize the given tensor assuming it to be a 2D image or batch thereof.
    This is a wrapper around :py:func:`torch.nn.functional.interpolate` which
    takes care of automatic unsqueezing and squeezing of batch and channel
    dimensions assuming 2D images.

    :param tens: the tensor holding a 2D image or batch thereof; dimensions are
        interpreted as ``([[batch,] channel,] height, width)``
    :param size: the new 2D size
    :param mode: the interpolation mode; one of the options for
        :py:func:`torch.nn.functional.interpolate`
    :return: tensor representing resized 2D image (batch)
    """
    if tens.dim() < 2:
        raise ValueError(("Given tensor only is {}D, but was expected to be "
                          ">= 2D (height, width)!").format(tens.dim()))
    if tens.dim() > 4:
        raise ValueError(("Given tensor is {}D, but was expected to be <= 4D "
                          "(batch, channel, height, width)").format(tens.dim()))
    # Take care of unsqueezing batch and channel dimension:
    unsqueeze_dims: int = max(0, 4 - tens.dim())
    for _ in range(unsqueeze_dims):
        tens = tens.unsqueeze(0)

    # actual resizing:
    align_setting = dict(align_corners=False) \
        if 'linear' in mode or 'cubic' in mode else {}
    interp_x = torch.nn.functional.interpolate(
        tens, size=size, mode=mode, **align_setting)

    # Now squeeze unsqueezed dimensions again:
    for _ in range(unsqueeze_dims):
        interp_x = interp_x.squeeze(0)
    return interp_x


class ImageTransform(Transform):
    """Transformations that can be applied to images.
    Images should be given as :py:class:`torch.Tensor` version of a
    :py:class:`PIL.Image.Image` instance.
    The transformation will iteratively descent into mappings and sequences
    to find tensor values to apply the transformation to
    (``None`` values are left untouched).
    An error will be raised if values are found which are neither tensors
    nor ``None``.
    """

    @abc.abstractmethod
    def apply_to(self, img: torch.Tensor) -> torch.Tensor:
        """Application of transformation."""
        raise NotImplementedError()

    def __call__(self,
                 img: Optional[Union[torch.Tensor,
                                     Dict[Any, torch.Tensor],
                                     Sequence[torch.Tensor]]]
                 ) -> Optional[Union[torch.Tensor,
                                     Dict[Any, torch.Tensor],
                                     Sequence[torch.Tensor]]]:
        """Application of transformation."""
        if img is None:
            return img
        # Recursion instructions:
        if isinstance(img, Sequence) and not isinstance(img, str):
            return tuple([self.__call__(val) for val in img])
        if isinstance(img, Mapping):
            return {key: self.__call__(val) for key, val in img.items()}
        return self.apply_to(img)


class RecursiveLambda(ImageTransform):
    """Generic lambda transformation that applies the given function
    with the standard :py:class:`ImageTransform` recursion.
    The same caveats hold as for
    :py:class:`~hybrid_learning.datasets.transforms.common.Lambda`.
    """

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        return dict(fun=self.fun)

    def __repr__(self) -> str:
        return settings_to_repr(self, dict(
            fun=(self.fun.__name__
                 if hasattr(self.fun, "name") else repr(self.fun))))

    def __init__(self, fun: Callable[[torch.Tensor], torch.Tensor]):
        """Init.

        :param fun: the function to apply on call
        """
        self.fun: Callable[[torch.Tensor], torch.Tensor] = fun
        """The function to apply on call."""

    def apply_to(self, img: torch.Tensor) -> torch.Tensor:
        """Application of the lambda."""
        return self.fun(img)


class Resize(ImageTransform):
    """Simple resize.
    The padding value is black.
    Internally, :py:func:`resize` is used.

    .. note::
        Depending on the mode, the used interpolation can cause
        overshooting values larger/smaller than the previous
        minimum/maximum value.
        Ensure to catch such behavior if necessary by using
        :py:func:`torch.clamp`.
    """

    @property
    def settings(self):
        """Settings to reproduce the instance."""
        return dict(img_size=self.img_size, interpolation=self.interpolation)

    def __init__(self, img_size: Tuple[int, int],
                 interpolation: str = "bilinear", force_type: bool = False):
        """Init.

        :param img_size: see :py:attr:`img_size`
        :param interpolation: see :py:attr:`interpolation`
        :param force_type: see :py:attr:`force_type`
        """
        self.img_size: Tuple[int, int] = img_size
        """Image target size as ``(height, width)``."""
        self.interpolation: str = interpolation
        """Interpolation mode to use for the resizing.
        See :py:func:`resize`."""
        self.force_type: bool = force_type
        """Whether to raise in case the input is no tensor or
        to silently skip the transformation.
        If set to ``False``, one can silently skip floats and other types."""

    def apply_to(self, img: torch.Tensor) -> torch.Tensor:
        """Resize ``img`` to the configured image size.
        See also :py:attr:`img_size`."""
        if not isinstance(img, torch.Tensor):
            if self.force_type:
                raise TypeError("Can only resize images encoded as torch.Tensor but got img of type {}"
                                .format(type(img)))
            return img
        img = resize(img, size=self.img_size, mode=self.interpolation)
        return img


class PadAndResize(Resize):
    """Transformation that pads an image to a given ratio and then
    resizes it to fixed size.
    This is especially suitable if going from larger image dimensions to
    smaller ones.
    For the other way round, consider first scaling, then padding.
    For further details see super class.
    """

    def apply_to(self, img: torch.Tensor) -> torch.Tensor:
        """Pad ``img`` to square, then resize it to the
        configured image size.
        See also :py:attr:`~Resize.img_size`."""
        img = pad_and_resize(img, img_size=self.img_size, interpolation=self.interpolation)
        return img


def pad_and_resize(img: torch.Tensor, img_size: Tuple[int, int],
                   interpolation: str = "bilinear"):
    """Pad and resize an image.
    For details see :py:class:`PadAndResize`."""
    img = pad_to_ratio(img, img_size[1] / img_size[0])[0]
    img = resize(img, size=img_size, mode=interpolation)
    return img


class Threshold(ImageTransform):
    """Threshold tensors and set new values below and/or above the threshold.
    The operation is:

    .. code-block: python

        x = val_low_class if x <= post_target_thresh else val_high_class

    Each of the thresholds :py:attr:`val_low_class` and
    :py:attr:`val_high_class` can also be set to ``None``,
    in which case``x`` is used instead.
    Set values to both to obtain a binarizing operation.

    .. note::
        :py:attr:`val_low_class` needs *not* to be lower than
        :py:attr:`val_high_class`, so one can also invert binary masks with
        this.
    """

    def __init__(self, threshold: Union[float, torch.Tensor] = 0.5,
                 val_low_class: Optional[Union[float, torch.Tensor]] = 0.,
                 val_high_class: Optional[Union[float, torch.Tensor]] = 1.):
        """Init.

        :param threshold: the threshold that defines the border between low
            and high class
        :param val_high_class: the value to which to set entries from high class
        :param val_low_class: the value to which to set entries from low class
        """
        super().__init__()
        self.threshold: Union[float, torch.Tensor] = threshold
        """Threshold by which to decide the class;
        low class if ``x<=post_target_thresh``, else high"""
        self.val_low_class: Optional[Union[float, torch.Tensor]] = \
            val_low_class
        """Value to set the low class to.
        If set to ``None``, the input value is used."""
        self.val_high_class: Optional[Union[float, torch.Tensor]] = \
            val_high_class
        """Value to set the high class to.
        If set to ``None``, the input value is used."""

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce instance."""
        settings = dict(threshold=self.threshold)
        if self.val_low_class != 0.:
            settings['val_low_class'] = self.val_low_class
        if self.val_high_class != 1.:
            settings['val_high_class'] = self.val_high_class
        return settings

    def apply_to(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Binarize ``input_tensor`` according to the settings.
        In case any of this instances settings are tensors, these are
        moved to the same device as ``input_tensor``."""
        # region Value checks and default values
        if not isinstance(input_tensor, torch.Tensor):
            raise ValueError(("input_tensor must be of type torch.Tensor, but "
                              "was {}").format(type(input_tensor)))

        val_low_class: Union[torch.Tensor, float] = self.val_low_class \
            if self.val_low_class is not None else input_tensor
        val_high_class: Union[torch.Tensor, float] = self.val_high_class \
            if self.val_high_class is not None else input_tensor

        # ensure all tensors are on the same device:
        if isinstance(self.threshold, torch.Tensor):
            self.threshold = self.threshold.to(input_tensor.device)
        if isinstance(val_high_class, torch.Tensor):
            val_high_class = val_high_class.to(input_tensor.device)
        if isinstance(val_low_class, torch.Tensor):
            val_low_class = val_low_class.to(input_tensor.device)
        # endregion

        return torch.where((input_tensor.float() > self.threshold),
                           torch.as_tensor(val_high_class, dtype=torch.float),
                           torch.as_tensor(val_low_class, dtype=torch.float))

class Binarize(Threshold):
    """Simple class for binarizing tensors into high and low class values.
    This is an alias for :py:class:`Threshold`. See there for details."""


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

    def apply_to(self, img: torch.Tensor) -> torch.Tensor:
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


class BatchWiseImageTransform(ImageTransform, abc.ABC):
    """Wrap a transformation operating on a batch of masks to also work on
    single masks."""

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        return dict(batch_wise=self.batch_wise) if self.batch_wise else {}

    def __init__(self, batch_wise: bool = False):
        self.batch_wise: bool = batch_wise
        """Whether to assume a batch of masks is given (``True``) or a
        single mask (``False``)."""

    def apply_to(self, mask: torch.Tensor) -> torch.Tensor:
        """Apply trafo to the mask (either considered as batch of mask or
        single mask)."""
        if not self.batch_wise:
            mask: torch.Tensor = mask.unsqueeze(0)
        mask = self.apply_to_batch(mask)
        if not self.batch_wise:
            return mask.squeeze(0)
        return mask

    @abc.abstractmethod
    def apply_to_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Batch-wise transformation."""
        raise NotImplementedError()


class AsBatch(BatchWiseImageTransform):
    """Ensure that the given transformation is fed with a batch of inputs.
    :py:attr:`~BatchWiseImageTransform.batch_wise` determines whether
    inputs are assumed to already be batches or not.
    The output is the same as the input (batch or not)."""

    def __init__(self, trafo: Callable[[torch.Tensor], torch.Tensor],
                 batch_wise: bool = False):
        super().__init__(batch_wise=batch_wise)
        self.trafo: Callable[[torch.Tensor], torch.Tensor] = trafo
        """The transformation that requires batch-wise input."""

    def apply_to_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Feed the batch to trafo."""
        return self.trafo(batch)


class ToFixedDims(ImageTransform):
    """Squeeze or unsqueeze a tensor to obtain specified number of
    dimensions."""

    def __init__(self, dims: int):
        assert dims > 0
        self.dims: int = dims

    def apply_to(self, img: torch.Tensor) -> torch.Tensor:
        """Squeezing or unsqueezing."""
        while len(img.size()) > self.dims:
            if img.size()[0] > 1:
                raise ValueError(("Cannot squeeze first dimension in tensor of "
                                  "size {} towards {} dimensions.")
                                 .format(img.size(), self.dims))
            img = img.squeeze(0)
        while len(img.size()) < self.dims:
            img = img.unsqueeze(0)
        return img


class WithThresh(BatchWiseImageTransform):
    # pylint: disable=line-too-long
    """Wrap a batch transformation with binarizing (and unsqueezing) before
    and after.

    The transformation should accept a tensor holding a masks (respectively a
    batch of masks if
    :py:attr:`~hybrid_learning.datasets.transforms.image_transforms.BatchWiseImageTransform.batch_wise`
    is ``True``) and return a transformed batch.
    If given, ``pre_thresh`` is applied before, and
    ``post_thresh`` after the transformation.
    The transformation is assumed to require a batch of masks, so if
    :py:attr:`~hybrid_learning.datasets.transforms.image_transforms.BatchWiseImageTransform.batch_wise`
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
            :py:attr:`~hybrid_learning.datasets.transforms.image_transforms.BatchWiseImageTransform.batch_wise`
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

        super().__init__(batch_wise=batch_wise)

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
        settings = super().settings
        settings['trafo'] = self.trafo
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
        return settings

    def apply_to_batch(self, masks: torch.Tensor) -> torch.Tensor:
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
        modified_masks: torch.Tensor = self.trafo(masks)
        if self.post_thresholder is not None:
            modified_masks = self.post_thresholder(modified_masks)
        return modified_masks


class ToBBoxes(BatchWiseImageTransform):
    """Treat pixels of given mask as scores of constant-size bounding boxes,
    and return a mask with the non-max-suppressed bounding boxes."""

    def __init__(self,
                 bbox_size: Tuple[int, int],
                 iou_threshold: float = 0.5,
                 batch_wise: bool = False
                 ):
        """Init.

        :param bbox_size: see :py:attr:`bbox_size`
        :param iou_threshold: see :py:attr:`iou_threshold`
        """
        super().__init__(batch_wise=batch_wise)

        self.iou_threshold: float = iou_threshold
        """The threshold for the intersection over union
        between two bounding boxes above which the lower-scored box is
        pruned.
        See also :py:func:`torchvision.ops.nms`."""

        self.bloater: BatchBoxBloat = BatchBoxBloat(kernel_size=bbox_size)
        """The bloating operation used to create a mask with bounding boxes
        from anchors and scores."""

    @property
    def bbox_size(self) -> Tuple[int, int]:
        """The constant size to be assumed for all bounding boxes
        in pixels. Give as ``(height, width)``."""
        return self.bloater.kernel_size

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        iou_info = dict(iou_threshold=self.iou_threshold) \
            if self.iou_threshold != 0.5 else {}
        return dict(bbox_size=self.bbox_size, **iou_info, **super().settings)

    def apply_to_batch(self, score_masks: torch.Tensor) -> torch.Tensor:
        """Bloat the ``score_masks`` to a mask of non-max-suppressed bounding
        boxes.
        Each pixel in ``score_masks`` should represent the score of a bounding
        box of fixed size anchored at this pixel.
        The box size is derived from :py:attr:`bbox_size`.
        ``score_masks`` should be a mask of size ``(..., height, width)``.
        For non-max-suppression of the bounding boxes,
        :py:func:`torchvision.ops.nms` is used.

        :return: a mask of the same size as ``mask`` with each anchor
            in ``mask`` bloated to a bounding box filled with the score value;
            for overlapping boxes, the higher scored one is up front
        """
        # Some pylint issues with coordinate naming and torch.tensor:
        # pylint: disable=not-callable
        # pylint: disable=invalid-name
        if len(score_masks.size()) < 3:
            raise ValueError(
                ("Given batch of masks has size {} of dimension {} < 3"
                 ).format(score_masks.size(), len(score_masks.size())))
        if len(score_masks.size()) == 3:
            score_masks = score_masks.unsqueeze(1)  # add channel dimension

        # Box dimensions and offsets:
        bbox_h, bbox_w = self.bbox_size
        # else (round(self.bbox_size[0] * score_masks.size()[-2]),
        #       round(self.bbox_size[1] * score_masks.size()[-1]))
        top, left = bbox_h // 2, bbox_w // 2
        bottom, right = bbox_h - top, bbox_w - left

        # Prepare NMS input: Anchors to boxes
        scores: torch.Tensor = score_masks.view((-1,))
        _boxes: List[List[int]] = []
        _batch_idxs: List[int] = []
        # box in mask b of batch centered at (x, y) has index
        # i = b * (mask_height * mask_width) + y * mask_width + x
        for batch_idx in range(score_masks.size()[0]):
            for y in range(score_masks.size()[-2]):
                for x in range(score_masks.size()[-1]):
                    _boxes.append([x - left, y - top, x + right, y + bottom])
                    _batch_idxs.append(batch_idx)
        boxes_t: torch.Tensor = torch.tensor(_boxes, dtype=score_masks.dtype)
        batch_idxs_t: torch.Tensor = torch.tensor(_batch_idxs, dtype=torch.int)

        # NMS: Collect idxs of boxes (resp. box centers) to keep
        keep: torch.Tensor = tv.ops.batched_nms(
            boxes=boxes_t, scores=scores, idxs=batch_idxs_t,
            iou_threshold=self.iou_threshold)
        # To determine the mask center corresponding to an entry in keep_idxs:
        # keep_idx = batch_idx * (mask_height * mask_width) + y * mask_width + x
        _keep_mask: np.ndarray = np.zeros(score_masks.size(), dtype=np.float)
        for keep_idx in keep:
            batch_idx: int = keep_idx // (score_masks.size()[-1] *
                                          score_masks.size()[-2])
            assert batch_idx == batch_idxs_t[keep_idx]
            # xy_idx = y * mask_width + x
            xy_idx: int = keep_idx % (score_masks.size()[-1] *
                                      score_masks.size()[-2])
            y: int = xy_idx // score_masks.size()[-1]
            x: int = xy_idx % score_masks.size()[-1]
            _keep_mask[batch_idx, ..., y, x] = 1
        keep_mask_t: torch.Tensor = torch.from_numpy(_keep_mask)

        # Set all scores of boxes to abandon to 0:
        nms_score_mask: torch.Tensor = score_masks * keep_mask_t

        # Now bloat each box center to a box filled with its score:
        return self.bloater(nms_score_mask)
        # pylint: enable=not-callable
        # pylint: enable=invalid-name


class ToTensor(ImageTransform):
    """Turn objects into tensors or move tensors to given device or dtype.
    The operation avoids copying of data if possible.
    For details see :py:func:`torch.as_tensor`.

    .. note::
        The default return type for :py:class:`PIL.Image.Image` instances is
        a tensor of dtype :py:class:`torch.float` with value range
        in ``[0, 1]``.
    """

    DTYPE_SIZES: Dict[torch.dtype, int] = {
        torch.bool: 1,
        torch.uint8: 8, torch.int8: 8,
        torch.int16: 16, torch.float16: 16, torch.bfloat16: 16,
        torch.int32: 32, torch.float32: 32, torch.complex32: 32,
        torch.int64: 64, torch.float64: 64, torch.complex64: 64,
        torch.complex128: 128,
    }

    def __init__(self, device: Optional[Union[str, torch.device]] = None,
                 dtype: Optional[torch.dtype] = None,
                 sparse: Optional[Union[bool, str]] = None,
                 requires_grad: Optional[bool] = None):
        self.device: torch.device = torch.device(device) if device is not None \
            else ("cpu" if not torch.cuda.is_available() else None)
        """The device to move tensors to."""
        self.dtype: torch.dtype = dtype
        """The dtype created tensors shall have."""
        self.sparse: Optional[bool] = sparse
        """Whether the tensor should be sparse or dense or dynamically choose
        the smaller one (option 'smallest').
        No modification is made if set to ``None``."""
        self.requires_grad: Optional[bool] = requires_grad
        """Whether the new tensor should require grad."""

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings."""
        setts = dict()
        if self.device is not None:
            setts.update(device=self.device)
        if self.dtype is not None:
            setts.update(dtype=self.dtype)
        return setts

    @staticmethod
    def to_sparse(tens: torch.Tensor,
                  device: Optional[Union[torch.device, str]] = None,
                  dtype: Optional[torch.dtype] = None,
                  requires_grad: Optional[bool] = None
                  ) -> torch.sparse.Tensor:
        """Convert dense tensor ``tens`` to sparse tensor.
        Scalars are not sparsified but returned as normal tensors.
        """
        if len(tens.size()) == 0:
            return torch.as_tensor(tens, device=device, dtype=dtype)

        # scalar case
        indices = torch.nonzero(tens).t()
        values = tens[tuple(indices)]
        sparse_tens: torch.sparse.Tensor = torch.sparse_coo_tensor(
            indices, values, size=tens.size(), device=device, dtype=dtype,
            requires_grad=(requires_grad if requires_grad is not None
                           else tens.requires_grad)
        )
        return sparse_tens

    @classmethod
    def is_sparse_smaller(cls, tens):
        r"""Given a tensor, return whether its sparse representation occupies
        less storage.
        Given the size formulas

        .. math::
            \text{sparse size:}\quad
            \text{numel} \cdot d \cdot (d\cdot s_{ind} + s_{val}) \\
            \text{dense size:}\quad
            \text{numel} \cdot s_{val}

        for the size in bit of one index resp. value entry
        :math:`s_{val}, s_{ind}`, the dimension of the tensor :math:`d`,
        the formula whether the sparse representation is better is:

        .. math::
            p < \frac {s_{val}} {d \cdot s_{ind} + s_{val}}

        and the proportion of non-zero elements :math:`p`.
        """
        bitsize_index: int = 64
        bitsize_value: int = cls.DTYPE_SIZES[tens.dtype]
        return ((tens.count_nonzero() / tens.numel()) <
                (bitsize_value / (tens.dim() * bitsize_index + bitsize_value)))

    def apply_to(self, tens: Union[torch.Tensor, np.ndarray, PIL.Image.Image]
                 ) -> Union[torch.Tensor, torch.sparse.Tensor]:
        """Create tensor from ``tens`` with configured device and dtype.
        See :py:attr:`device` and :py:attr:`dtype`."""
        return self.to_tens(tens,
                            device=self.device, dtype=self.dtype,
                            sparse=self.sparse,
                            requires_grad=self.requires_grad)

    @classmethod
    def to_tens(cls, tens: Union[torch.Tensor, np.ndarray, PIL.Image.Image],
                device: Union[str, torch.device] = None,
                dtype: Optional[torch.dtype] = None,
                sparse: Optional[Union[bool, str]] = None,
                requires_grad: Optional[bool] = None):
        """See ``apply_to`` and ``__init__``."""
        if isinstance(tens, PIL.Image.Image):
            tens: torch.Tensor = \
                torchvision.transforms.functional.to_tensor(tens)

        # to correct device and dtype
        tens: torch.Tensor = \
            torch.as_tensor(tens, device=device, dtype=dtype)

        # possibly sparsify
        if sparse and (not tens.is_sparse) and \
                (sparse != 'smallest' or cls.is_sparse_smaller(tens)):
            tens: torch.sparse.Tensor = cls.to_sparse(
                tens, device=device, dtype=dtype,
                requires_grad=requires_grad)

        # explicitly densify
        if not sparse and sparse is not None and tens.is_sparse:
            # bfloat16 cannot be densified in older torch versions:
            if tens.dtype == torch.bfloat16:
                tens = tens.to(torch.float)
            tens: torch.Tensor = tens.to_dense()

        if requires_grad is not None:
            if not requires_grad and tens.requires_grad and not tens.is_leaf:
                tens = tens.detach()
            else:
                tens = tens.requires_grad_(requires_grad)
        return tens


class NoGrad(ImageTransform):
    """Disable ``requires_grad`` for the given tensors."""

    def apply_to(self, tens: torch.Tensor) -> torch.Tensor:
        """Set ``requires_grad`` to ``False`` for ``tens`` in-place."""
        if isinstance(tens, torch.Tensor):
            return tens.detach()
        if isinstance(tens, torch.nn.Module):
            # noinspection PyTypeChecker
            return tens.requires_grad_(False)
        return tens


class ToActMap(ImageTransform):
    """Evaluate a given image by a torch model on the correct device.
    The model should return tensors, e.g. be a
    :py:class:`~hybrid_learning.concepts.models.model_extension.ModelStump`.
    If :py:attr:`device` is given, the parameters of the model
    :py:attr:`act_map_gen` are moved to this device.

    .. warning::
        Ensure moving of the model parameters to a different device
        does not interfere with e.g. optimization of these parameters
        in case :py:attr:`device` is given!
    """

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        return dict(act_map_gen=self.act_map_gen,
                    device=self.device)

    def __init__(self, act_map_gen: torch.nn.Module,
                 device: Optional[Union[str, torch.device]] = None,
                 requires_grad: bool = False):
        """Init.

        :param act_map_gen: model the output of which is interpreted as
            activation maps
        :param device: the device to operate the transformation on
        :param requires_grad: whether the model and the transformation output
            should require gradients
            (this trafo may be unpickleable in combination with cuda usage of
            set to ``True``)
        """
        self.requires_grad: bool = requires_grad
        """Whether to turn gradient tracking on for the transformation
        calculation."""
        self.act_map_gen: torch.nn.Module = \
            act_map_gen.eval().requires_grad_(requires_grad)
        """Callable torch model that accepts and returns a
        :py:class:`torch.Tensor`."""
        self.device: Optional[Union[str, torch.device]] = torch.device(device) \
            if isinstance(device, str) else device
        """If given, the device to move model and image to before evaluation."""

    def apply_to(self, img_t: torch.Tensor) -> torch.Tensor:
        """Collect output of activation map generator for image ``img_t`` as
        input.
        The evaluation of :py:attr:`act_map_gen` on ``img_t`` is conducted
        on :py:attr:`device` if this is set.

        :param img_t: image for which to obtain activation map;
            make sure all necessary transformations are applied
        :return: activation map as :py:class:`torch.Tensor`
        """
        # Run wrapper to obtain intermediate outputs
        with torch.set_grad_enabled(self.requires_grad):
            if self.device is not None:
                self.act_map_gen = self.act_map_gen.to(self.device)
                img_t = img_t.to(self.device)
            # move input to correct device
            elif len(list(self.act_map_gen.parameters())) > 0:
                img_t = img_t.to(next(self.act_map_gen.parameters()).device)

            act_map = self.act_map_gen.eval()(img_t.unsqueeze(0))
            # Squeeze batch dimension
            act_map = act_map.squeeze(0)
        return act_map


class ConvOpWrapper(WithThresh):
    """Base wrapper class to turn convolutional batch operations into single
    mask operations.
    Wraps classes inheriting from
    :py:class:`~hybrid_learning.datasets.transforms.encoder.BatchConvOp`."""

    def __init__(self, trafo: BatchConvOp, **kwargs):
        super().__init__(trafo=trafo, **kwargs)
        self.trafo: BatchConvOp = self.trafo

    @property
    def proto_shape(self) -> np.ndarray:
        """Wrap the
        :py:class:`~hybrid_learning.datasets.transforms.encoder.BatchConvOp.proto_shape`."""
        return self.trafo.proto_shape

    @property
    def kernel_size(self) -> Tuple[int, ...]:
        """Wrap the
        :py:class:`~hybrid_learning.datasets.transforms.encoder.BatchConvOp.kernel_size`."""
        return self.trafo.kernel_size

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings; essentially merged from wrapped encoder and super."""
        return dict(**self.trafo.settings, **super().settings)


class IntersectEncode(ConvOpWrapper):
    """Intersection encode a single mask.
    This is a wrapper around
    :py:class:`~hybrid_learning.datasets.transforms.encoder.BatchIntersectEncode2D`.
    """

    def __init__(self, kernel_size: Tuple[int, int] = None, *,
                 normalize_by: str = 'proto_shape',
                 proto_shape: np.ndarray = None,
                 **thresh_args):
        # pylint: disable=line-too-long
        """Init.

        :param thresh_args: thresholding arguments;
            see :py:class:`~hybrid_learning.datasets.transforms.image_transforms.WithThresh`
        """
        # pylint: enable=line-too-long
        super().__init__(
            trafo=BatchIntersectEncode2D(kernel_size=kernel_size,
                                         proto_shape=proto_shape,
                                         normalize_by=normalize_by),
            **thresh_args)


class IoUEncode(ConvOpWrapper):
    """IoU encode a single mask.
    This is a wrapper around
    :py:class:`~hybrid_learning.datasets.transforms.encoder.BatchIoUEncode2D`.
    """

    def __init__(self,
                 kernel_size: Tuple[int, int], *,
                 proto_shape: np.ndarray = None,
                 smooth: float = None,
                 **thresh_args):
        # pylint: disable=line-too-long
        """Init.

        :param thresh_args: thresholding arguments;
            see :py:class:`~hybrid_learning.datasets.transforms.image_transforms.WithThresh`
        """
        # pylint: enable=line-too-long
        super().__init__(
            trafo=BatchIoUEncode2D(kernel_size=kernel_size,
                                   proto_shape=proto_shape,
                                   **(dict(smooth=smooth) if smooth is not None
                                      else {})),
            **thresh_args)


class IntersectDecode(ConvOpWrapper):
    """IoU encode a single mask.
    This is a wrapper around
    :py:class:`~hybrid_learning.datasets.transforms.encoder.BatchIntersectDecode2D`.
    """

    def __init__(self,
                 kernel_size: Tuple[int, int], *,
                 proto_shape: np.ndarray = None,
                 **thresh_args):
        # pylint: disable=line-too-long
        """Init.

        :param thresh_args: thresholding arguments;
            see :py:class:`~hybrid_learning.datasets.transforms.image_transforms.WithThresh`
        """
        # pylint: enable=line-too-long
        super().__init__(
            trafo=BatchIntersectDecode2D(kernel_size=kernel_size,
                                         proto_shape=proto_shape),
            **thresh_args)
