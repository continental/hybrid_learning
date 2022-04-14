"""Batch-wise encoding and decoding operations for binary and non-binary masks.

The basic operations are the batch encoder based on :py:class:`BatchWindowOp`.
Batch-wise operations on images can be wrapped by
:py:class:`hybrid_learning.datasets.transforms.image_transforms.BatchWiseImageTransform`
to work on either batches or single masks.
"""

#  Copyright (c) 2022 Continental Automotive GmbH

# The pytorch forward method should be overridden with a more specific
# signature, thus:
# pylint: disable=arguments-differ
import abc
from typing import Tuple, Dict, Any, Sequence

import numpy as np
import torch
from .common import settings_to_repr


class BatchWindowOp(abc.ABC, torch.nn.Module):
    """Base class for encoder that use windowing operations.
    E.g. convolutions or pooling."""

    AREA_DIMS: Tuple[int] = (2, 3)
    """Indices of axes in which the image area is defined.
    The targets are expected to have size with the following dimensions:

    - ``size()[0]``: batch_dim
    - ``size()[1]``: channels
    - ``size()[AREA_DIMS]``: size dimensions of one filter's activation map,
      e.g. ``(2, 3)`` for 2D and ``(2, 3, 4)`` for 3D
    """

    @property
    def kernel_size(self) -> Tuple[int, ...]:
        """The kernel size of the window."""
        raise NotImplementedError()

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        return dict(kernel_size=self.kernel_size)

    def __repr__(self):
        """Representation based on this instances settings."""
        return settings_to_repr(self, self.settings)

    def _validate_masks(self, masks: torch.Tensor):
        """Raise if ``masks`` are invalid with informative error message.

        :raises: :py:exc:`ValueError`
        """
        # masks must be tensor
        if not isinstance(masks, torch.Tensor):
            raise ValueError("masks must be given as torch.Tensor, but was {}"
                             .format(type(masks)))
        # masks size check
        if len(masks.size()) != len(self.AREA_DIMS) + 2 or masks.size()[1] != 1:
            raise ValueError(("masks must have a size of len {} of the form "
                              "(batch_size, 1, <img info, e.g. "
                              "*(width, height)>), but was of size {}"
                              ).format(len(self.AREA_DIMS) + 2, masks.size()))
        if masks.size()[-2] < self.kernel_size[-2] or \
                masks.size()[-1] < self.kernel_size[-1]:
            raise ValueError(("Input masks (size {}) are smaller than kernel "
                              "size ({})")
                             .format(masks.size(), self.kernel_size))

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        """Wrapper for the convolutional operation on batch of masks.
        Validates the masks and ensures the encoder is on the correct device.

        :param masks: :py:class:`torch.Tensor` of shape
            ``(batch_size, 1, width, height)`` holding masks for one batch.
        """
        self._validate_masks(masks)
        masks = masks.float()
        self.to(masks.device)
        return self.conv_op(masks)

    @abc.abstractmethod
    def conv_op(self, masks: torch.Tensor) -> torch.Tensor:
        """The convolutional operation on the masks (without validation).

        :param masks: :py:class:`torch.Tensor` of shape
            ``(batch_size, 1, width, height)`` holding masks for one batch.
        """
        raise NotImplementedError()


class BatchBoxBloat(BatchWindowOp):
    """Bloat single pixels to full boxes, always choosing the maximum box to
    be up front.
    This is a max-pooling operation."""

    @property
    def kernel_size(self) -> Tuple[int, int]:
        """The max-pooling operation kernel size."""
        return self.nms_bloating.kernel_size

    def __init__(self, kernel_size: Tuple[int, int]):
        """Init.

        :param kernel_size: the window size to which each pixel shall
            be blown up
        """
        super().__init__()

        # Padding
        # Beware: The padding for ZeroPad2d has crude specification:
        # 1. width pad, 2. height pad
        self.padding = torch.nn.ZeroPad2d(
            padding=same_padding((kernel_size[1], kernel_size[0])))
        """Padding to obtain same size as input after
        non-max-suppression bloating."""

        # NMS
        self.nms_bloating: torch.nn.MaxPool2d = torch.nn.MaxPool2d(
            kernel_size=kernel_size, stride=(1, 1))
        """The actual bloating operation.
        Bloats each pixel to a window of size
        :py:attr:`BatchBoxBloat.kernel_size`,
        then overlays them sorted by decreasing pixel value
        (i.e. the whitest box is up front).
        This essentially is a max-pooling operation."""

    def conv_op(self, masks: torch.Tensor) -> torch.Tensor:
        """Apply the bloating to the pixels."""
        masks = self.padding(masks)
        masks = self.nms_bloating(masks)
        return masks


class BatchPeakDetection(BatchBoxBloat):
    """Keep only peak points, i.e. ones that take the maximum value within a
    window around them.
    The window is given by :attr:`~BatchBoxBloat.kernel_size`.
    The returned mask has all non-peaks set to 0.
    This is a max-pooling operation."""

    def conv_op(self, masks: torch.Tensor) -> torch.Tensor:
        """Apply the peak filtering."""
        # bloat the masks using max-pooling
        bloated_masks: torch.Tensor = super().conv_op(masks)
        binary_peak_masks: torch.Tensor = (bloated_masks == masks).float()
        return masks * binary_peak_masks


class BatchConvOp(BatchWindowOp):
    """Base class for encoder that use convolution operations."""

    @property
    @abc.abstractmethod
    def proto_shape(self) -> np.ndarray:
        """The kernel associated with the convolutional masking operation."""
        raise NotImplementedError()

    @property
    def kernel_size(self) -> Tuple[int, ...]:
        """The kernel size of the proto-type shape.
        (See :py:attr:`proto_shape`)."""
        return self.proto_shape.shape

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        # If proto shape is default from kernel_size, only provide kernel_size
        if np.allclose(self.proto_shape, 1):
            return dict(kernel_size=self.kernel_size)
        return dict(proto_shape=self.proto_shape.tolist())

    @classmethod
    def _to_valid_proto_shape(cls, proto_shape: np.ndarray = None,
                              kernel_size: Tuple[int, int] = None):
        """Validate or create :py:attr`proto_shape`.

        :raises: :py:exc:`ValueError` if both ``proto_shape`` and
            ``kernel_size`` are ``None`` or the ``proto_shape`` is invalid.
        """
        if kernel_size is None and proto_shape is None:
            raise ValueError("Either kernel_size or proto_shape must be given, "
                             "but both None")
        if proto_shape is not None:
            proto_shape = np.array(proto_shape)
            if len(proto_shape.shape) != len(cls.AREA_DIMS):
                raise ValueError(("proto_shape must be {}-dimensional, but "
                                  "shape was {}").format(len(cls.AREA_DIMS),
                                                         proto_shape.shape))
        else:
            proto_shape: np.ndarray = np.ones(kernel_size)
        return proto_shape


class BatchIntersectEncode2D(BatchConvOp):
    r"""Apply intersection encoding to batch of input masks of shape
    ``(batch, 1, height, width)``.

    The idea of intersection encoding is to have a generalized and continuous
    bounding box score. The better a given proto-shape (e.g. a box)
    intersects with the ground truth shape, the higher the value in ``[0,1]``.
    More precisely, in an IoU encoding of a segmentation target, a pixel holds
    the value how much of a pre-defined proto-shape centered at this pixel
    intersects with the ground truth segmentation.

    Such a proto shape is defined by a (not necessarily binary) mask,
    which may be at most the size of the target mask.


    **Theoretical notes**

    The set intersection is implemented as fuzzy set intersection with
    respect to the product t-norm. This means, the intersection is
    ``element_wise_prod(A, B)``, and the value of a pixel in a mask is
    treated similar to probability that the pixel belongs to the mask.
    We use reduction by sum to calculate the area of a fuzzy mask.

    Using sum reduction allows to use the intersection area value for union
    calculation:
    The product t-norm fuzzy union is defined as element-wise
    :math:`A + B - A \cdot B` for sets :math:`A, B`.
    Thus, the union area is:

    .. code-block:: python

        sum(A + B - A*B) = sum(A) + sum(B) - sum(A*B)

    where ``sum`` is the reduction by sum and ``sum(A*B)``
    is the intersection area.

    The following holds for a factor ``c``:

    .. code-block:: python

        intersect(c*A, B) == c * intersect(A, B).

    This is used to normalize the intersection value by the proto shape size
    if requested, i.e. ``c=area(proto shape)``.

    .. note::
        In the case that at least one of proto shape or the segmentation mask
        is binary with values in ``{0, 1}``, the product t-norm fuzzy
        intersection is equivalent to the minimum t-norm fuzzy intersection,
        which is ``element_wise_min(A, B)``. If both proto shape and
        segmentation feature non-binary values, the product intersection can
        become much smaller than the minimum intersection.


    **Implementation Notes**

    The intersection calculation for each possible "center pixel",
    i.e. proto shape location, is done by applying a convolution to the
    segmentation mask, the weights of which hold the mask.

    To change from 2D to other image dimensionality, replace the padding and
    convolution layer and adapt :py:attr:`~BatchWindowOp.AREA_DIMS`
    accordingly.
    """

    def __init__(self,
                 proto_shape: np.ndarray = None,
                 kernel_size: Tuple[int, ...] = None,
                 normalize_by: str = 'proto_shape'):
        """Init.

        :param proto_shape: the proto shape definition in a form accepted by
            :py:func:`numpy.ndarray`
        :param kernel_size: if ``proto_shape`` is ``None``,
            use all-ones rectangular shape of ``kernel_size``
        :param normalize_by: whether to normalize the intersection output,
            i.e. divide it by the total area of the proto shape or the
            target, or none; allowed options:
            ``'proto_shape'``, ``'target'``, ``'none'``
        """
        if normalize_by not in ('none', 'proto_shape', 'target'):
            raise ValueError(("normalize_by must be one of "
                              "('none', 'proto_shape', 'target') but was {}"
                              ).format(normalize_by))
        super().__init__()
        proto_shape = self._to_valid_proto_shape(proto_shape=proto_shape,
                                                 kernel_size=kernel_size)
        kernel_size = proto_shape.shape

        # Attributes:
        self.normalize_by = normalize_by
        """Whether to normalize the intersection area.
        Divide the intersection area value by:

        ========================= =======
        Value of ``normalize_by``  Divisor = total area of ...
        ========================= =======
        ``'proto_shape'``         ... the proto shape
        ``'target'``              ... the target.
        ========================= =======
        """

        # Modules:
        # Beware: The padding for ZeroPad2d has crude specification:
        # 1. width pad, 2. height pad
        self.padding = torch.nn.ZeroPad2d(
            padding=same_padding((kernel_size[1], kernel_size[0])))
        """Padding to obtain same size as input after convolution"""
        self.intersect_conv = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                              kernel_size=kernel_size)
        """Convolution to calculate the intersection for each location of
        ``proto_shape``"""
        # pylint: disable=no-member
        self.intersect_conv.weight.data = (torch.from_numpy(proto_shape).float()
                                           # unsqueeze batch and channel axes
                                           .unsqueeze(0).unsqueeze(0))
        self.intersect_conv.bias.data = (torch.from_numpy(np.array(0.)).float()
                                         # unsqueeze batch axis
                                         .unsqueeze(0))
        # pylint: enable=no-member

        # Set to non-trainable:
        self.requires_grad_(False)
        self.eval()

    @property
    def proto_shape(self) -> np.ndarray:
        """The proto shape used for intersection calculation"""
        return (self.intersect_conv.weight.data
                .detach().cpu().numpy().squeeze(0).squeeze(0))

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce instance."""
        return dict(**super().settings,
                    normalize_by=self.normalize_by)

    def __repr__(self):
        return settings_to_repr(self, self.settings)

    def conv_op(self, masks: torch.Tensor) -> torch.Tensor:
        """Encode given masks as IoU masks.

        :param masks: :py:class:`torch.Tensor` of shape
            ``(batch_dim, 1, width, height)`` holding the segmentation masks
            for one batch.
        :return: tensor of the same size as masks tensor holding IoU encoding
            of the latter
        """
        masks = self.padding(masks)
        # Calc intersect in each location of proto_shape:
        intersect: torch.Tensor = self.intersect_conv(masks)

        # Normalization:
        if self.normalize_by == "proto_shape":
            intersect /= float(np.sum(self.proto_shape))
        elif self.normalize_by == "target":
            intersect /= \
                masks.sum(dim=self.AREA_DIMS).unsqueeze(-1).unsqueeze(-1)
        return intersect


class BatchIoUEncode2D(BatchConvOp):
    """Apply intersection over union encoding to an input batch.
    The batch is assumed to be of shape ``(batch_size, 1, height, width).``

    **Idea**

    The idea of intersection over union (IoU) encoding is to have a generalized
    and continuous bounding box score. The better a given proto-shape
    (e.g. a box) overlaps with the ground truth shape, the higher the value
    in ``[0,1]``.
    More precisely, in an IoU encoding of a segmentation target, a pixel holds
    the IoU value of the actual segmentation with a pre-defined proto-shape
    centered at this pixel. In a traditional segmentation a pixel holds the
    information whether it is part of a visual object in the image.

    Such a proto shape is defined by a (not necessarily binary) mask,
    which may be at most the size of the target mask.


    **Theoretical Notes**

    Since the pixel values of proto shape and segmentation mask need not be
    binary, they are treated as fuzzy sets. Intersection and union of masks
    are calculated with respect to the product t-norm (see
    :py:class:`BatchIntersectEncode2D`).
    The area values are collected by reduction by sum.


    **Implementation Notes**

    To change from 2D to other image dimensionality, replace the padding and
    pooling layer and adapt :py:attr:`~BatchWindowOp.AREA_DIMS` accordingly.
    """

    def __init__(self,
                 proto_shape: np.ndarray = None,
                 kernel_size: Tuple[int, ...] = None,
                 smooth: float = 1e-7):
        """Init.

        :param proto_shape: the proto shape definition in a form accepted by
            :py:func:`numpy.ndarray`
        :param kernel_size: if ``proto_shape`` is ``None``,
            use all-ones rectangular shape of ``kernel_size``
        :param smooth: smoothing summand for smooth division
        """
        super().__init__()

        # Modules:
        self.intersect_encoder: BatchIntersectEncode2D = BatchIntersectEncode2D(
            proto_shape=proto_shape, kernel_size=kernel_size,
            normalize_by="none")

        # Attributes:
        self.area_proto_shape: float = float(np.sum(self.proto_shape))
        """Area of the proto shape; calculate only once for speed-up"""
        self.smooth: float = smooth
        """Smoothening summand for smooth division."""

        # Set to non-trainable:
        self.requires_grad_(False)
        self.eval()

    @property
    def proto_shape(self) -> np.ndarray:
        """The proto shape used for IoU calculation"""
        return self.intersect_encoder.proto_shape

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce instance."""
        return dict(**super().settings, smooth=self.smooth)

    def smooth_division(self, dividend, divisor):
        """Smoothed division using smoothening summand to avoid division by 0.

        :return: result of smooth division.
        """
        return (dividend + self.smooth) / (divisor + self.smooth)

    def conv_op(self, masks: torch.Tensor) -> torch.Tensor:
        """Encode given masks as IoU masks.

        :param masks: :py:class:`torch.Tensor` of shape
            ``(batch_size, 1, width, height)`` holding the segmentation masks
            for one batch
        :return: :py:class:`torch.Tensor` of the same size as masks tensor
            holding IoU encoding of the latter
        """
        # Calc intersect in each location of proto_shape
        # (also does value checks):
        intersect: torch.Tensor = self.intersect_encoder(masks)

        # Calc:
        # union = area_masks + area_proto_shape
        #         - intersect in each location of proto_shape
        area_masks: torch.Tensor = masks.sum(self.AREA_DIMS)  # along area axes
        area_sum = ((area_masks + self.area_proto_shape)
                    # unsqueeze image dimensions
                    .view([*area_masks.size(), *([1] * len(self.AREA_DIMS))]))
        union: torch.Tensor = area_sum - intersect

        # Calc IoU in each location of proto_shape
        iou = self.smooth_division(intersect, union)
        return iou


class BatchIntersectDecode2D(BatchConvOp):
    r"""Given batch of IoU encoded masks, estimates the original
    segmentation mask.

    This estimation is done by

    1. "bloating" each pixel: Create a mask with the :py:attr:`proto_shape`
       at the pixel's location, with the :py:attr:`proto_shape` weighted by
       the pixel value
    2. adding up all bloated pixel masks to obtain one mask

    Consider the convolution that describes the IoU encoding of the given mask.
    Then above steps can be simplified to a convolution with kernel and padding
    the ones from the encoding convolution but

    - kernel and padding flipped along each dimension;
      in 2D, the two flips are equivalent to a rotation by 180°,
    - kernel normalized by L1 norm (after that, the values in the convolution
      sum up to 1)

    **Derivation**

    The decoder formulas can be derived as follows:
    Consider a pixel :math:`p` in the to be estimated segmentation mask, and
    its coordinates :math:`(p_{a})_{a \in \text{axes}}`
    with image axes being the axes describing a single image,
    e.g. in 2D ``(width, height)``.
    The bloat mask of a pixel :math:`p^{iou}` in the IoU encoded mask can
    only contribute to the value of :math:`p` if the :py:attr:`proto_shape`
    centered at :math:`p^{iou}` would reach to :math:`p`, i.e. if the
    distance to :math:`p` in each image axis :math:`a` is

    .. math::
        - \text{ceil}(0.5 \cdot (\text{proto_shape_size}[a] - 1))
        &\leq p_{a} - p^{iou}_{a}  \\
        &\leq  \text{floor}(0.5 \cdot (\text{proto_shape_size}[a] - 1))

    where
    :math:`p_{a} - p^{iou}_{a} < 0`
    means :math:`p^{iou}` is left of :math:`p`, and right else.
    This describes a kernel of the same size as the IoU encoding kernel but
    with padding flipped in each dimension.
    The kernel entries, i.e. the contribution of :math:`p^{iou}` in kernel
    position :math:`(pos_{a})_{a \in \text{axes}}` to :math:`p`, are:

    .. math::
        \text{proto_shape}[
            (p_{a} - p^{iou}_{a}
             + \text{floor}(0.5 \cdot (\text{proto_shape_size}[a] - 1))_{a}
        ] \\
        = \text{proto_shape}[(\text{proto_shape_size}[a] - pos_{a})_{a}]

    which is the :py:attr:`proto_shape` kernel of the IoU encoding but
    flipped in each dimension.
    """

    def __init__(self,
                 proto_shape: np.ndarray = None,
                 kernel_size: Tuple[int, ...] = None):
        """Init.

        :param proto_shape: the proto shape used for IoU encoding in a form
            accepted by :py:func:`numpy.ndarray`
        :param kernel_size: if ``proto_shape`` is ``None``,
            use all-ones rectangular shape of ``kernel_size``
        """
        # Parameter post-processing:
        proto_shape = self._to_valid_proto_shape(proto_shape=proto_shape,
                                                 kernel_size=kernel_size)

        # flip all axes and normalize:
        kernel = np.flipud(np.fliplr(proto_shape)) / np.sum(proto_shape)
        kernel_size = kernel.shape
        # Beware: The padding for ZeroPad2d has crude specification:
        # 1. width pad, 2. height pad
        unflipped_padding = same_padding((kernel_size[1], kernel_size[0]))
        padding = [unflipped_padding[1], unflipped_padding[0],
                   unflipped_padding[3], unflipped_padding[2]]

        super().__init__()

        # Modules:
        self.padding = torch.nn.ZeroPad2d(padding=padding)
        """Padding to obtain same size as input after convolution"""
        self.decoder_conv = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                            kernel_size=kernel_size)
        """Convolution to calculate the intersection for each location of
        proto_shape"""
        # pylint: disable=no-member
        self.decoder_conv.weight.data = (torch.from_numpy(kernel).float()
                                         # unsqueeze batch and channel axes
                                         .unsqueeze(0).unsqueeze(0))
        self.decoder_conv.bias.data = (torch.from_numpy(np.array(0.)).float()
                                       # unsqueeze batch axis
                                       .unsqueeze(0))
        # pylint: enable=no-member

        # Set to non-trainable:
        self.requires_grad_(False)
        self.eval()

    @property
    def decoding_proto_shape(self) -> np.ndarray:
        """The kernel of the decoder (flipped and normalized IoU encoding
        proto shape)."""
        return (self.decoder_conv.weight.data
                .detach().cpu().numpy().squeeze(0).squeeze(0))

    @property
    def proto_shape(self) -> np.ndarray:
        """The (L1-normalized) proto shape used for the IoU encoding."""
        return np.flipud(np.fliplr(self.decoding_proto_shape))  # undo flipping

    def conv_op(self, masks: torch.Tensor) -> torch.Tensor:
        """Forward pass: Apply decoding convolution"""
        masks = self.padding(masks)
        masks = self.decoder_conv(masks)
        return masks


def same_padding(kernel_size: Sequence[int], hang_front: bool = False) -> Tuple[int, ...]:
    """Calculate the left and right padding for mode ``'same'`` for each dim
    and concat.

    Mode ``'same'`` here means
    ``Conv(kernel_size)(Pad(padding)(x)).size() == x.size()``.

    Padding is distributed equally on both sides of a dimension.
    If unequal padding is needed in one dimension, by default (``hang_front==False``)
    the front gets padded by one pixel less than the back.
    To instead pad the front more, set ``hang_font==True``.

    .. warning::
        Currently (Apr 2019), :py:class:`torch.nn.ZeroPad2d` requires the
        padding in a special format: first width paddings, then height paddings.
        So the entries from kernel_size need to be swapped to obtain the
        correct padding as input for :py:class:`torch.nn.ZeroPad2d`.

    :param kernel_size: the list of kernel dimension sizes
    :param hang_front: whether the front instead of the rear padding should
        be larger by 1 in case unequal padding is needed
    :return: padding tuple as
        ``(left dim0, right dim0, left dim1, right dim1, ...)``
    """
    if not all([isinstance(i, int) and i > 0 for i in kernel_size]):
        raise ValueError("Invalid kernel size {}; must be integers > 0"
                         .format(kernel_size))

    # get tuples of (left pad, right pad) for each dim
    left_right_paddings = [_same_padding_for_dim(i, hang_front=hang_front)
                           for i in kernel_size]
    # flatten and return:
    return tuple([p for left_right in left_right_paddings for p in left_right])


def _same_padding_for_dim(kernel_dim_size: int, hang_front: bool = False) -> Tuple[int, int]:
    """Calc left and right padding to have same padding for kernel dim size.

    If the kernel dimension is odd, one obtains symmetric padding,
    else asymmetric one (with right padding the larger one).
    The padding is distributed over left and right using the constraints
    ``total_padding = kernel_dim_size - 1`` and
    ``0 <= (right_padding - left_padding) <= 1``.
    By default (``hang_front==False``), the rear is padded more by 1 in case unequal
    padding is needed. For ``hang_front==True``, the front instead is padded by 1 more.
    If ``hang_front==False``, this yields the same results as ``same`` padding in tensorflow with
    ``stride == 1``, compare https://stackoverflow.com/questions/45254554/
    *(no code is taken from there)*.

    :param kernel_dim_size: the size of the kernel dimension to calculate padding for
    :param hang_front: whether the front instead of the rear padding should
        be larger by 1 in case unequal padding is needed
    :return: Tuple of front and rear 1D padding for the given kernel dimension
    """
    total_padding: int = kernel_dim_size - 1
    smaller_padding: int = total_padding // 2  # the shorter value
    larger_padding: int = total_padding - smaller_padding
    if not hang_front:
        return smaller_padding, larger_padding
    return larger_padding, smaller_padding

