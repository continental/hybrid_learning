"""Custom modules for handling and modifying the EfficientDet models
from https://github.com/rwightman/efficientdet-pytorch.
Make sure to install the corresponding package via

.. code:: shell

    # EfficientDet pytorch implementation
    pip install -e "git+https://github.com/rwightman/efficientdet-pytorch.git@75e16c2f#egg=effdet"
"""
#  Copyright (c) 2022 Continental Automotive GmbH
from typing import Sequence, Union

import torch

from .models_common import bbox_to_segmask
from ....fuzzy_logic import logic_by_name, Merge, Logic


class EfficientDetToSegMask(torch.nn.Module):
    """Turn batch output of torchvision Mask R-CNN into segmentation masks
    for given label.
    For the different choices of fuzzy logical ``OR`` applicable to merge
    masks of different bounding boxes see the transforms module."""

    def __init__(self, fuzzy_logic: Union[str, Logic],
                 image_size: Sequence[int] = (640, 640),
                 target_class: int = 1):
        super().__init__()
        fuzzy_logic: Logic = logic_by_name(fuzzy_logic) if isinstance(fuzzy_logic, str) else fuzzy_logic
        self.logical_or: Merge = fuzzy_logic.logical_('OR') if fuzzy_logic else None
        """The fuzzy logical OR callable to use to reduce the instance segmentation
        masks to a semantic segmentation. OR operation is used to reduce
        non-binary segmentation mask tensors stacked in dim 0."""
        self.image_size: Sequence[int] = image_size
        """Output image size as ``(height, width)``.
        Make sure to use the ``efficientdet.config.image_size`` here."""
        self.target_class: int = target_class
        """The int target class ID. E.g. for MS COCO 1=person"""

    def forward(self, effdet_out: torch.Tensor) -> torch.Tensor:
        """Transform efficientdet output into segmentation mask.

        :param effdet_out: tensor with shape ``[max_det_per_image, 6]``,
            each row representing ``[x_min, y_min, x_max, y_max, score, class]``;
            cf. https://github.com/rwightman/efficientdet-pytorch/blob/75e16c2f/effdet/anchors.py#L121
        :return: masks tensor of size ``(batch, 1, height, width)``
        """
        return bbox_to_segmask(
            effdet_out,
            logical_or=self.logical_or,
            image_size=self.image_size,
            target_class=self.target_class,
        )
