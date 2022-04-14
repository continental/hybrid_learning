"""Custom modules for handling and modifying the pytorch Mask R-CNN model."""
#  Copyright (c) 2022 Continental Automotive GmbH
from typing import List, Dict, Sequence, Union

import torch

from .models_common import bbox_to_segmask
from ....fuzzy_logic import logic_by_name, Merge, Logic


class MaskRCNNToSegMask(torch.nn.Module):
    # pylint: disable=line-too-long
    """Turn batch segmentation output of torchvision Mask R-CNN into segmentation masks
    for given label.
    This is a fuzzy version of :py:class:`~hybrid_learning.dataset.custom.coco.keypoints_dataset.COCOSegToSegMask`.
    For the different choices of fuzzy ``fuzzy_logic_key`` see
    :py:mod:`~hybrid_learning.fuzzy_logic`.

    .. note::
        For post-processing applied to the bbox & mask predictions see
        ``torchvision.models.detection.roi_heads.RoIHeads.postprocess_detections``::

        - remove predictions with the background label
        - remove low scoring boxes (see RoIHeads.score_thresh)
        - remove empty boxes
        - non-maximum suppression, independently done per class
        - keep only topk scoring predictions (see RoIHeads.detections_per_img)
          https://github.com/pytorch/vision/blob/8aba1dc146f4df21e6f7549154819af9f431a813/torchvision/models/detection/roi_heads.py#L664
    """

    # pylint: enable=line-too-long

    def __init__(self, fuzzy_logic: Union[str, Logic] = 'goedel',
                 min_score: float = 0, target_class: int = 1):
        super().__init__()
        self.target_class: int = target_class
        """Label index. By default using person."""
        self.min_score: float = min_score
        """Minimum scores to respect."""
        fuzzy_logic: Logic = logic_by_name(fuzzy_logic) if isinstance(fuzzy_logic, str) else fuzzy_logic
        self.logical_or: Merge = fuzzy_logic.logical_('OR') if fuzzy_logic else None
        """The fuzzy logical OR callable to use to reduce the instance segmentation
        masks to a semantic segmentation. OR operation is used to reduce
        non-binary segmentation mask tensors stacked in dim 0."""

    def forward(self, outs: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Mask R-CNN outputs to segmentation masks."""
        results: List[torch.Tensor] = []
        for out in outs:
            mask = self.to_seg_mask(scores=out['scores'], labels=out['labels'],
                                    masks=out['masks'], boxes=out['boxes'])
            results.append(mask)
        return torch.stack(results)

    def to_seg_mask(self, scores: torch.Tensor, labels: torch.Tensor,
                    masks: torch.Tensor = None, boxes: torch.Tensor = None):
        """Turn single image prediction into a segmentation mask."""
        masks: torch.Tensor = masks[(scores >= self.min_score) & (labels == self.target_class)]
        if 0 in masks.size():  # no masks found -> 0
            mask_t: torch.Tensor = torch.zeros(*[d or 1 for d in masks.size()[1:]],
                                               device=masks.device, requires_grad=masks.requires_grad)
        else:
            mask_t = self.logical_or(masks)
        return mask_t


class MaskRCNNBoxToSegMask(MaskRCNNToSegMask):
    # pylint: disable=line-too-long
    """Turn batch bounding box output of torchvision Mask R-CNN into segmentation masks
    for given label.
    See :py:meth:`to_seg_mask` and super class for details.
    """

    # pylint: enable=line-too-long

    def __init__(self, fuzzy_logic: Union[str, Logic] = 'goedel',
                 min_score: float = 0, target_class: int = 1,
                 image_size: Sequence[int] = (400, 400), ):
        super().__init__(fuzzy_logic=fuzzy_logic, min_score=min_score, target_class=target_class)
        self.image_size: Sequence[int] = image_size
        """Output image size as ``(height, width)``.
        Make sure to use the ``efficientdet.config.image_size`` here."""

    def to_seg_mask(self, scores: torch.Tensor, labels: torch.Tensor,
                    masks: torch.Tensor = None, boxes: torch.Tensor = None):
        """Turn single image prediction into a segmentation mask.
        This uses :py:meth:"""
        return bbox_to_segmask(
            torch.cat([boxes.view(1, -1, 4), scores.view(1, -1, 1), labels.view(1, -1, 1)], dim=-1),
            logical_or=self.logical_or,
            image_size=self.image_size,
            target_class=self.target_class,
            min_score=self.min_score).squeeze(0)
