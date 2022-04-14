"""Common utility functions for custom model extensions."""
#  Copyright (c) 2022 Continental Automotive GmbH
from typing import Callable, List, Sequence

import torch


def bbox_to_segmask(bbox_info_batch: torch.Tensor,
                    logical_or: Callable[[List[torch.Tensor]], torch.Tensor],
                    image_size: Sequence[int] = (640, 640),
                    target_class: int = 1,
                    min_score: float = None) -> torch.Tensor:
    """Transform EfficientDet output into segmentation mask.
    The ``bbox_info_batch`` must have shape ``[batch, #bboxes per image, 6]``,
    each row representing ``[x_min, y_min, x_max, y_max, score, class]``.
    The returned mask has shape ``[batch, (#bboxes | 1), *image_size]``.
    The per-image-bbox-masks are reduced to per-image-masks using ``logical_or``
    if this is set.
    Note that not setting ``logical_or`` will only succeed in case of uniform #bboxes!
    For details see the forward method and attributes of
    :py:class:`EfficientDetToSetMask`.

    :param bbox_info_batch: tensor of shape ``[batch, #bbox per image, 6]``
        where the last dimension encodes ``(x_min, y_min, x_max, y_max, score, class)``
    :param logical_or: callable that accepts a list of tensor segmentation masks
        with values in [0, 1] (one 2D mask for each bounding box in an image)
        and returns a single combined segmentation mask;
        e.g. use a variadic fuzzy logical OR object
    :param image_size: the size of the original image as ``(height, width)``
        for which to draw the bounding boxes
    :param target_class: the ID of the target class (see also ``bbox_info_batch``) for selection
    :param min_score: the minimum score a box must have to be included in the combined mask
    :return: tensor of size ``(batch, 1, height, width)``
    """
    assert len(image_size) == 2, "Invalid image_size: {}".format(image_size)
    _tens_args = dict(dtype=bbox_info_batch.dtype, layout=bbox_info_batch.layout, device=bbox_info_batch.device)
    out_tensors = []
    for pred in bbox_info_batch:
        bbox_masks: List[torch.Tensor] = []

        # Subsetting:
        if target_class is not None:
            classes = pred[:, 5]
            pred = pred[classes == target_class]
        if min_score is not None:
            scores = pred[:, 4]
            pred = pred[scores >= min_score]

        # Box generation:
        for bbox_id, bbox in enumerate(pred):
            bbox_mask = torch.zeros(*image_size, **_tens_args)
            x_min, y_min, x_max, y_max, score, _ = torch.round(bbox).int()[:]
            bbox_mask[y_min: y_max, x_min: x_max] = score
            bbox_masks.append(bbox_mask)

        # Logics
        if len(bbox_masks) == 0:
            pred_mask: torch.Tensor = torch.zeros(1, *image_size, **_tens_args)  # size = [1, h, w]
        elif logical_or is not None:
            pred_mask: torch.Tensor = logical_or(bbox_masks).unsqueeze(0)  # size = [1, h, w]
        else:
            pred_mask = torch.stack(bbox_masks)  # size = [num_bboxes, h, w]

        out_tensors.append(pred_mask)
    return torch.stack(out_tensors)  # size = [batch_size, (num_bboxes | 1), h, w]
