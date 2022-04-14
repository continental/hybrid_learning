"""Handle that derives keypoint heatmaps from MS COCO keypoint annotations."""
#  Copyright (c) 2022 Continental Automotive GmbH
from typing import Tuple, Callable, Optional, Union

import PIL.Image
import torch

from .keypoints_processing import annotations_to_heatmap
from .mask_dataset import ConceptDataset
from ... import transforms as trafos


class HeatmapDataset(ConceptDataset):
    """Data handle for ground truth heatmaps of centroids of visual concepts
    (body parts) generated from COCO keypoints.
    Works the same as the
    :py:class:`hybrid_learning.datasets.custom.coco.mask_dataset.ConceptDataset`
    only with different :py:attr:`MASKS_ROOT_ROOT` and different
    :py:meth:`~hybrid_learning.datasets.custom.coco.mask_dataset.ConceptDataset.annotations_to_mask`
    method. See there for details on how the centroid heatmaps look like.
    """

    MASKS_ROOT_ROOT = "kpt_heatmaps"
    """Usual parent to all heatmap folders for all body parts; sibling to
    ``images`` folder"""

    @classmethod
    def get_default_transforms(cls, img_size: Optional[Tuple[int, int]] = None,
                               mask_size: Optional[Tuple[int, int]] = None,
                               device: Union[str, torch.device] = None,
                               ) -> Callable[[PIL.Image.Image, PIL.Image.Image],
                                             Tuple[torch.Tensor, torch.Tensor]]:
        """Default transformation that pads and resizes both images and masks,
        and binarizes the masks."""
        # noinspection PyTypeChecker
        trafo: trafos.Compose = super().get_default_transforms(
            img_size=img_size, mask_size=mask_size, device=device)
        # The super method adds binarizing as a last transformation -> remove it
        assert trafo.transforms.pop(-1) == trafos.OnTarget(trafos.Binarize())
        # noinspection PyTypeChecker
        return trafo

    ANNOTATION_TRANSFORM: Callable = annotations_to_heatmap
    # pylint: disable=line-too-long
    """Callable used to transform a list of annotations into a heatmap mask.
    Create a heatmap of all body parts given by keypoints in the
    ``annotations``.

    .. warning::
        Currently, all keypoints occurring in the body parts are treated
        separately!
        Centroids of body parts consisting of more than one keypoint are
        not supported yet.

    For details see
    :py:func:`~hybrid_learning.datasets.custom.coco.keypoints_processing.annotations_to_heatmap`.
    """
    # pylint: enable=line-too-long
