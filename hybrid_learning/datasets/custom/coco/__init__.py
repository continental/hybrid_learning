#  Copyright (c) 2022 Continental Automotive GmbH

# pylint: disable=line-too-long
"""Standard and concept datasets derived from `MS COCO dataset <coco>`_.

There are the following dataset handles, both derived from
:py:class:`~hybrid_learning.datasets.custom.coco.base.COCODataset`:

- :py:class:`~hybrid_learning.datasets.custom.coco.keypoints_dataset.KeypointsDataset`:
  *In:* COCO images, *GT:* COCO annotations
- :py:class:`~hybrid_learning.datasets.custom.coco.mask_dataset.ConceptDataset`:
  *In:* COCO images, *GT:* masks for the given body parts

Masks are generated and saved in COCO style folder structure if they do not
exist. For the documentation of the coco handle used under the hood, see the
short `COCO API documentation <coco-api>`_ and the
`COCO dataset format doc <coco-format-doc>`_.

.. _coco: https://cocodataset.org/
.. _coco-api: https://cocodataset.org/#download
.. _coco-format-doc: https://cocodataset.org/#format-data
"""
from .base import COCODataset
from .body_parts import BodyParts
from .mask_dataset import ConceptDataset
from .heatmap_dataset import HeatmapDataset
from .keypoints_dataset import KeypointsDataset, COCOSegToSegMask, COCOBoxToSegMask, PadAndScaleAnnotatedImg
