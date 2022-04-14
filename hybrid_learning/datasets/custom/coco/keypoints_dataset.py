"""MS COCO Keypoints dataset handle with subsetting functionality and annotation transformations."""
#  Copyright (c) 2022 Continental Automotive GmbH
import abc
import logging
from typing import Tuple, List, Dict, Any, Sequence, Union

import PIL.Image
import numpy as np
import torch
from pycocotools.coco import COCO

from .base import COCODataset
from .keypoints_processing import pad_and_scale_annotation
from ... import transforms as trafos


class KeypointsDataset(COCODataset):
    # noinspection PyArgumentEqualDefault
    """Handler for (a subset of) a COCO keypoints dataset.
    Input images are the original COCO images.
    Annotations are the keypoint annotations for the images.

    This is essentially a wrapper around a COCO handle following the
    scheme of :py:class:`~hybrid_learning.datasets.base.BaseDataset` and
    allowing for restriction to specific licenses.
    """

    def __init__(self, **kwargs):
        """Init.

        :param spec: Specification
        :param kwargs: Arguments for super class
        """
        super().__init__(**kwargs)

    @classmethod
    def _get_default_after_cache_trafo(cls, device=None):
        return trafos.OnInput(trafos.ToTensor(
            device=device, sparse=False, dtype=torch.float, requires_grad=False))

    @classmethod
    def get_default_transforms(cls, img_size: Tuple[int, int],
                               device: torch.device = None
                               ) -> trafos.TupleTransforms:
        """Return the default transformation, which is pad and resize."""
        return (trafos.OnInput(trafos.ToTensor(device=device)) +
                PadAndScaleAnnotatedImg(img_size=img_size))

    def getitem(self, i: int) -> Tuple[PIL.Image.Image, List[Dict]]:
        """Collect the image and the keypoint annotations at position ``i``.
        Used for
        :py:meth:`~hybrid_learning.datasets.base.BaseDataset.__getitem__`."""
        annotations: List[Dict] = self.raw_anns(i)
        img: PIL.Image.Image = self.load_orig_image(i)
        return img, annotations


class PadAndScaleAnnotatedImg(trafos.TupleTransforms):
    """Pickleable wrapper to pad and scale both an image and its COCO annotation to a target size.
    See :py:func:`~hybrid_learning.datasets.custom.coco.keypoints_processing.pad_and_scale_annotation`."""

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        return dict(img_size=self.img_size)

    def __init__(self, img_size: Tuple[int, int]):
        super().__init__()
        self.img_size = img_size
        """The target size of the annotation."""

    def apply_to(self, img_t: torch.Tensor, anns: List[Dict[str, Any]]):
        """Pad and rescale both image and annotations."""
        new_img_t = trafos.pad_and_resize(img_t, img_size=self.img_size)
        new_anns = [pad_and_scale_annotation(
            ann, from_size=img_t.size()[-2:], to_size=self.img_size
        ) for ann in anns]
        return new_img_t, new_anns


class _ToSegMask(trafos.Transform, abc.ABC):
    """Base class for transformation of coco annotations to segmentation masks."""

    @property
    def settings(self) -> Dict[str, Any]:
        return dict(img_size=self.img_size, coco_handle=self.coco_handle, merge_masks=self.merge_masks)

    def __init__(self, img_size: Tuple[int, int] = None, coco_handle: COCO = None,
                 merge_masks: bool = True, include_crowds: bool = True):
        """Init.

        :param img_size: the constant image size to assume for all annotations;
            if not given, image size is automatically retrieved from the ``coco_handle``
        :param coco_handle: if ``img_size`` is not given, determine image size using
            the coco meta information handled by this object
        :param merge_masks: whether to merge the masks for single annotations into one
        :param include_crowds: whether to include annotations for groups
        """
        if coco_handle is None and img_size is None:
            raise ValueError(
                "Either coco_handle or img_size must be given but both were None.")
        super().__init__()
        self.coco_handle: COCO = coco_handle
        self.img_size: Tuple[int, int] = img_size
        self.merge_masks: bool = merge_masks
        self.include_crowds: bool = include_crowds

    @abc.abstractmethod
    def to_masks(self, anns: Sequence[Dict[str, Any]]) -> List[Union[np.ndarray, torch.BoolTensor]]:
        """Turn annotations into list of boolean masks."""
        raise NotImplementedError

    def apply_to(self, anns: Sequence[Dict[str, Any]]) -> torch.FloatTensor:
        """Turn annotations into a single binary float tensor.
        Tensor shape will be ``(masks, height, width)`` with value of ``masks``
        depending on the :py:attr:`merge_masks` setting."""
        if not all(ann['image_id'] == anns[0]['image_id'] for ann in anns):
            logging.warning("Transforming annotations not belonging to the same image."
                            "Found image IDs:\n%s", str([ann['image_id'] for ann in anns]))
        if not self.include_crowds:
            anns = [ann for ann in anns if not ann['iscrowd']]
        masks: List[torch.Tensor] = [torch.as_tensor(mask, dtype=torch.bool)
                                     for mask in self.to_masks(anns)]
        masks_t: torch.Tensor = torch.stack(masks).unsqueeze(1)  # shape: (masks, 1, h, w)
        mask_t: torch.Tensor = masks_t.squeeze(1) if not self.merge_masks else torch.any(masks_t, dim=0)
        return mask_t.to(torch.float)  # shape: (masks, h, w)


class COCOSegToSegMask(_ToSegMask):
    """Turn COCO instance segmentation annotations into a single binary segmentation mask.
    The reduction is the trivial logical AND, as masks are assumed to be binary.
    Output size: ``(masks, height, width)``."""

    def to_masks(self, anns: Sequence[Dict[str, Any]]) -> List[np.ndarray]:
        """Turn annotations into list of boolean masks."""
        if self.img_size is not None:
            # ugly trick to apply coco transform to transformed annotation with custom image size
            coco_handle = COCO()
            coco_handle.imgs = {anns[0]['image_id']: {'height': self.img_size[0], 'width': self.img_size[1]}}
        else:
            coco_handle = self.coco_handle
        return [coco_handle.annToMask(ann) for ann in anns]


class COCOBoxToSegMask(_ToSegMask):
    """Turn the bounding box info from COCO annotations into a single binary segmentation mask.
    The reduction is the trivial logical AND, as masks are assumed to be binary.
    Output size: ``(masks, height, width)``."""

    def _img_size_for(self, anns: Sequence[Dict[str, Any]] = None) -> Tuple[int, int]:
        """Return the image size for an image ID.
        It is either fixed, if :py:attr:`img_size` is given, or determined
        using :py:attr:`coco_handle`.

        :return: image size in ``(height, width)``
        """
        img_size = self.img_size
        if img_size is None:
            if anns is None:
                raise ValueError(
                    "Annotations not given but self.img_size unset -- could not determine image size!")
            img_info = self.coco_handle.imgs[anns[0]['image_id']]
            img_size = (img_info['height'], img_info['width'])
        return img_size

    def to_masks(self, anns: Sequence[Dict[str, Any]]) -> List[torch.BoolTensor]:
        """Turn annotations into list of boolean masks."""
        box_masks: List[torch.BoolTensor] = []
        height, width = self._img_size_for(anns)
        for x_min, y_min, w, h in [ann['bbox'] for ann in anns]:
            x1, x2 = int(round(x_min)), int(round(x_min + w))
            y1, y2 = int(round(y_min)), int(round(y_min + h))
            box_mask: torch.BoolTensor = torch.zeros(height, width, dtype=torch.bool).to(torch.bool)
            box_mask[y1: y2, x1: x2] = True
            box_masks.append(box_mask)
        return box_masks
