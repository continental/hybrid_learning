"""Handle that derives concept masks from MS COCO keypoint annotations."""

#  Copyright (c) 2022 Continental Automotive GmbH
import logging
import os
from typing import Sequence, Dict, Any, Tuple, Set, List, Iterable, Optional, \
    Union, Callable

import PIL.Image
import PIL.ImageDraw
import numpy as np
import torch
from tqdm import tqdm

from . import keypoints_processing as kpt_proc
from .base import COCODataset
from .body_parts import BodyParts
from ... import transforms as trafos, DatasetSplit

LOGGER = logging.getLogger(__name__)


class ConceptDataset(COCODataset):
    # pylint: disable=line-too-long
    r"""Data handle for ground truth segmentations of visual concepts
    (body parts) generated from COCO keypoints. The concepts that can be
    handled are combinations of objects (body parts) marked by keypoints.

    The input images the dataset contains are the original COCO images.
    The annotations are segmentation masks masking the desired concept which are
    generated from the keypoint annotations. The translation of keypoint
    annotations to masks can be found in :py:meth:`annotations_to_mask`.

    The ground truth masks are generated and stored in a sub-directory
    :py:attr:`masks_root`:

    .. parsed-literal::

        ``<root root>``
        |
        +---annotations
        |   +---``<annotations file>.json``
        |
        +---images
        |   +---:py:attr:`~hybrid_learning.datasets.base.BaseDataset.dataset_root`
        |
        +---masks
            +---:py:attr:`masks_root`
                (default: ``<dataset_root>_<concept desc>``)
    """
    # pylint: enable=line-too-long
    MASKS_ROOT_ROOT = "masks"
    """Usual parent to all masks folders for all body parts; sibling to
    ``images`` folder"""

    def __init__(self,
                 body_parts: Sequence[Union[BodyParts, Sequence[str]]],
                 force_rebuild: bool = False,
                 pt_radius: float = None,
                 masks_root: str = None,
                 masks_root_root: str = None,
                 lazy_mask_generation: bool = True,
                 img_size: Tuple[int, int] = None,
                 mask_size: Optional[Tuple[int, int]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 **kwargs):
        """Init.

        :param body_parts: see
            :py:attr:`~hybrid_learning.datasets.custom.coco.mask_dataset.ConceptDataset.body_parts`
        :param force_rebuild: Whether to regenerate all masks during init,
            or to skip existing ones.
        :param pt_radius: see
            :py:attr:`~hybrid_learning.datasets.custom.coco.mask_dataset.ConceptDataset.pt_radius`
        :param masks_root: set the root directory under which to generate the
            masks; note that one should prefer setting ``masks_root_root``
            instead
        :param masks_root_root: the path to the root under which the masks
            folder is created; the masks folder's name is uniquely determined
            by the mask generation settings;
            defaults to ``coco_root/MASKS_ROOT_ROOT``
            where :py:attr:`MASKS_ROOT_ROOT` is the class default
        :param lazy_mask_generation: whether to generate the masks lazily or
            directly during init;
            overwritten by ``force_rebuild``: if this is true,
            masks are directly regenerated.
        :param img_size: see super class
        :param mask_size: if given, the size of the generated masks
            (annotations are scaled and padded to this size before mask
            generation)
        :param kwargs: Any other arguments for the super class
        """
        self.mask_size: Optional[Tuple[int, int]] = mask_size
        """If given, annotations are scaled and padded to this size before
        generating the masks (of this size).
        If left ``None``, the generated masks have the same size as the
        original image."""

        # Application of general spec
        # Apply custom default transforms:
        self._default_transforms: Callable[[PIL.Image.Image, PIL.Image.Image],
                                           Tuple[torch.Tensor, torch.Tensor]] \
            = self.get_default_transforms(img_size, self.mask_size,
                                          device=device)
        if img_size is not None:
            kwargs.update(img_size=img_size)
        super().__init__(**kwargs)

        if pt_radius is not None and pt_radius <= 0:
            raise ValueError(("pt_radius to generate masks must be > 0 "
                              "but was {}").format(pt_radius))
        self.pt_radius: float = pt_radius or 10 / 400
        """Radius of white circles to draw centered at the keypoint coordinates
        in the masks. Also used as half of the width of link lines between
        keypoints in the masks.
        Unit: proportion of height of annotated person in the image
        (defaults to the image height if person height cannot be estimated)."""

        # Concepts
        self.body_parts: Sequence[Union[BodyParts, Sequence[str]]] = body_parts
        """List of concepts to mask in selected images.
        See :py:meth:`annotations_to_mask` for details."""

        # The masks root folder
        self.masks_root = masks_root or self.default_masks_root(
            dataset_root=self.dataset_root, masks_root_root=masks_root_root,
            body_parts=self.body_parts, pt_radius=self.pt_radius,
            mask_size=self.mask_size,
        )
        """Folder name of the directory in which all generated masks are
        stored."""
        os.makedirs(self.masks_root, exist_ok=True)

        # Generate and save masks to root folder
        if not lazy_mask_generation or force_rebuild:
            self.generate_masks(force_rebuild=force_rebuild)

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        return dict(body_parts=self.body_parts,
                    pt_radius=self.pt_radius,
                    masks_root=self.masks_root,
                    **({'mask_size': self.mask_size} if self.mask_size else {}),
                    **super().settings)

    @classmethod
    def settings_to_str(cls,
                        body_parts: Iterable[Iterable[str]] = None,
                        pt_radius: float = None,
                        person_rel_size_range: Tuple[
                            Optional[float], Optional[float]] = None,
                        img_size: Tuple[int, int] = None,
                        mask_size: Tuple[int, int] = None,
                        dataset_root: str = None,
                        split: Union[str, DatasetSplit] = None,
                        ) -> str:
        """Return a string representation of the given mask
        generation settings.
        Intended to serve as unique folder name.
        If optional specifiers are not given, they are omitted.

        :param body_parts: see
            :py:attr:`~hybrid_learning.datasets.custom.coco.mask_dataset.ConceptDataset.body_parts`
        :param pt_radius: see
            :py:attr:`~hybrid_learning.datasets.custom.coco.mask_dataset.ConceptDataset.pt_radius`
        :param person_rel_size_range: the minimum and maximum height of persons
            included, relative to the image height;
            provide as tuple ``(min, max)`` and set ``None`` to unset values
        :param mask_size: the size of the created masks as ``(height, width)``;
            defaults to ``img_size``
        :param img_size: the image size for which the mask is created;
            should be given if one cannot assume the mask fits the original
            image size, or if e.g. ``person_rel_size_range`` is dependent on the
            image size; should be given as ``(height, width)``
        :param dataset_root: if given, used instead of split
        :param split: the dataset split
        :return: string that is unique for the given settings and can be used
            as folder name
        """
        # concepts
        body_parts = body_parts or []
        concept_hash: str = '-'.join(sorted({kpt for body_part in body_parts
                                             for kpt in body_part}))
        # radii
        min_size, max_size = person_rel_size_range or (None, None)
        radius_info: str = "_rad{:.3f}".format(pt_radius) \
            if pt_radius is not None else ""
        rel_height_info: str = "_relsize{fromv}to{tov}".format(
            fromv="{:.2f}".format(min_size if min_size is not None else 0),
            tov="{:.2f}".format(max_size) if max_size is not None else "+inf"
        ) if max_size is not None or min_size is not None else ""

        # size
        mask_size = mask_size if mask_size is not None else img_size
        mask_size_info: str = "_{}x{}".format(*mask_size) if mask_size else ""

        # split
        split: Optional[str] = split if not isinstance(split, DatasetSplit) \
            else split.value
        img_base: Optional[str] = os.path.basename(dataset_root) \
            if dataset_root is not None else split
        img_base: str = img_base + "_" if img_base is not None else ""

        return "{img_base}{concept_hash}{rad}{height}{size}".format(
            img_base=img_base,
            concept_hash=concept_hash,
            rad=radius_info,
            height=rel_height_info,
            size=mask_size_info
        )

    # noinspection PyIncorrectDocstring
    @classmethod
    def default_masks_root(cls, *,
                           dataset_root: str = None,
                           body_parts: Iterable[Iterable[str]] = None,
                           pt_radius: float = None,
                           person_rel_size_range: Tuple[
                               Optional[float], Optional[float]] = None,
                           img_size: Tuple[int, int] = None,
                           mask_size: Tuple[int, int] = None,
                           masks_root_root: str = None,
                           split: str = None
                           ) -> str:
        """Return default mask root directory for given settings.
        By default, a directory structure as follows is assumed:

        .. parsed-literal::

            coco_root
             |
             +-images_root_root
             |  +-os.path.basename(dataset_root)
             |
             +-masks_root_root
               +-os.path.basename(dataset_root)+spec_specifier

        If optional specifiers are not given, they are omitted.
        For the details on the parameters, see
        :py:meth:`~hybrid_learning.datasets.custom.coco.mask_dataset.ConceptDataset.settings_to_str`.

        :param dataset_root: the root directory of the coco images;
            basename is used to prefix the mask folder;
            the folder is used to determine the default for ``masks_root_root``
        :param masks_root_root: the root folder under which to put the masks;
            defaults to ``dataset_root/../../``:py:attr:`MASKS_ROOT_ROOT`;
        """
        if masks_root_root is None and dataset_root is None:
            raise ValueError("No root directory given.")
        masks_dirname = masks_root_root or os.path.join(
            os.path.dirname(os.path.dirname(dataset_root)),
            cls.MASKS_ROOT_ROOT
        )
        masks_basename: str = cls.settings_to_str(
            body_parts=body_parts,
            pt_radius=pt_radius,
            person_rel_size_range=person_rel_size_range,
            img_size=img_size,
            mask_size=mask_size,
            dataset_root=dataset_root,
            split=split,
        )
        return os.path.join(masks_dirname, masks_basename)

    def getitem(self, i: int) -> Tuple[PIL.Image.Image, PIL.Image.Image]:
        """Get image and mask by index ``i`` in dataset.
        Used for
        :py:meth:`~hybrid_learning.datasets.base.BaseDataset.__getitem__`."""
        img: PIL.Image.Image = self.load_orig_image(i)

        # generate and save mask if it does not exist
        if not self.mask_exists(i):
            # obtain image keypoint annotations
            anns = self.raw_anns(i)
            mask = self.annotations_to_mask(orig_img_wh=img.size,
                                            annotations=anns)
            self.save_mask(i, mask)

        mask: PIL.Image.Image = self.load_orig_mask(i)

        return img, mask

    def load_orig_mask(self, i: int) -> PIL.Image.Image:
        """Load and return unmodified mask as image by index in dataset.
        Image mode of returned masks is 'L'.
        Note that the mask has
        :py:attr:`~hybrid_learning.datasets.custom.coco.mask_dataset.ConceptDataset.mask_size`
        if this is set, not the original image's size (which is only changed
        to ``img_size`` during transformation)."""
        mask_fp: str = self.mask_filepath(i)
        mask = PIL.Image.open(mask_fp)

        # Check mask size
        if self.mask_size is not None and \
                (mask.size[1], mask.size[0]) != tuple(self.mask_size):
            raise RuntimeError(
                f"Mask cached at {mask_fp} does not match mask_size "
                f"{(mask.size[1], mask.size[0])} (HxW), but was of size "
                f"{self.mask_size}. Is masks_root correct?")

        # Possibly fix mask mode:
        if mask.mode != 'L':
            mask = mask.convert(mode='L')
        return mask

    @classmethod
    def get_default_transforms(cls, img_size: Optional[Tuple[int, int]] = None,
                               mask_size: Optional[Tuple[int, int]] = None,
                               device: Union[str, torch.device] = None,
                               ) -> Callable[[PIL.Image.Image, PIL.Image.Image],
                                             Tuple[torch.Tensor, torch.Tensor]]:
        """Default transformation that pads and resizes both images and masks,
        and binarizes the masks.
        If ``mask_size`` is ``None``, the masks are scaled and
        padded to the original image, otherwise to ``mask_size``.
        """
        mask_size = mask_size or img_size
        if img_size is not None:
            size_trafos = [trafos.OnInput(trafos.PadAndResize(img_size)),
                           trafos.OnTarget(trafos.PadAndResize(mask_size))]
        elif mask_size is not None:
            size_trafos = [trafos.OnTarget(trafos.PadAndResize(mask_size))]
        else:
            size_trafos = [trafos.SameSize()]

        # noinspection PyTypeChecker
        return trafos.Compose(
            [trafos.OnBothSides(trafos.ToTensor(device=device)),
             # pad an resize both img and mask:
             *size_trafos,
             # binarize the mask:
             trafos.OnTarget(trafos.Binarize())]
        )

    def mask_exists(self, i: int) -> bool:
        """Check whether a mask for the specified index already exists."""
        # Is it a valid ID?
        if not 0 <= i < len(self):
            m_exists = False

        # Does the mask already exist?
        else:
            mask_fp = self.mask_filepath(i)
            m_exists = os.path.exists(mask_fp)
        return m_exists

    def mask_filepath(self, i: int) -> str:
        """Provide the path under which the mask for the given image ID
        should lie."""
        return os.path.join(self.masks_root, self.descriptor(i))

    ANNOTATION_TRANSFORM: Callable = kpt_proc.annotations_to_mask
    # pylint: disable=line-too-long
    """Callable used to transform a list of annotations into a mask.
    All keypoints belonging to any of the :py:attr:`body_parts` are
    marked, and keypoints belonging to a common body part are linked
    according to the person skeleton.
    For details see
    :py:func:`~hybrid_learning.datasets.custom.coco.keypoints_processing.annotations_to_mask`."""

    # pylint: enable=line-too-long

    def annotations_to_mask(self, orig_img_wh: Tuple[int, int],
                            annotations: List[dict]
                            ) -> PIL.Image.Image:
        """Create mask of all configured keypoints from ``annotations``.
        It is assumed that the original image considered in ``annotations``
        has ``orig_img_wh``.
        If :py:attr:`~hybrid_learning.datasets.custom.coco.mask_dataset.ConceptDataset.mask_size`
        is ``None``, the mask size will be ``orig_img_wh``.
        For details see :py:attr:`ANNOTATION_TRANSFORM`.
        """
        # get skeleton; see also pycocotools
        skeleton: np.ndarray = self._get_skeleton(annotations[0])
        keypoint_idxs: List[Set[int]] = \
            [set(kpt_proc.to_keypoint_idxs(body_part, self.coco))
             for body_part in self.body_parts]
        annotation_wh = orig_img_wh
        if self.mask_size is not None:
            annotations = [
                kpt_proc.pad_and_scale_annotation(
                    ann, from_size=(orig_img_wh[1], orig_img_wh[0]),
                    to_size=self.mask_size)
                for ann in annotations]
            annotation_wh = (self.mask_size[1], self.mask_size[0])
        return self.__class__.ANNOTATION_TRANSFORM(
            annotation_wh=annotation_wh,
            annotations=annotations,
            pt_radius=self.pt_radius,
            keypoint_idxs=keypoint_idxs,
            skeleton=skeleton)

    def _get_skeleton(self, annotation: Dict[str, Any] = None) -> np.ndarray:
        """Obtain the coco skeleton information.
        The skeleton is returned as a list of lists of the form
        ``[[link1_start, link2_end], [link2_start, link2_end], ...]``
        where link starting and end points are the indices of keypoints in the
        ``keypoints`` list of an annotation.

        :param annotation: a COCO annotation for the category of which the
            skeleton should be loaded; if not given, the ``'person'``
            category is used
        """
        cat_id = annotation['category_id'] if annotation is not None \
            else self.coco.getCatIds(catNms=['person'])[0]
        skeleton: np.ndarray = np.array(
            self.coco.loadCats(cat_id)[0]['skeleton']
        ) - 1
        return skeleton

    def generate_masks(self, force_rebuild: bool = False,
                       show_progress_bar: bool = True,
                       **kwargs) -> None:
        """Generate and save the masks for the images in this dataset.

        :param force_rebuild: whether to overwrite masks that already exist.
        :param show_progress_bar: whether to show the progress using
            :py:class:`tqdm.tqdm`
        :param kwargs: further arguments to the progress bar
        """
        masks_to_process = [i for i in range(len(self))
                            if force_rebuild or not self.mask_exists(i)]
        if len(masks_to_process) == 0:
            return

        if show_progress_bar:
            masks_to_process = tqdm(**{**dict(iterable=masks_to_process,
                                              unit="mask",
                                              desc="Masks newly generated: "),
                                       **kwargs})
        for i in masks_to_process:
            if self.mask_exists(i) and force_rebuild:
                mask_fp = self.mask_filepath(i)
                LOGGER.info("Overwriting file %s", mask_fp)
                os.remove(mask_fp)

            # obtain image keypoint annotations
            anns = self.raw_anns(i)

            # obtain original image's size
            img: PIL.Image.Image = self.load_orig_image(i)
            img_size = img.size

            # generate and save mask
            mask = self.annotations_to_mask(orig_img_wh=img_size,
                                            annotations=anns)
            self.save_mask(i, mask)

    def save_mask(self, i: int, mask: PIL.Image.Image) -> None:
        """Save the given mask according to the given index in the dataset.

        :param i: ID of the image the mask belongs to.
        :param mask: mask image
        """
        filepath = self.mask_filepath(i)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        mask.save(filepath)
