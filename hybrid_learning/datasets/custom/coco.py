"""Standard and concept datasets derived from `MS COCO dataset <coco>`_.

There are the following dataset handles, both derived from
:py:class:`COCODataset`:

- :py:class:`KeypointsDataset`:
  *In:* COCO images, *GT:* COCO annotations
- :py:class:`ConceptDataset`:
  *In:* COCO images, *GT:* masks for the given body parts

Masks are generated and saved in COCO style folder structure if they do not
exist. For the documentation of the coco handle used under the hood, see the
short `COCO API documentation <coco-api>`_ and the
`COCO dataset format doc <coco-format-doc>`_.

.. _coco: https://cocodataset.org/
.. _coco-api: https://cocodataset.org/#download
.. _coco-format-doc: https://cocodataset.org/#format-data
"""
#  Copyright (c) 2020 Continental Automotive GmbH

import abc
import contextlib
import enum
import json
import logging
import math
import os
import random
import shutil
from abc import ABC
from typing import Optional, Generator, List, Sequence, Iterable, \
    Tuple, Set, Callable, Any, Dict, Union

import PIL.Image
import PIL.ImageDraw
import numpy as np
import torch
import torchvision as tv
from pycocotools.coco import COCO
from tqdm import tqdm

from .. import transforms as trafos
from ..base import BaseDataset, DatasetSplit
from ..transforms import TupleTransforms

LOGGER = logging.getLogger(__name__)


def _left_right(keypoint_id: str) -> Tuple[str, str]:
    """Yield tuple of names of left and right instances."""
    return 'left_{}'.format(keypoint_id), 'right_{}'.format(keypoint_id)


class BodyParts(tuple, enum.Enum):
    """Mapping of a visual concept description (body part) to COCO keypoint ID
    collections approximating it.
    E.g. a ``"face"`` can be approximated by
    ``("left_eye", "right_eye", "nose")``.

    Available keypoints:

    ::

        "nose",
        "left_eye", "right_eye",
        "left_ear", "right_ear",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle"
    """

    # Single keypoints:
    NOSE = ('nose',)

    # Keypoints pairs:
    EYE = _left_right('eye')
    EAR = _left_right('ear')
    SHOULDER = _left_right('shoulder')
    ELBOW = _left_right('elbow')
    WRIST = _left_right('wrist')
    HAND = WRIST
    HIP = _left_right('hip')
    KNEE = _left_right('knee')
    ANKLE = _left_right('ankle')
    FOOT = ANKLE

    # Combinations:
    # Head
    FACE = (*EYE, *NOSE)
    HEAD = (*FACE, *EAR)

    # Arm
    UPPER_ARM = (*SHOULDER, *ELBOW)
    LOWER_ARM = (*ELBOW, *HAND)
    ARM = (*UPPER_ARM, *LOWER_ARM)

    # Leg
    UPPER_LEG = (*HIP, *KNEE)
    LOWER_LEG = (*KNEE, *FOOT)
    LEG = (*UPPER_LEG, *LOWER_LEG)

    # Torso
    TORSO = (*SHOULDER, *HIP)


class COCODataset(ABC, BaseDataset):
    """Attributes and functions common for keypoint based datasets derived
    from MS COCO.

    The handle encompasses functionality for:

    :item retrieval:
        :py:meth:`hybrid_learning.datasets.base.BaseDataset.__getitem__` must be
        implemented in subclasses; available items:

        - *raw annotations:*
          Get all raw COCO annotations for an image via :py:meth:`raw_anns`.
        - *image file path:*
          via :py:meth:`image_filepath`.
        - *image meta info:*
          via :py:meth:`image_meta`.
        - *image attribution and license info:*
          via :py:meth:`image_attribution`.

    :subsetting:
        Use :py:meth:`subset` for subsetting according to given conditions.
        To create/store annotations reflecting this subset, use
        :py:meth:`to_raw_anns`.
    """
    # Some COCO specific constants
    # ----------------------------
    _KEY_FILE_NAME = 'file_name'
    """Key for the ``"file name"`` property in COCO image annotations."""

    COMMERCIAL_LICENSE_IDS: Tuple[int] = (4, 5, 6, 7, 8)
    """IDs of COCO image licenses that allow for commercial use."""

    DATASET_ROOT_TEMPL: str = os.path.join(
        '..', 'dataset', 'coco', 'images', '{split}2017')
    """Default root directory template for image files that accepts the
    ``split`` (``'train'`` or ``'val'``)."""

    ANNOTATION_FP_TEMPL: str = os.path.join(
        '{root}', 'annotations', 'person_keypoints_{split}2017.json')
    """Default template for the annotation file path that accepts the
    ``split`` (``'train'`` or ``'val'``) and the ``root`` directory."""

    DEFAULT_IMG_SIZE: Tuple[int, int] = (400, 400)
    """Default target size of images to use for the default transforms."""

    def __init__(self,
                 dataset_root: str = None,
                 annotations_fp: str = None,
                 split: Union[DatasetSplit, str] = None,
                 **kwargs):
        """Init.

        :param dataset_root: root directory under which to find
            images and annotations
        :param annotations_fp: path to the annotations file
        :param split: the dataset split (as string specifier or directly as
            :py:class:`~hybrid_learning.datasets.base.DatasetSplit`);
            must be given in case the default ``dataset_root`` or
            ``annotations_fp`` are to be used
        :param kwargs: further arguments for super class ``__init__()``
        """
        self._default_transforms: Callable[[PIL.Image.Image, Any],
                                           Tuple[torch.Tensor, Any]] \
            = self._get_default_transforms(self.DEFAULT_IMG_SIZE)
        """Default transformation for tuples of COCO image and target."""

        # Root directory under which to find the images
        if isinstance(split, str):
            split: DatasetSplit = DatasetSplit[split.upper()]
        if dataset_root is None or annotations_fp is None:
            split = split if split is not None else DatasetSplit.TRAIN
        if dataset_root is None:
            dataset_root = self.DATASET_ROOT_TEMPL.format(split=split.value)
        super(COCODataset, self).__init__(dataset_root=dataset_root,
                                          split=split,
                                          **kwargs)

        # Annotation file
        self.annotations_fp = annotations_fp if annotations_fp is not None \
            else self._default_annotations_fp
        """File path to the COCO annotations json file with image and
        keypoint annotations."""

        # region Sanity check: Does annotation file exist?
        if not os.path.exists(self.annotations_fp) \
                or not os.path.isfile(self.annotations_fp):
            raise FileNotFoundError(
                "Annotation file {} not found".format(self.annotations_fp))
        if not os.path.exists(dataset_root) \
                or not os.path.isdir(dataset_root):
            raise NotADirectoryError(
                "Dataset root folder {} not found".format(self.dataset_root))
        # endregion

        # COCO handle
        with open(os.devnull, "w") as dnull, contextlib.redirect_stdout(dnull):
            self.coco: COCO = COCO(self.annotations_fp)
            """Internal COCO handle."""

        # Actual image selection
        self.img_ids: List[int] = self._get_person_img_ids(self.coco)
        """Mapping of indices in this dataset to COCO image IDs."""

    def subset(self, *,
               license_ids: Optional[Iterable[int]] = COMMERCIAL_LICENSE_IDS,
               body_parts: Optional[Iterable[BodyParts]] = None,
               keypoint_names: Optional[Iterable[str]] = None,
               num: Optional[int] = None,
               shuffle: bool = False) -> 'COCODataset':
        """Restrict the items by the given selection criteria.
        Operation changes :py:attr:`img_ids`.
        Selection criteria are:

        - Len: Maximum total number of images (and whether to shuffle before
          selecting the first X IDs)
        - License: IDs of licenses one of which the image must have
        - Contained keypoints: the names of keypoints one of which must be
          contained in the image

        :param license_ids: IDs of accepted licenses;
            if set to ``None``, all licenses are accepted
        :param body_parts: body parts that must (all) be visible in the image
        :param keypoint_names: names of keypoints that must be visible
            (alternative to ``body_parts``; see :py:meth:`to_keypoint_idxs`)
        :param num: number of images to produce
            (take first ``num`` ones)
        :param shuffle: whether to shuffle the IDs
            (before applying ``num``)
        """
        if isinstance(body_parts, BodyParts):
            body_parts = [body_parts]
        if isinstance(keypoint_names, str):
            keypoint_names = [keypoint_names]
        if keypoint_names is None and body_parts is not None:
            keypoint_names = [k for part in body_parts for k in part]
        keypoint_idxs = None if keypoint_names is None else \
            self.to_keypoint_idxs(self.coco, keypoint_names)

        self.img_ids: List[int] = list(self.img_id_iterator(
            coco=self.coco,
            img_ids=self.img_ids,
            license_ids=license_ids,
            keypoint_idxs=keypoint_idxs,
            num=num,
            shuffle=shuffle))
        return self

    def to_raw_anns(self, description: str = None, save_as: str = None):
        """Create the content of a new valid annotations file restricted
        to the current image IDs.
        Optionally also save to ``save_as``.
        The content is based on the contents of :py:attr:`annotations_fp`.

        If the restricted content is stored into a JSON annotations file,
        it can be used to init further instances that are already restricted
        to the current images.
        Useful in combination with :py:meth:`subset`.

        :param description: change the ``'description'`` under ``'info'`` to
            ``description``; no change if set to ``None``
        :param save_as: if not ``None``, dump the new annotations to a
            file located there (will overwrite)
        :return: dict in the format of a COCO annotations file with
            images and annotations restricted to those used in this
            dataset instance
        """
        with open(self.annotations_fp) as ann_file:
            anns = json.load(ann_file)

        # info
        if description is not None:
            anns['info']['description'] = description

        # images
        anns['images'] = [self.image_meta(i) for i in range(len(self))]
        anns['annotations'] = [ann for i in range(len(self))
                               for ann in self.raw_anns(i)]

        if save_as is not None:
            with open(save_as, 'w+') as anns_file:
                json.dump(anns, anns_file, indent=4)
        return anns

    def copy_to(self,
                root_root: str = None,
                description: str = None,
                overwrite: bool = False,
                dataset_root: str = None) -> Dict[str, str]:
        """Create a new dataset by copying used images and annotations
        to new root folder.

        The following files and folders will be created:

        - ``new_root_root/annotations/``: annotations root folder
        - ``new_root_root/annotations/<anns_file>``:
          An annotations file of the same basename as :py:attr:`annotations_fp`
          is created and stored in the annotations folder
          (see :py:meth:`to_raw_anns`).
        - ``new_root_root/images/<img_root>/``:
          An images root is created of the same basename as
          :py:attr:`hybrid_learning.datasets.base.BaseDataset.dataset_root`.
        - ``new_root_root/images/<img_root>/<img_file>``:
          Each image file used in this dataset is copied to the new images root
          keeping the file basename.

        :param root_root: root directory under which to create the
            annotations root and new ``dataset_root``
        :param description: description used in the annotations info;
            see :py:meth:`to_raw_anns`
        :param overwrite: do not raise if file or folder exist
        :param dataset_root: if new_root_root is not given, it is assumed
            to be ``dataset_root/../..``.
        :return: a dict with the ``dataset_root`` and ``annotations_fp``
            settings to init the new dataset
        """
        # folders
        if root_root is None:
            root_root = os.path.dirname(os.path.dirname(dataset_root))
        anns_root: str = os.path.join(root_root, "annotations")
        dataset_root: str = os.path.join(root_root, "images",
                                         os.path.basename(self.dataset_root))
        os.makedirs(anns_root, exist_ok=True)
        os.makedirs(dataset_root, exist_ok=overwrite)

        # annotations
        anns_fp = os.path.join(anns_root, os.path.basename(self.annotations_fp))
        if not overwrite and os.path.exists(anns_fp):
            raise FileExistsError(("Annotations file {} exits and overwrite "
                                   "is disabled!").format(anns_fp))
        self.to_raw_anns(description=description, save_as=anns_fp)

        # images
        for i in range(len(self)):
            src_fp = self.image_filepath(i)
            dst_fp = os.path.join(dataset_root, os.path.basename(src_fp))
            if not overwrite and os.path.exists(dst_fp):
                raise FileExistsError(("Image file {} exits and overwrite "
                                       "is disabled!").format(dst_fp))
            shutil.copyfile(src_fp, dst_fp)

        return dict(dataset_root=dataset_root,
                    annotations_fp=anns_fp)

    @classmethod
    def img_id_iterator(cls,
                        coco: Optional[COCO],
                        img_ids: Optional[Sequence[int]] = None,
                        license_ids: Optional[
                            Iterable[int]] = COMMERCIAL_LICENSE_IDS,
                        keypoint_idxs: Optional[Sequence[int]] = None,
                        num: Optional[int] = None,
                        shuffle: bool = False
                        ) -> Generator[int, None, None]:
        """Generator that iterates over image IDs of the COCO dataset that
        fulfill given selection criteria.
        For details see :py:meth:`subset`.

        :param coco: coco handler to get ``img_ids``
        :param img_ids: IDs to select from;
            defaults to all image IDs from ``coco``
        :param license_ids: IDs of accepted licenses;
            if set to ``None``, all licenses are accepted
        :param keypoint_idxs: indices of keypoints in COCO annotations that
            must be visible (cf. :py:meth:`to_keypoint_idxs`)
        :param num: number of images to produce (take first ``num`` ones)
        :param shuffle: whether to shuffle the IDs
            (before applying effect of ``num``)
        """
        # Sort out categories
        if img_ids is None:
            if coco is None:
                raise ValueError("Both coco and img_ids are None")
            img_ids = cls._get_person_img_ids(coco)

        # Stopping criterion
        if shuffle:
            img_ids = random.sample(img_ids, k=len(img_ids))

        num_selected_imgs: int = 0
        for img_id in img_ids:
            # Selection by license:
            if license_ids is not None and \
                    coco.imgs[img_id]['license'] not in license_ids:
                continue

            # Selection by feature:
            if keypoint_idxs is not None and any(
                    not cls.all_keypts_visible(ann, keypoint_idxs)
                    for ann in coco.imgToAnns[img_id]):
                continue

            yield img_id

            # Braking condition:
            num_selected_imgs += 1
            if num is not None and \
                    num_selected_imgs >= num:
                break

    @classmethod
    def _get_person_img_ids(cls, coco: COCO):
        """Get the image IDs for category ``'person'``."""
        cat_ids: List[int] = coco.getCatIds(catNms=['person'])
        img_ids: List[int] = coco.getImgIds(catIds=cat_ids)
        return img_ids

    @staticmethod
    def to_keypoint_idxs(coco: COCO, keypoint_names: Iterable[str]
                         ) -> List[int]:
        """Return the start indices for the given keypoints within an
        annotations ``'keypoint'`` field.

        Keypoints

            ``{keypt_name1: (x1,y1,visibility1), keypt_name2: (...), ...}``

        are in COCO annotated as a list of

            ``[x1, y1, visibility1,   x2, y2, ...]``

        This method returns the index of the x-coordinates within the
        annotation list.

        :param coco: coco handler
        :param keypoint_names: list of keypoint names to collect the start
            indices for
        :return: list with the start indices for the given keypoints
        """
        if isinstance(keypoint_names, str):
            raise ValueError("keypoint_names is str not iterable of strings: {}"
                             .format(keypoint_names))
        all_keypoint_names: list = [cat for cat in coco.cats.values()
                                    if cat['name'] == 'person'][0]['keypoints']
        keypoint_idxs = []
        for kpt_name in keypoint_names:
            if kpt_name not in all_keypoint_names:
                raise KeyError(("keypoint name {} not in available keypoint "
                                "names {}").format(kpt_name,
                                                   all_keypoint_names))
            keypoint_idxs.append(3 * all_keypoint_names.index(kpt_name))
        return keypoint_idxs

    @staticmethod
    def all_keypts_visible(ann, keypoint_idxs: Optional[Sequence[int]]
                           ) -> bool:
        """Whether all given keypoints are marked as visible in the given
        annotation.

        :param ann: coco annotation to check
        :param keypoint_idxs: keypoint constraints:
            starting indices of keypoints in the ``keypoint`` list of the
            annotation that must be marked as visible
            (cf. :py:meth:`to_keypoint_idxs`);
            if ``None`` or empty, ``True`` is returned for the annotation
            to be regarded valid
        :return: whether all specified keypoints are marked as visible in the
            annotation
        """
        # Keypoint constraints empty?
        if keypoint_idxs is None or len(keypoint_idxs) == 0:
            return True

        # Any keypoint annotated?
        if not ann['num_keypoints'] >= 1:
            return False

        # Needed keypoints visible?
        keypoint_infos: List[List[int]] = [ann['keypoints'][idx:idx + 3]
                                           for idx in keypoint_idxs]
        # for each info: info[2] is visibility info, a value of 2 means visible
        if not all([k[2] == 2 for k in keypoint_infos]):
            return False
        return True

    @classmethod
    @abc.abstractmethod
    def _get_default_transforms(cls, img_size: Tuple[int, int]
                                ) -> TupleTransforms:
        """Create the default transformation for this dataset depending on
        ``img_size``."""
        raise NotImplementedError()

    @property
    def _default_annotations_fp(self) -> str:
        """Default file path to the annotations derived from
        :py:attr:`hybrid_learning.datasets.base.BaseDataset.dataset_root`
        and spec."""
        return self.ANNOTATION_FP_TEMPL.format(
            split=("val" if self.split in [DatasetSplit.VAL, DatasetSplit.TEST]
                   else "train"),
            root=os.path.dirname(os.path.dirname(self.dataset_root)))

    @property
    def settings(self) -> Dict[str, Any]:
        """Return information to init new dataset."""
        return dict(annotations_fp=self.annotations_fp,
                    **super().settings)

    def load_orig_image(self, i: int) -> PIL.Image.Image:
        """Load unmodified image by index in dataset."""
        img_fp = self.image_filepath(i)
        img: PIL.Image.Image = PIL.Image.open(img_fp).convert('RGB')
        return img

    def image_filepath(self, i: int) -> str:
        """Path to image file at index ``i``."""
        img_fn: str = self.image_meta(i)[COCODataset._KEY_FILE_NAME]
        return os.path.join(self.dataset_root, img_fn)

    def image_attribution(self, i: int) -> Dict[str, Union[int, str]]:
        # pylint: disable=line-too-long
        """Get attribution information for image at index ``i``.
        This encompasses:

        :file_path: the image file path
        :source: the static flickr URL (image source)
        :flickr_page: link to the image flickr page featuring author information
        :license: the license name
        :license_url: link to the license
        :coco_id: the COCO image ID
        :coco_license_id: the COCO license ID

        The license information is taken from :py:meth:`license_mapping`.
        See also

        - `best practices regarding image attribution information
          <https://wiki.creativecommons.org/wiki/best_practices_for_attribution>`_, and
        - `information on how to get the flickr page of an image via its
          static URL <https://hadinur.net/2013/11/02/how-to-find-the-original-flickr-photo-url-and-user-from-a-static-flickr-image-urlpermalink/>`_
        """
        # pylint: enable=line-too-long
        meta = self.image_meta(i)
        source = meta['flickr_url']
        flickr_id = os.path.basename(source).split('_')[0]
        flickr_page = 'http://flickr.com/photo.gne?id=' + flickr_id
        coco_license_id = meta['license']
        license_info = self.license_mapping[coco_license_id]

        return {'file_path': self.image_filepath(i),
                'source': source,
                'flickr_page': flickr_page,
                'license': license_info['name'],
                'license_url': license_info['url'],
                'coco_id': meta['id'],
                'coco_license_id': license_info['id'], }

    def image_meta(self, i: int) -> Dict[str, Any]:
        """Load the dict with meta information for image at index ``i``."""
        return self.coco.imgs[self.img_ids[i]]

    def raw_anns(self, i: int) -> List[Dict[str, Any]]:
        """Return the list of raw annotations for image at index ``i``."""
        return self.coco.imgToAnns[self.img_ids[i]]

    @property
    def license_mapping(self) -> Dict[int, Dict[str, Any]]:
        """The mapping of image IDs to license descriptions and URLs.
        This is extracted from the annotations file loaded by :py:attr:`coco`.

        :return: dict ``{ID: license_info}``"""
        return {license_info["id"]: license_info
                for license_info in self.coco.dataset["licenses"]}

    def __len__(self):
        """Length is given by the length of the index mapping."""
        return len(self.img_ids)

    @abc.abstractmethod
    def getitem(self, item: int):
        """Item selection differs depending on the desired annotation data.
        Used for
        :py:meth:`~hybrid_learning.datasets.base.BaseDataset.__getitem__`."""
        raise NotImplementedError()


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
        super(KeypointsDataset, self).__init__(**kwargs)

    @classmethod
    def _get_default_transforms(cls, img_size: Tuple[int, int]
                                ) -> TupleTransforms:
        """Return the default transformation, which is pad and resize."""
        return trafos.Compose(
            [trafos.OnInput(tv.transforms.ToTensor()),
             trafos.OnInput(trafos.PadAndResize(img_size))]
        )

    def getitem(self, i: int) -> Tuple[PIL.Image.Image, List[Dict]]:
        """Collect the image and the keypoint annotations at position ``i``.
        Used for
        :py:meth:`~hybrid_learning.datasets.base.BaseDataset.__getitem__`."""
        annotations: List[Dict] = self.raw_anns(i)
        img: PIL.Image.Image = self.load_orig_image(i)
        return img, annotations


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
    _MASKS_ROOT_ROOT = "masks"
    """Usual parent to all masks folders for all body parts; sibling to
    ``images`` folder"""

    def __init__(self,
                 body_parts: Sequence[BodyParts],
                 force_rebuild: bool = False,
                 pt_radius: float = None,
                 masks_root: str = None,
                 lazy_mask_generation: bool = True,
                 **kwargs):
        """Init.

        :param body_parts: see :py:attr:`body_parts`
        :param force_rebuild: Whether to regenerate all masks during init,
            or to skip existing ones.
        :param pt_radius: radius of white circles to draw centered
            at the keypoint coordinates; unit: proportion of image width
        :param lazy_mask_generation: whether to generate the masks lazily or
            directly during init;
            overwritten by ``force_rebuild``: if this is true,
            masks are directly regenerated.
        :param kwargs: Any other arguments for the super class
        """
        # Application of general spec
        super(ConceptDataset, self).__init__(**kwargs)
        if pt_radius is not None and pt_radius <= 0:
            raise ValueError(("pt_radius to generate masks must be > 0 "
                              "but was {}").format(pt_radius))
        self.pt_radius: float = pt_radius or 10 / 400

        # Concepts
        self.body_parts: Sequence[BodyParts] = body_parts
        """List of concepts that must be contained in selected images."""

        # The masks root folder
        self.masks_root = masks_root or self._default_masks_root
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
                    masks_root=self.masks_root)

    @property
    def _default_masks_root(self) -> str:
        """Default mask root directory obtained from
        :py:attr:`hybrid_learning.datasets.base.BaseDataset.dataset_root`
        and spec."""
        concept_hash = '{}_rad{:.3}' \
            .format('-'.join(sorted(self.keypoint_names)), self.pt_radius)
        masks_basename = "{img_base}_{concept_hash}".format(
            img_base=os.path.basename(self.dataset_root),
            concept_hash=concept_hash)
        masks_dirname = os.path.join(
            os.path.dirname(os.path.dirname(self.dataset_root)),
            "masks"
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
            mask = self.annotations_to_mask(orig_img_size=img.size,
                                            annotations=anns)
            self.save_mask(i, mask)

        mask: PIL.Image.Image = self.load_orig_mask(i)

        return img, mask

    def load_orig_mask(self, i: int) -> PIL.Image.Image:
        """Load and return unmodified mask as image by index in dataset."""
        return PIL.Image.open(self.mask_filepath(i)).convert(mode='1')

    @property
    def keypoint_names(self) -> Set[str]:
        """Names of keypoints which represent body parts shown in this COCO
        subset."""
        return {kpt for body_part in self.body_parts for kpt in body_part}

    @property
    def keypoint_idxs(self) -> Set[int]:
        """IDs of keypoints which represent body parts shown in this
        COCO subset."""
        return set(self.to_keypoint_idxs(self.coco, self.keypoint_names))

    @classmethod
    def _get_default_transforms(cls, img_size: Tuple[int, int]
                                ) -> TupleTransforms:
        """Default transformation that pads and resizes both images and masks,
        and binarizes the masks."""
        return trafos.Compose(
            [trafos.OnBothSides(tv.transforms.ToTensor()),
             # pad an resize both img and mask:
             trafos.OnBothSides(trafos.PadAndResize(img_size)),
             # binarize the mask:
             trafos.OnTarget(trafos.Binarize())]
        )

    def mask_exists(self, i: int) -> bool:
        """Check whether a mask for the specified index already exists."""
        # Is it a valid ID?
        if not 0 <= i < len(self.img_ids):
            m_exists = False

        # Does the mask already exist?
        else:
            mask_fp = self.mask_filepath(i)
            m_exists = os.path.exists(mask_fp)
        return m_exists

    def mask_filepath(self, i: int) -> str:
        """Provide the path under which the mask for the given image ID
        should lie."""
        img_fn: str = self.image_meta(i)[COCODataset._KEY_FILE_NAME]
        return os.path.join(self.masks_root, img_fn)

    def annotations_to_mask(self, orig_img_size: Tuple[int, int],
                            annotations: List[dict]
                            ) -> PIL.Image.Image:
        """Create mask of all configured keypoints from ``annotations``.
        Keypoints are specified by :py:attr:`keypoint_idxs`.
        It is assumed that the original image considered in ``annotations``
        has ``orig_img_size``, which is also the size of the created mask.

        :param orig_img_size: :py:class:`PIL.Image.Image` size of the
            original image and output size of the mask
        :param annotations: annotations from which to create mask
        :return: mask (same size as original image)
        """
        # start with empty mask
        mask = PIL.Image.new('1', orig_img_size)
        draw = PIL.ImageDraw.Draw(mask)
        abs_pt_radius: int = int(math.ceil(self.pt_radius * orig_img_size[0])) # TODO: choose max of width or height instead of statically width?

        for annotation in annotations:
            # obtain all keypoints:
            # The keypoints annotations have format [x1,y1,v1,x2,y2,...];
            # turn this into numpy arrays
            # xs=[x1,x2,..], ys=[y1,y2,..], vs=[v1,v2,..]:
            kpt_xs, kpt_ys, visibilities = \
                tuple(zip(*np.array_split(
                    # kpts in format [x1,y1,v1,x2,..]:
                    np.array(annotation['keypoints']),
                    # number of kpts:
                    indices_or_sections=len(annotation['keypoints']) // 3)))

            def in_scope_and_visible(idx, vis=visibilities):
                """Whether a keypoint at given index is to be considered and
                visible."""
                return vis[idx] == 2 and idx * 3 in self.keypoint_idxs

            # sanity check: are all keypoints within image?
            for kpt_x, kpt_y in zip(kpt_xs, kpt_ys):
                if not (0 <= kpt_x <= orig_img_size[0]
                        and 0 <= kpt_y <= orig_img_size[1]):
                    raise ValueError(
                        ("Keypoint coordinates ({}, {}) out of bounds of"
                         " image size ({}, {})"
                         ).format(kpt_x, kpt_y, *orig_img_size))

            # get skeleton; see also pycocotools
            skeleton: np.ndarray = np.array(
                self.coco.loadCats(annotation['category_id'])[0]['skeleton']
            ) - 1

            # draw skeleton
            for link in skeleton:
                # sanity check: indices out of range?
                if link[0] >= len(visibilities) or link[1] >= len(visibilities):
                    raise ValueError(
                        ("Skeleton link (from_idx={}, to_idx={}) out of range "
                         "of range of visibilities (length {})"
                         ).format(link[0], link[1], len(visibilities)))
                # if visible and of interest, draw link
                if all([in_scope_and_visible(link[i]) for i in (0, 1)]):
                    draw.line(xy=[(kpt_xs[link[0]], kpt_ys[link[0]]),
                                  (kpt_xs[link[1]], kpt_ys[link[1]])],
                              width=2 * abs_pt_radius,
                              fill=1)

            # draw keypoints
            for kpt_x, kpt_y in \
                    [(kpt_xs[i], kpt_ys[i]) for i in range(len(visibilities))
                     if in_scope_and_visible(i)]:
                draw.ellipse(
                    (kpt_x - abs_pt_radius, kpt_y - abs_pt_radius,
                     kpt_x + abs_pt_radius, kpt_y + abs_pt_radius),
                    fill=1)

        return mask

    def generate_masks(self, force_rebuild: bool = False,
                       show_progress_bar: bool = True,
                       **kwargs) -> None:
        """Generate and save the masks for the images in this dataset.

        :param force_rebuild: whether to overwrite masks that already exist.
        :param show_progress_bar: whether to show the progress using
            :py:class:`tqdm.tqdm`
        :param kwargs: further arguments to the progress bar
        """
        masks_to_process = [i for i in range(len(self.img_ids))
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
            mask = self.annotations_to_mask(orig_img_size=img_size,
                                            annotations=anns)
            self.save_mask(i, mask)

    def save_mask(self, i: int, mask: PIL.Image.Image) -> None:
        """Save the given mask according to the given index in the dataset.

        :param i: ID of the image the mask belongs to.
        :param mask: mask image
        """
        mask.save(self.mask_filepath(i))
