"""Base classes for handling MS COCO like datasets."""
#  Copyright (c) 2022 Continental Automotive GmbH

import abc
import contextlib
import json
import os
import random
import shutil
from abc import ABC
from typing import Optional, Generator, List, Sequence, Iterable, \
    Tuple, Callable, Any, Dict, Union

import PIL.Image
import PIL.ImageDraw
import torch
from pycocotools.coco import COCO
from tqdm import tqdm

from .keypoints_processing import to_keypoint_idxs, all_keypts_visible
from ...base import BaseDataset, DatasetSplit
from ...transforms import TupleTransforms


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
    """Default target size of images to use for the default transforms
    as ``(height, width)``."""

    def __init__(self,
                 dataset_root: str = None,
                 annotations_fp: str = None,
                 split: Union[DatasetSplit, str] = None,
                 img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
                 device: Optional[Union[str, torch.device]] = None,
                 **kwargs):
        # pylint: disable=line-too-long
        """Init.

        :param dataset_root: root directory under which to find
            images and annotations
        :param annotations_fp: path to the annotations file
        :param split: the dataset split (as string specifier or directly as
            :py:class:`~hybrid_learning.datasets.base.DatasetSplit`);
            must be given in case the default ``dataset_root`` or
            ``annotations_fp`` are to be used
        :param img_size: a convenience argument to set a default transformation
            that resizes images to ``img_size``;
            format: ``(height, width)`` in pixels;
            defaults to
            :py:attr:`hybrid_learning.datasets.custom.coco.base.COCODataset.DEFAULT_IMG_SIZE`
        :param device: the device to run the default trafos on and move
            loaded items to
        :param kwargs: further arguments for super class ``__init__()``
        """
        # pylint: enable=line-too-long
        if self._default_transforms is None:
            self._default_transforms: Callable[[PIL.Image.Image, Any],
                                               Tuple[torch.Tensor, Any]] \
                = self.get_default_transforms(img_size=img_size, device=device)
            """Default transformation for tuples of COCO image and target."""

        # Root directory under which to find the images
        if isinstance(split, str):
            split: DatasetSplit = DatasetSplit[split.upper()]
        if dataset_root is None or annotations_fp is None:
            split = split if split is not None else DatasetSplit.TRAIN
        if dataset_root is None:
            dataset_root = self.DATASET_ROOT_TEMPL.format(split=split.value)
        super().__init__(dataset_root=dataset_root,
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
        self.img_ann_ids: List[Tuple[int, List[int]]] = \
            self._get_person_img_ids(self.coco)
        """Mapping of indices in this dataset to COCO image and
        annotation IDs.
        Each entry in the list is a tuple of the form
        ``(image_id, [annotation_id, ...])`` where the annotations belong
        to the corresponding image."""

    def subset(
            self, *,
            license_ids: Optional[Iterable[int]] = COMMERCIAL_LICENSE_IDS,
            body_parts: Optional[Union[Sequence[str],
                                       Sequence[Sequence[str]]]] = None,
            num: Optional[int] = None,
            shuffle: bool = False,
            condition: Callable[[Dict[str, Any], Dict[str, Any]], bool] = None,
            show_progress_bar: bool = True
    ) -> 'COCODataset':
        """Restrict the items by the given selection criteria and an optional
        custom condition.
        Operation changes :py:attr:`img_ann_ids`.
        Selection criteria are:

        - Len: Maximum total number of images (and whether to shuffle before
          selecting the first X IDs)
        - License: IDs of licenses one of which the image must have
        - Contained body parts: body parts (collections of of keypoint names),
            one of which must be fully contained in the image;
            e.g. ``[["left_eye"], ["right_eye"]]`` means either ``left_eye`` or
            ``right_eye`` must be visible, while ``[["left_eye", "right_eye"]]``
            means both must be.
        - Any custom condition specified via ``condition``.

        :param license_ids: IDs of accepted licenses;
            if set to ``None``, all licenses are accepted
        :param body_parts: sequence of body parts any of which must be visible
            in the image;
            a body part is a sequence of string keypoint names;
            if just one body part is given, this may be provided as list of
            strings, e.g. ``["left_eye", "right_eye"]``
        :param num: number of images to produce
            (take first ``num`` ones)
        :param shuffle: whether to shuffle the IDs
            (before applying ``num``)
        :param condition: callable that accepts image meta data and annotation
            metadata, and returns a bool stating whether to skip the annotation
            instance or not
        :param show_progress_bar: whether to show the progress of image checking
        """
        # region Default value for filter condition
        # Add annotation selection by feature visibility:
        filters: List[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = []
        if body_parts is not None:
            if isinstance(body_parts[0], str):
                body_parts: List[Sequence[str]] = [body_parts]
            keypoint_idxs = [to_keypoint_idxs(part, coco=self.coco)
                             for part in body_parts]
            filters.append(lambda _, ann: any((all_keypts_visible(ann, idxs)
                                               for idxs in keypoint_idxs)))
        # Add annotation selection by custom condition:
        if condition is not None:
            filters.append(condition)
        filter_condition = None if len(filters) == 0 else \
            (lambda i_m, a_m: all((filt(i_m, a_m) for filt in filters)))
        # endregion

        self.img_ann_ids: List[Tuple[int, List[int]]] = list(
            self.img_id_iterator(
                coco=self.coco,
                img_ann_ids=self.img_ann_ids,
                license_ids=license_ids,
                num=num,
                shuffle=shuffle,
                condition=filter_condition,
                show_progress_bar=show_progress_bar
            ))
        return self

    def shuffle(self) -> 'COCODataset':
        """Wrapper around :py:meth:`subset` that only shuffles the instance.

        :return: self
        """
        return self.subset(shuffle=True, license_ids=None)

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
    def img_id_iterator(
            cls,
            coco: Optional[COCO],
            img_ann_ids: Optional[Sequence[Tuple[int, Sequence[int]]]] = None,
            license_ids: Optional[Iterable[int]] = COMMERCIAL_LICENSE_IDS,
            num: Optional[int] = None,
            shuffle: bool = False,
            condition: Optional[Callable[[Dict[str, Any], Dict[str, Any]],
                                         bool]] = None,
            show_progress_bar: bool = True,
    ) -> Generator[Tuple[int, List[int]], None, None]:
        """Generator that iterates over image IDs of the COCO dataset that
        fulfill given selection criteria.
        For details see :py:meth:`subset`.

        :param coco: coco handler to get ``img_ann_ids``
        :param img_ann_ids: image and annotation IDs to select from;
            format should be as for :py:attr:`img_ann_ids`;
            defaults to IDs of all images and annotations given by ``coco``
        :param license_ids: IDs of accepted licenses;
            if set to ``None``, all licenses are accepted
        :param num: number of images to produce (take first ``num`` ones)
        :param shuffle: whether to shuffle the IDs
            (before applying effect of ``num``)
        :param condition: callable that accepts image meta data and annotation
            metadata, and returns a bool stating whether to skip the annotation
            instance or not
        :param show_progress_bar: whether to show a progress bar while
            iterating over all image IDs
        """
        # region Value checks and defaults
        if license_ids is not None and coco is None:
            raise ValueError("license_ids given but coco None")
        if img_ann_ids is None:
            if coco is None:
                raise ValueError("Both coco and img_ann_ids are None")
            img_ann_ids = cls._get_person_img_ids(coco)
        if condition is not None and coco is None:
            raise ValueError("condition give but coco None")
        # endregion

        # Shuffle
        if shuffle:
            img_ann_ids = random.sample(img_ann_ids, k=len(img_ann_ids))

        num_selected_imgs: int = 0
        img_ann_ids = img_ann_ids if not show_progress_bar else \
            tqdm(img_ann_ids, desc="COCO images checked", total=num)
        for img_id, ann_ids in img_ann_ids:
            img_meta: Dict[str, Any] = coco.imgs[img_id]

            # Image selection by license
            # (don't do via filter to avoid loading of annotations):
            if license_ids is not None and \
                    img_meta['license'] not in license_ids:
                continue

            # Annotation selection by condition:
            if condition is not None:
                ann_ids: List[int] = [a_id for a_id in ann_ids
                                      if condition(img_meta, coco.anns[a_id])]

            # Image selection by presence of any annotation:
            if len(ann_ids) == 0:
                continue

            yield img_id, ann_ids

            # Breaking condition:
            num_selected_imgs += 1
            if num is not None and \
                    num_selected_imgs >= num:
                break
        if isinstance(img_ann_ids, tqdm):
            img_ann_ids.close()

    @classmethod
    def _get_person_img_ids(cls, coco: COCO) -> List[Tuple[int, List[int]]]:
        """Get the image and annotation IDs for category ``'person'``.

        :return: a list of tuples of the form
            ``(image_id, [annotation_id, ...])`` where the ``annotation_id``
            entries are IDs of annotations within the image given by
            ``image_id``
        """
        cat_ids: List[int] = coco.getCatIds(catNms=['person'])
        img_ids: List[int] = coco.getImgIds(catIds=cat_ids)
        ids: List[Tuple[int, List[int]]] = \
            [(img_id, [ann['id'] for ann in coco.imgToAnns[img_id]])
             for img_id in img_ids]
        return ids

    @classmethod
    @abc.abstractmethod
    def get_default_transforms(cls, img_size: Tuple[int, int],
                               device: Union[str, torch.device] = None,
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
        img: PIL.Image.Image = PIL.Image.open(img_fp)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def descriptor(self, i: int) -> str:
        """Return the image file name for the item at index ``i``.
        This is unique within the COCO dataset and may serve as ID e.g. for
        caching."""
        return self.image_meta(i)[COCODataset._KEY_FILE_NAME]

    def image_filepath(self, i: int) -> str:
        """Path to image file at index ``i``."""
        return os.path.join(self.dataset_root, self.descriptor(i))

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
        return self.coco.imgs[self.img_ann_ids[i][0]]

    def raw_anns(self, i: int) -> List[Dict[str, Any]]:
        """Return the list of raw annotations for image at index ``i``."""
        return [self.coco.anns[ann_id] for ann_id in self.img_ann_ids[i][1]]

    @property
    def license_mapping(self) -> Dict[int, Dict[str, Any]]:
        """The mapping of image IDs to license descriptions and URLs.
        This is extracted from the annotations file loaded by :py:attr:`coco`.

        :return: dict ``{ID: license_info}``"""
        return {license_info["id"]: license_info
                for license_info in self.coco.dataset["licenses"]}

    def __len__(self):
        """Length is given by the length of the index mapping."""
        return len(self.img_ann_ids)

    @abc.abstractmethod
    def getitem(self, item: int):
        """Item selection differs depending on the desired annotation data.
        Used for
        :py:meth:`~hybrid_learning.datasets.base.BaseDataset.__getitem__`."""
        raise NotImplementedError()
