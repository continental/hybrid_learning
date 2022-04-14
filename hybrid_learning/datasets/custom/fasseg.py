"""Dataset handles for `FASSEG-like datasets <fasseg-datasets>`_.

.. _fasseg-datasets: https://github.com/massimomauro/FASSEG-repository
"""
#  Copyright (c) 2022 Continental Automotive GmbH

import enum
import os
from typing import List, Tuple, Union, Optional

import PIL.Image
import numpy as np

from .. import transforms as trafos
from ..base import BaseDataset


class FASSEGParts(tuple, enum.Enum):
    """Mapping of part names to colors in FASSEG dataset."""
    EYES = (0, 0, 255)
    NOSE = (0, 255, 255)
    SKIN = (255, 255, 0)
    HAIR = (127, 0, 0)  # incl. brows: "background",
    BACKGROUND = (255, 0, 0)
    MOUTH = (0, 255, 0)


class FASSEGHandle(BaseDataset):
    """Handle for FASSEG-like datasets.

    .. note::
        The original FASSEG dataset is not required for this handle.
        Any dataset with a format following that of the FASSEG dataset is
        supported (folder structure, file types, color codes).

    The FASSEG dataset can be found here::
    https://github.com/massimomauro/FASSEG-repository

    The required structure for supported datasets is very simple

    - under :py:attr:`~hybrid_learning.datasets.base.BaseDataset.dataset_root`
      all input images
    - under :py:attr:`annotations_root`
      all annotation files with the same file name as the corresponding input
      image

    Once a handle is instantiated, its ``__getitem__`` yields tuples
    of input image and binary part segmentation mask (possibly transformed
    by :py:attr:`hybrid_learning.datasets.base.BaseDataset.transforms`).
    For details see :py:meth:`getitem`.
    """

    @property
    def part_name(self) -> Optional[str]:
        """The string name of the part that is extracted from the masks."""
        return getattr(self.part, 'name', self._part_name)

    def __init__(self, dataset_root: str, annotations_root: str = None,
                 part: Union[FASSEGParts, Tuple[int, int, int]] = None,
                 part_name: str = None,
                 **kwargs):
        """Init.

        :param dataset_root: the directory under which to find the images;
        :param annotations_root: the directory under which to find the
            segmentation masks; assumes as default that

            - ``dataset_root`` is :file:`{path}/{split}_RGB` and
            - ``annotations_root`` is :file:`{path}/{split}_Labels`

        :param part: the :py:class:`FASSEGParts` instance to use the color of
            or the color tuple
        :param part_name: the string name of the ``part`` that is extracted;
            overridden by ``part.name`` if ``part`` features such an attribute
        :param kwargs: parameters for
            :py:class:`~hybrid_learning.datasets.base.BaseDataset`
        """
        # Value checks and default values
        if part is None and part_name is None:
            raise ValueError("Part specification or part_name needed!")
        part = part if part is not None else FASSEGParts[part_name]
        # Dataset root exists?
        if not os.path.isdir(dataset_root):
            raise NotADirectoryError("Given dataset_root {} does not exist"
                                     .format(dataset_root))
        # Annotations root (given or default) exists?
        if annotations_root is not None and not os.path.isdir(annotations_root):
            raise NotADirectoryError("Given annotations_root {} does not exist"
                                     .format(annotations_root))
        if annotations_root is None:
            annotations_root: str = "{}Labels" \
                .format(dataset_root.rsplit("RGB", 1)[0])
            if not os.path.isdir(annotations_root):
                raise ValueError(("annotations_root not given and default {} "
                                  "does not exist").format(annotations_root))

        super().__init__(dataset_root=dataset_root, **kwargs)

        self.part = part
        """Part of the face and its color to select mask of."""
        self._part_name = part_name
        """Default for string name of the part if :py:attr:`part` has no
        attribute ``name``."""
        self.annotations_root: str = annotations_root
        """Path to the annotations root folder under which to find the
        annotation files."""
        self.img_fns: List[str] = \
            [fn for fn in os.listdir(self.dataset_root)
             if os.path.isfile(os.path.join(self.dataset_root, fn))]
        """List of file names of images (and their annotations) handled by
        this instance.
        Images can be found in the
        :py:attr:`~hybrid_learning.datasets.base.BaseDataset.dataset_root` and
        annotations in :py:attr:`annotations_root`.
        These are used for :py:meth:`getitem`."""

        # Are there labels for all files?
        for i in range(len(self.img_fns)):
            if not os.path.isfile(self.mask_filepath(i)):
                raise FileNotFoundError(
                    "For input image {} no mask file {} exists".format(
                        os.path.isfile(os.path.join(self.dataset_root,
                                                    self.img_fns[i])),
                        os.path.isfile(os.path.join(self.annotations_root,
                                                    self.img_fns[i]))))

    def __len__(self):
        return len(self.img_fns)

    # noinspection PyTypeChecker
    def getitem(self, i: int) -> Tuple[PIL.Image.Image, PIL.Image.Image]:
        """Load image and its mask at index ``i`` and select binary part mask.

        Used for
        :py:meth:`~hybrid_learning.datasets.base.BaseDataset.__getitem__`.
        The value of the return tuple ``(input_img, part_mask)`` are
        :py:class:`torch.Tensor` representations of
        :py:class:`PIL.Image.Image` image instances.
        """
        img: PIL.Image.Image = PIL.Image.open(self.image_filepath(i))
        mask: PIL.Image.Image = PIL.Image.open(self.mask_filepath(i))

        # obtain binary mask for specific part:
        rgb_color: np.ndarray = np.array(self.part)
        part_mask: PIL.Image.Image = PIL.Image.fromarray(
            np.all(np.array(mask) == rgb_color, axis=-1))

        return img, part_mask

    def descriptor(self, i: int) -> str:
        """Return the image file name for index ``i``.
        This is unique throughout a FASSEG like dataset and can be used for
        e.g. image IDs for caching."""
        return self.img_fns[i]

    def image_filepath(self, i):
        """Provide the path to the image at index ``i``."""
        return os.path.join(self.dataset_root, self.descriptor(i))

    def mask_filepath(self, i):
        """Provide the path to the mask at index ``i``."""
        return os.path.join(self.annotations_root, self.descriptor(i))

    _default_transforms: trafos.TupleTransforms = \
        trafos.OnBothSides(trafos.ToTensor())
    """Default transformation function transforming images to
    :py:class:`torch.Tensor`."""
