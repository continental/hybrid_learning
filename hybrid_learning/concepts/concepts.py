"""Concept base class, concept types and corresponding sub-classes.

A concept is represented by dataset splits.
Different concept types are implemented using sub-classes.
Depending on the type, the data sets must fulfill different criteria.
E.g. a segmentation concept requires segmentation masks as ground truth.
"""
#  Copyright (c) 2020 Continental Automotive GmbH

import enum
from collections import Iterable
from typing import Union, Tuple, Dict, Any, Optional

import torch
import torch.utils.data
from torch.utils.data import Dataset

from ..datasets import BaseDataset, DataTriple


class ConceptTypes(enum.Enum):
    """Types the data of a concept can have."""
    SEGMENTATION = 'segmentation'
    IMAGE_LEVEL = 'image_level'
    BBOX = 'bbox'


class Concept:
    """Representation of a concept with data and meta information.
    Sub-classes should implement the property type and extend the data checks in
    :py:meth:`_validate_dataset`.
    Different general types can be found in
    :py:class:`~hybrid_learning.concepts.concepts.ConceptTypes`.
    """

    @property
    def type(self) -> ConceptTypes:
        """The type of the concept including according type checks."""
        raise NotImplementedError("Implement with data property checks "
                                  "in sub-class.")

    @property
    def train_data(self) -> Dataset:
        """Training data set.
        See :py:attr:`~hybrid_learning.datasets.base.DataTriple.train`
        attribute of :py:attr:`data`."""
        return self.data.train

    @property
    def val_data(self) -> Dataset:
        """Validation data set.
        See :py:attr:`~hybrid_learning.datasets.base.DataTriple.val`
        attribute of :py:attr:`data`."""
        return self.data.val

    @property
    def test_data(self) -> Dataset:
        """Test data set.
        See :py:attr:`~hybrid_learning.datasets.base.DataTriple.test`
        attribute of :py:attr:`data`."""
        return self.data.test

    @property
    def train_val_data(self) -> Dataset:
        """Combined dataset of the defining training and validation data.
        See :py:attr:`~hybrid_learning.datasets.base.DataTriple.train_val`
        attribute of :py:attr:`data`."""
        return self.data.train_val

    def __init__(self,
                 name: str,
                 data: DataTriple):
        r"""Init.

        :param name: name and ID of the concept
            (should be descriptive and unique)
        :param data: dataset splits of data representing the concept
        """

        # ID
        self.name: str = name
        """The name, or ID, of the concept. Should be descriptive."""

        data.validate_by(self._validate_dataset)
        self.data: DataTriple = data
        """The data defining the concept, split into train/val/test data."""

    @staticmethod
    def _validate_dataset(data: Dataset, data_desc: str = "??") -> Dataset:
        """Check whether the given data fulfills all required properties and
        raise if not. Extend in sub-class.

        :meta public:
        :param data: dataset to validate
        :param data_desc: some description of the data for more informative
            error messages
        :raises: :py:exc:`ValueError`
        :return: the unchanged given dataset
        """
        # Check that data is not empty:
        if len(data) == 0:
            raise ValueError("Dataset {desc} is empty!".format(desc=data_desc))

        # Check data output format
        out = data[0]
        if not isinstance(out, tuple):
            raise ValueError(("Dataset {} output is not of type tuple "
                              "but of type {}").format(data_desc, type(out)))
        if len(out) != 2:
            raise ValueError(("Dataset {desc} output at position 0 should be "
                              "tuple of (input image, ground truth), but was of"
                              " type {} and length {}")
                             .format(type(out), len(out), desc=data_desc))
        img, _ = out
        if not isinstance(img, torch.Tensor):
            raise ValueError(("Dataset {desc} input at index 0, was no "
                              "torch.tensor but of type {}")
                             .format(type(img), desc=data_desc))
        return data

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings dict to reproduce instance. Use as keyword args for init."""
        return dict(name=self.name,
                    data=self.data)

    @classmethod
    def new(cls, concept: 'Concept'):
        """Initialize a concept from the attributes of another by references.
        This is helpful when converting concepts to specific sub-classes
        (with the corresponding data checks during init)."""
        return cls(**concept.settings)

    def __eq__(self, other):
        """Equality check by checking on values.
        Especially, datasets must be the same objects."""
        return (self.name == other.name) and \
               (self.data == other.data)

    def __repr__(self) -> str:
        """Nice printing function."""
        return "{cls}(\n    name={name},\n    data={data})".format(
            cls=self.__class__.__name__,
            name=repr(self.name),
            data=repr(self.data),
        )


class SegmentationConcept2D(Concept):
    """Concept with segmentation data."""

    def __init__(
            self, name: str,
            data: DataTriple,
            rel_size: Union[float, Tuple[float, float]] = None):
        """Init.

        For arguments ``name``, ``data`` see
        :py:class:`~hybrid_learning.concepts.concepts.Concept`.

        :param rel_size: size of the concept relative to the image size;
            either one value if quadratic, or a tuple of
            ``(relative width, relative height)``.
        """
        # Argument post-processing:
        if rel_size is not None:
            if not isinstance(rel_size, (Iterable, float, int)):
                raise ValueError(("Allowed types for rel_size are Iterable,"
                                  "float, int, but was {}"
                                  ).format(type(rel_size)))
            rel_size = tuple(rel_size) if isinstance(rel_size, Iterable) \
                else (rel_size, rel_size)
            if not len(rel_size) == 2:
                raise ValueError(("Got concept size tuple of wrong length "
                                  "({} instead of 2): {}"
                                  ).format(len(rel_size), rel_size))

        super(SegmentationConcept2D, self).__init__(name=name, data=data)
        self.rel_size: Optional[Tuple[float, float]] = rel_size
        """Size of the concept in ``(width, height)`` relative to the
        image size. If set, used by detection and segmentation concept models
        to determine kernel size.
        May be None if not given (e.g. if variance too high)."""

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings dict to reproduce instance. Use as kwargs for init."""
        return dict(**super(SegmentationConcept2D, self).settings,
                    rel_size=self.rel_size)

    @property
    def type(self) -> ConceptTypes:
        """Type of the concept, which is segmentation for this sub-class."""
        return ConceptTypes.SEGMENTATION

    @staticmethod
    def _validate_dataset(data: BaseDataset, data_desc: str = "??"):
        """Validate that the dataset yields tuples of ``(input image, mask)``.
        :meta public:
        """
        # Input are images?
        super(SegmentationConcept2D, SegmentationConcept2D)._validate_dataset(
            data, data_desc)
        img, mask = data[0]

        # Image check:
        # Correct number of axes?
        if not 2 <= len(img.size()) <= 3:
            raise ValueError(("Dataset {desc} input at index 0 is not 2D, but "
                              "has size {}").format(img.size(), desc=data_desc))
        # Valid number of channels?
        if len(img.size()) == 3 and img.size()[0] not in (1, 3):
            raise ValueError(("Dataset {desc} input at index 0 has no proper "
                              "channel information: should be at dimension 0 "
                              "with 1 or 3 channels, but img has size {}")
                             .format(img.size(), desc=data_desc))

        # Mask check:
        # Correct type?
        if not isinstance(mask, torch.Tensor):
            raise ValueError(("Dataset {desc} mask at index 0, was no "
                              "torch.tensor but of type {}"
                              ).format(type(mask), desc=data_desc))
        # Correct number of axes?
        if not 2 <= len(mask.size()) <= 3:
            raise ValueError(("Dataset {desc} mask at index 0 is not 2D, but "
                              "has size {}"
                              ).format(mask.size(), desc=data_desc))
        # Valid number of channels?
        if len(mask.size()) == 3 and mask.size()[0] != 1:
            raise ValueError(("Dataset {desc} input at index 0 has no proper "
                              "channel information: should be at dimension 0 "
                              "with 1 channel, but mask has size {}")
                             .format(img.size(), desc=data_desc))
        # Same size as image?
        if not mask.size()[-2:] == img.size()[-2:]:
            raise ValueError(("Dataset {desc} input at index 0 has input and "
                              "mask of differing size: "
                              "input size={}, mask size={}")
                             .format(img.size(), mask.size(), desc=data_desc))

        return data

    def __eq__(self, other: 'SegmentationConcept2D'):
        """In addition to
        :py:meth:`~hybrid_learning.concepts.concepts.Concept.__eq__`
        check ``rel_size``.
        """
        return (super(SegmentationConcept2D, self).__eq__(other)) and \
               (hasattr(other, "rel_size")) and \
               (self.rel_size == other.rel_size)
