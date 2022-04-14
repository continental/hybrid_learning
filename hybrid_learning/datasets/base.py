#  Copyright (c) 2022 Continental Automotive GmbH
"""Basic dataset model.

The abstract BaseDataset handle provides a ``Sequence``, which yields a
:py:class:`torch.Tensor` tuple of ``(input image, ground truth)`` upon a call to
__getitem__.
The transformations from image to tensor data can be changed.
"""

import abc
import enum
import os
from typing import Callable, Tuple, Union, Dict, List, Optional, NamedTuple, \
    Any, Sequence, ItemsView, KeysView, Hashable

import PIL.Image
import PIL.ImageEnhance
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

from . import caching
from . import transforms as trafos
from .transforms.tuple_transforms import TupleTransforms


class DatasetSplit(enum.Enum):
    """Types of dataset splits."""
    VAL = "val"
    """The validation set for quality checks/early stopping during training."""
    TRAIN = "train"
    """The training set (not for testing or validation)."""
    TEST = "test"
    """The test set for testing after finished training."""
    TRAIN_VAL = "trainval"
    """The combined training and validation set."""
    ALL = "all"
    """Combination of all training, validation, and test set."""

    def __repr__(self):
        """Nice representation."""
        return self.__class__.__name__ + '.' + self.name


class _IncompleteDataTriple(NamedTuple):
    """Intermediate storage for data triplets accepting ``None`` entries."""
    train: Dataset
    val: Dataset
    test: Dataset
    train_val: Dataset
    data: Dataset


class DataTriple:
    """Tuple of train/test/validation datasets (w/ automatic splitting if
    necessary).
    The splitting is conducted on init.
    This data structure is considered immutable.
    So, in order to re-do the splitting, create a new instance with the old
    specification.

    To access the held splits either use

    - the corresponding attributes,
    - the dict-like getter functionality, or
    - the dictionary representation of the tuple via :py:meth:`as_dict`.
    """

    DEFAULT_VAL_SPLIT: float = 0.2
    """Default validation split proportion.
    This is the proportion of :py:attr:`val` in :py:attr:`train_val`.
    """

    @property
    def train(self) -> Dataset:
        """Training data set."""
        return self._train

    @property
    def val(self) -> Dataset:
        """Validation data set."""
        return self._val

    @property
    def val_split(self) -> Optional[float]:
        r"""Value of
        :math:`\frac{len(val)} {len(val) + len(train)}`
        if none of the datasets is ``None`` or empty."""
        if self.val is None or self.train is None \
                or (len(self.val) + len(self.train)) == 0:
            return None
        return len(self.val) / (len(self.val) + len(self.train))

    @property
    def test(self) -> Dataset:
        """Testing dataset split."""
        return self._test

    @property
    def test_split(self) -> Optional[float]:
        r"""Value of
        :math:`\frac{len(test)} {len(test) + len(train\_val)}`
        if none of the datasets is ``None`` or empty."""
        if self.test is None or self.train_val is None \
                or (len(self.test) + len(self.train_val)) == 0:
            return None
        return len(self.test) / (len(self.test) + len(self.train))

    @property
    def train_val(self) -> Dataset:
        """Combined dataset of training and validation data.

        It is a concatenation of :py:attr:`train` and :py:attr:`val`
        or a permutation thereof.
        """
        return self._train_val

    @property
    def data(self) -> Dataset:
        """Concatenation of all data (train, val, test) stored in this tuple."""
        return self._data

    def __init__(
            self,
            data: Dataset = None,
            *,
            train: Dataset = None,
            val: Dataset = None,
            test: Dataset = None,
            train_val: Dataset = None,
            validator: Callable[[Dataset, str], Any] = None,
            **split_kwargs
    ):
        r"""Init.

        Exactly one combination of the following must be given:

        - ``train``, ``test``, ``val``
        - ``train_val``, ``test``
        - ``data``

        :param test: test dataset
        :param val: validation dataset
        :param train: training dataset
        :param train_val: dataset to split into training and validation dataset
        :param data: dataset to split into training, validation, and test data
        :param validator: callable that raises if given dataset is
            invalid with informative error messages including the given
            context string;
            defaults to identity
        :param split_kwargs: arguments to :py:meth:`split_train_val` and
            :py:meth:`split_trainval_test` if not all splits are explicitly
            given; see there for defaults
        :raises: :py:exc:`ValueError` if the data specification is
            insufficient or ambiguous,
            or if the datasets do not pass the validity check
        """
        triplet: _IncompleteDataTriple = \
            self._complete_init_input(
                train=train, val=val, test=test, train_val=train_val, data=data,
                validator=validator, **split_kwargs)

        self._data: Dataset = triplet.data
        """Internal storage of :py:attr:`data`."""
        self._train: Dataset = triplet.train
        """Internal storage of :py:attr:`train`."""
        self._val: Dataset = triplet.val
        """Internal storage of :py:attr:`val`."""
        self._train_val: Dataset = triplet.train_val
        """Internal storage of :py:attr:`train_val`."""
        self._test: Dataset = triplet.test
        """Internal storage of :py:attr:`test`."""

    def _complete_init_input(
            self, *,
            train: Dataset = None, val: Dataset = None,
            test: Dataset = None, train_val: Dataset = None,
            data: Dataset = None,
            validator: Callable[[Dataset, str], Any] = None,
            **split_kwargs
    ) -> _IncompleteDataTriple:
        """Validate and complete the input arguments to ``__init__`` by
        splitting or concatenation."""

        self._validate_init_input(data=data, test=test, train=train,
                                  train_val=train_val, val=val,
                                  validator=validator)

        # region Complete datasets arguments by splitting
        if data is not None:
            train_val, test = self.split_trainval_test(data, **split_kwargs)
        if train_val is not None:
            train, val = self.split_train_val(train_val, **split_kwargs)
        else:
            train_val: Dataset = torch.utils.data.ConcatDataset([train, val])
        if data is None:
            data: Dataset = torch.utils.data.ConcatDataset([train_val, test])
        # endregion

        return _IncompleteDataTriple(data=data, train=train, val=val,
                                     test=test, train_val=train_val)

    @staticmethod
    def _validate_init_input(*,
                             data: Dataset = None,
                             test: Dataset = None,
                             train: Dataset = None,
                             train_val: Dataset = None,
                             val: Dataset = None,
                             validator: Callable[[Dataset, str],
                                                 Any] = None) -> None:
        """Parse the given argument combination to a data tuple for
        :py:meth`_tuple_to_datasets` data preparation.
        For details on the arguments see ``__init__``.

        :raises: :py:exc:`ValueError` if the specification is ambiguous or
            insufficient; errors according to the check in ``validator``
            for all given datasets
        """
        # region Validity of given datasets
        validator = validator \
            if validator is not None else (lambda d, s: None)
        any(d is not None and validator(d, desc) for desc, d in
            {'train': train, 'val': val, 'test': test,
             'train_val': train_val, 'data': data}.items())
        # endregion

        # region ambiguity or incompleteness of given dataset tuple
        if data is not None:
            if any(d is not None for d in [train, val, test]):
                raise ValueError("Ambiguous spec: data was given but also a "
                                 "combination of other dataset specifiers.")
            if len(data) < 3:
                raise ValueError("data too small for splitting (len {})"
                                 .format(len(data)))
        if train_val is not None:
            if any(d is not None for d in [train, val]):
                raise ValueError("Ambiguous spec: train_val was given but also "
                                 "train or val data was specified.")
            if test is None:
                raise ValueError("Insufficient spec: test not specified "
                                 "(test, data, data_tuple were all None).")
            if len(train_val) < 2:
                raise ValueError("train_val too small for splitting (len {})"
                                 .format(len(train_val)))
        elif any(d is None or validator(d, desc) for desc, d in
                 {'train': train, 'val': val, 'test': test}.items()):
            raise ValueError("Insufficient spec: all of the following combis "
                             "contained None values: data_tuple, data, "
                             "(train_val, test), (train, test, val)")
        # endregion

    @staticmethod
    def _randomly_split_dataset(dataset: Dataset,
                                split1: float = None,
                                len1: int = None) -> Tuple[Subset, Subset]:
        r"""Randomly split the given dataset exhaustively into two subsets with
        specified lengths. Length of the first split will be the floor of
        :math:`split_1 \cdot len(dataset)`, but at minimum 1.

        :param dataset: data to split
        :param split1: proportion of the data samples in first split within
            dataset;
            either ``split1`` or ``len1`` must be given; overridden by ``len1``
        :param len1: length of the desired second split;
            either ``split1`` or ``len1`` must be given; overrides by ``split1``
        :return: tuple of two random and exhaustive splits of dataset with
            according lengths
        :raises: :py:exc:`ValueError` if the dataset len is < 2 or the ``len1``
            is not in the necessary range;
            :py:exc:`AssertionError` if the splitting fails.
        """
        # Value check:
        if len(dataset) < 2:
            raise ValueError("Dataset to split must have len > 1, but was {}"
                             .format(len(dataset)))
        if len1 is not None and not 1 <= len1 < len(dataset):
            raise ValueError(("Given length of second split {} is not in "
                              "required range [1, {}-1]")
                             .format(len1, len(dataset)))

        # Dataset lengths according to val_split:
        len1: int = len1 or max(1, int(split1 * len(dataset)))
        len2: int = len(dataset) - len1

        # Dataset splitting with success check:
        data_split1, data_split2 = \
            torch.utils.data.random_split(dataset, lengths=[len1, len2],
                                          generator=None)
        assert len(data_split1) == len1 and len(data_split2) == len2, \
            (("Dataset splitting failed: expected lengths ({}, {}), "
              "but got ({}, {})").format(len1, len2, len(data_split1),
                                         len(data_split2)))

        return data_split1, data_split2

    @staticmethod
    def split_dataset(dataset,
                      indices1: Sequence[int] = None,
                      indices2: Sequence[int] = None,
                      len1: int = None, split1: float = None
                      ) -> Tuple[Subset, Subset]:
        """Split dataset exhaustively into two subsets, either randomly or
        according to indices.
        Yields the resulting splits without changing dataset.
        For random splitting, the length ``len1`` or split proportion
        ``split1`` of the first split are used.
        For splitting by indices, the indices are validated
        (may take some time ...).

        Parameter constraints:

        - At least one of the optional splitting specifiers must be given.
        - Only true splits of dataset are allowed, i.e. indices if given must
          not occur twice!
        - Precedence of given specifiers is as follows (strongest to weakest):

          - indices
          - len
          - split

        :param dataset: the dataset to split
        :meta public:
        :param indices1: Optional indices of the first data split;
            must be disjoint to ``indices2`` and contain no duplicates;
            defaults to a random set of indices or those not in indices2 if
            that is given
        :param indices2: see ``indices1``
        :param len1: length of the desired first data split
        :param split1: proportion of the data samples in second data split of
            all dataset samples
        """
        # Value checks:
        # Any splitting specifier given?
        if all(i is None for i in [indices1, indices2, len1, split1]):
            raise ValueError("Any of the splitting specifiers must be given, "
                             "but all were none.")

        total_len: int = len(dataset)
        # Are all given indices in range and unique?
        for split, idxs in ((n, i) for n, i in enumerate((indices1, indices2))
                            if i is not None):
            seen = set()  # elements encountered already
            for i in (i for i in idxs if
                      i >= total_len or  # in range?
                      (i in seen and seen.add(i) is None)  # not duplicate?
                      ):
                raise IndexError(("Index {} in split {} out of range of len {} "
                                  "of dataset").format(i, split, total_len))

        # Do the indices not share an index?
        if indices1 is not None and indices2 is not None:
            for i in (i for i in indices1 if i in indices2):
                raise ValueError(("train_indices and val_indices share the "
                                  "index {}").format(i))

        # Actual splitting:
        # Either split by given indices ...
        if indices1 is not None or indices2 is not None:
            if indices1 is None:
                indices1 = [i for i in range(total_len) if i not in indices2]
            if indices2 is None:
                indices2 = [i for i in range(total_len) if i not in indices1]

            data_split1 = torch.utils.data.Subset(dataset, indices=indices1)
            data_split2 = torch.utils.data.Subset(dataset, indices=indices2)
        # ... or split randomly
        else:
            data_split1, data_split2 = DataTriple._randomly_split_dataset(
                dataset, len1=len1, split1=split1)
        return data_split1, data_split2

    # noinspection PyUnusedLocal
    @classmethod
    def split_train_val(cls,
                        train_val_data: Dataset,
                        train_indices: Sequence[int] = None,
                        val_indices: Sequence[int] = None,
                        val_len: int = None, val_split: float = None,
                        **ignored_args  # pylint: disable=unused-argument
                        ) -> Tuple[Subset, Subset]:
        r"""Split ``train_val_data`` either randomly or according to indices
        and return splits.
        This is a wrapper around :py:meth:`split_dataset` with nicer parameter
        naming, order correction, and defaults. The same parameter constraints
        apply.

        :param train_val_data: the dataset to split
        :param train_indices: Optional indices of the training part of
            the data set
        :param val_indices: Optional indices of the validation data set
        :param val_len: length of the desired validation set split
        :param val_split: proportion of validation data samples in the total
            dataset;
            defaults to :py:attr:`val_split`
        :return: tuple of splits (``train``, ``val``)
        """
        val_split: Optional[float] = val_split or cls.DEFAULT_VAL_SPLIT
        val_data, train_data = cls.split_dataset(train_val_data,
                                                 indices1=val_indices,
                                                 indices2=train_indices,
                                                 len1=val_len,
                                                 split1=val_split)
        return train_data, val_data

    # noinspection PyUnusedLocal
    @classmethod
    def split_trainval_test(cls,
                            data: Dataset,
                            train_val_indices: Sequence[int] = None,
                            test_indices: Sequence[int] = None,
                            test_len: int = None, test_split: float = None,
                            **ignored_args  # pylint: disable=unused-argument
                            ) -> Tuple[Subset, Subset]:
        r"""Split ``data`` either randomly or according to indices
        and return splits.
        This is a wrapper around :py:meth:`split_dataset` with nicer parameter
        naming, order correction, and defaults. The same parameter constraints
        apply.

        :param data: the dataset to split
        :param train_val_indices: Optional indices of the training part of
            the data set
        :param test_indices: Optional indices of the validation data set
        :param test_len: length of the desired validation set split
        :param test_split: proportion of validation data samples in the total
            dataset;
            defaults to :py:attr:`test_split`
        :return: tuple of splits (``train_val``, ``test``)
        """
        test_split: Optional[float] = test_split or cls.DEFAULT_VAL_SPLIT
        val_data, train_data = cls.split_dataset(data,
                                                 indices1=test_indices,
                                                 indices2=train_val_indices,
                                                 len1=test_len,
                                                 split1=test_split)
        return train_data, val_data

    def validate_by(self, validator: Callable[[Dataset, str], Any]) -> None:
        """Validate all data splits using validator, which raises in case of
        invalid format."""
        for desc, data_split in self.items():
            validator(data_split, desc.value)

    @property
    def info(self) -> pd.DataFrame:
        """Provide a string with some statistics on the held datasets."""
        # Some size information
        info: pd.DataFrame = pd.DataFrame(
            [{"split": split.value,
              "len": len(d),
              "type": d.__class__.__name__}
             for split, d in self.items()]
        ).set_index('split')
        return info

    def __repr__(self) -> str:
        """String representation of the held dataset splits."""
        return ("{cls}(\n"
                "    train={train},\n"
                "    val={val},\n"
                "    test={test}\n"
                ")").format(cls=self.__class__.__name__,
                            train=repr(self.train),
                            val=repr(self.val),
                            test=repr(self.test))

    def __eq__(self, other) -> bool:
        """Check that all data sub-sets are the same objects."""
        return (self.train is other.train) and \
               (self.test is other.test) and \
               (self.val is other.val) and \
               (self.train_val is other.train_val) and \
               (self.data is other.data)

    def as_dict(self) -> Dict[DatasetSplit, Dataset]:
        """Dict of the splits (train, val, test) held in this triple."""
        return {DatasetSplit.TRAIN: self.train,
                DatasetSplit.TEST: self.test,
                DatasetSplit.VAL: self.val,
                DatasetSplit.TRAIN_VAL: self.train_val}

    def items(self) -> ItemsView[DatasetSplit, Dataset]:
        """Dataset split items.
        Items of :py:meth:`as_dict` output."""
        return self.as_dict().items()

    def keys(self) -> KeysView[DatasetSplit]:
        """Dataset split keys.
         Keys of :py:meth:`as_dict` output."""
        return self.as_dict().keys()

    def __getitem__(self, key: DatasetSplit):
        """Get dataset split by split identifier."""
        return self.as_dict()[key]

    @classmethod
    def from_dict(cls, splits: Dict[DatasetSplit, Dataset]) -> 'DataTriple':
        """Create :py:class:`DataTriple` from a dict of datasets indexed by
        their split."""
        return cls(train=splits[DatasetSplit.TRAIN],
                   test=splits[DatasetSplit.TEST],
                   val=splits[DatasetSplit.VAL])


class BaseDataset(Dataset):
    """Abstract base class for tuple datasets with storage location.

    Derived datasets should yield tuples of ``(input, target)``.
    The transformation :py:attr:`transforms` is applied to data
    tuples before return from :py:meth:`__getitem__` can be controlled.
    The default for :py:attr:`transforms` is given by
    The default for :py:attr:`transforms` is given by
    :py:attr:`_default_transforms`. Override in sub-classes if necessary.
    The default combination of collected dataset tuples and
    :py:attr:`_default_transforms` should yield a tuple
    of :py:class:`torch.Tensor` or dicts thereof.

    The :py:attr:`hybrid_learning.datasets.base.BaseDataset.dataset_root` is
    assumed to provide information about the storage location.
    Best, all components (input data, annotations, etc.)
    should be stored relative to this root location.

    The transformed tuple values are cached by :py:attr:`transforms_cache`
    if it is given. Then values are only collected and transformed if they
    cannot be loaded from the cache. To get the cache descriptor for an entry
    the, the :py:meth:`descriptor` method is consulted. Make sure to override
    this appropriately (e.g. by image ID or image file name).

    .. note::
        In case a :py:class:`~hybrid_learning.datasets.caching.CacheTuple` is
        used, make sure that ``None`` is returned if any tuple value is
        ``None``.
    """

    def __init__(self, split: DatasetSplit = None, dataset_root: str = None,
                 transforms: Callable = None,
                 transforms_cache: caching.Cache = None,
                 after_cache_transforms: Optional[Callable] = None,
                 device: Optional[Union[torch.device, str]] = 'cpu'):
        """Init.

        :param split: The split of the dataset
            (e.g. :py:attr:`DatasetSplit.TRAIN`,
            :py:attr:`DatasetSplit.VAL`, :py:attr:`DatasetSplit.TEST`).
        :param dataset_root: The location where to store the dataset.
        :param transforms: The transformations to be applied to the data when
            loaded; defaults to :py:attr:`_default_transforms`
        :param transforms_cache: optional cache instance for caching transformed
            tuples; must return ``None`` in case one of the tuple values has not
            been cached yet; see :py:attr:`transforms_cache`
        :param after_cache_transforms: transformations applied after
            consulting the cache (no matter, whether the tuples was retrieved
            from cache or not); by default, tensor gradients are disabled
            and tensors are moved to a common device
        :param device: device to use in the default ``after_cache_transforms``
        """
        super().__init__()
        self.split: Optional[DatasetSplit] = split
        """Optional specification what use-case this dataset is meant to
        represent, e.g. training, validation, or testing."""
        self.dataset_root: str = dataset_root
        """Assuming the dataset is saved in some storage location, a root
        from which to navigate to the dataset information."""
        self.transforms: Callable = transforms or self._default_transforms
        """Transformation function applied to each item tuple before return.
        Applied in :py:meth:`__getitem__`.
        Default transformations are sub-class-specific.
        Items transformed using :py:attr:`transforms` can be cached by setting
        :py:attr:`transforms_cache`. If the transformations should be applied
        *always*, regardless of caching, use :py:attr:`after_cache_transforms`.
        """
        self.transforms_cache: Optional[caching.Cache] = transforms_cache
        """Cache for the transformed ``(input, target)`` tuples.
        If set, :py:meth:`__getitem__` will first try to load the tuple from
        cache before loading and transforming it normally.
        Items not in the cache are put in there after :py:attr:`transforms` is
        applied."""
        self.after_cache_transforms: Callable = \
            after_cache_transforms or \
            self._get_default_after_cache_trafo(device)
        """Transformation function applied after consulting the cache
        (no matter, whether the tuples was retrieved from cache or not).
        Use these transformations instead of :py:attr:`transforms` to ensure
        the transformation is *always* applied, regardless of caching.
        By default, tensor gradients are disabled and tensors are moved to a
        common device (see :py:meth:`_get_default_after_cache_trafo`)."""

        if dataset_root is not None:
            if not os.path.exists(dataset_root):
                raise FileNotFoundError("Dataset root {} does not exist"
                                        .format(dataset_root))
            if not os.path.isdir(dataset_root):
                raise NotADirectoryError("Dataset root is not a directory: {}"
                                         .format(dataset_root))

    @classmethod
    def _get_default_after_cache_trafo(
            cls, device: Optional[Union[str, torch.device]] = None):
        """The default transformation after cache depending on the chosen
        device.
        Used to get the default for :py:attr:`after_cache_transforms`.
        Transformation will remove requires_grad from tensors and move tensors
        to a common ``device`` (defaults to ``'cpu'``).
        """
        device = device or 'cpu'
        return (trafos.OnBothSides(trafos.NoGrad())
                + trafos.OnBothSides(trafos.ToTensor(
                    device, sparse=False, dtype=torch.float,
                    requires_grad=False)))

    @abc.abstractmethod
    def __len__(self):
        """Number of data points in the dataset; to be implemented in
        subclasses."""
        raise NotImplementedError()

    @abc.abstractmethod
    def getitem(self, idx: int
                ) -> Tuple[Union[torch.Tensor, PIL.Image.Image],
                           Union[torch.Tensor, PIL.Image.Image,
                                 Dict[torch.Tensor, PIL.Image.Image]]]:
        """Get data item tuple from ``idx`` in this dataset.

        :param idx: index to retrieve data point from
        :return: tuple ``(input, label)`` with

            - ``input`` one of: image (as :py:class:`PIL.Image.Image`),
              Radar/Lidar point cloud
            - ``label`` one of:

              - None
              - class label (as :py:class:`torch.Tensor` or ``bool``),
              - semantic segmentation map (as :py:class:`PIL.Image.Image` or
                :py:class:`torch.Tensor` compatible with torchvision
                transforms),
              - bounding box
              - string-indexed dict of combinations
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def descriptor(self, i: int) -> Hashable:
        """Return a unique descriptor for the item at position ``i``.
        This can e.g. be an image ID or the image file name.
        It is used for caching.
        """
        raise NotImplementedError()

    @staticmethod
    def _tupled(item) -> Tuple:
        """Wrap item in a tuple if is not already is one."""
        return item if isinstance(item, tuple) else (item,)

    def __getitem__(self, idx: int):
        """Get item from ``idx`` in dataset with transformations applied.
        Transformations must be stored as single tuple transformation in
        :py:attr:`transforms`.

        :return: tuple output of :py:meth:`getitem` transformed by
            :py:attr:`transforms`
        """
        inp_target, desc = None, None
        put_to_cache = load_from_cache = self.transforms_cache is not None
        if load_from_cache:
            desc: Hashable = self.descriptor(idx)
            inp_target = self.transforms_cache.load(desc)
            put_to_cache = inp_target is None
        if inp_target is None:
            inp_target = self.getitem(idx)
            if self.transforms is not None:
                inp_target = self.transforms(*self._tupled(inp_target))
        if put_to_cache:
            self.transforms_cache.put(desc, inp_target)
        if self.after_cache_transforms is not None:
            inp_target = self.after_cache_transforms(*self._tupled(inp_target))
        return inp_target

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings of the instance.
         :py:attr:`transforms` info is skipped if set to default."""
        trafo_info = {'transforms': self.transforms} \
            if self.transforms != self._default_transforms \
            else {}  # pylint: disable=comparison-with-callable
        split_info = {'split': self.split} if self.split is not None else {}
        cache_info = ({'transforms_cache': self.transforms_cache}
                      if self.transforms_cache is not None else {})
        return dict(dataset_root=self.dataset_root,
                    **split_info,
                    **trafo_info,
                    after_cache_transforms=self.after_cache_transforms,
                    **cache_info)

    def __repr__(self) -> str:
        """Nice printing function."""
        other_info: str = ', '.join(['{}={}'.format(k, repr(i))
                                     for k, i in self.settings.items()])
        return '{cls}(len={len}, {other})'.format(cls=self.__class__.__name__,
                                                  len=self.__len__(),
                                                  other=other_info)

    # The default function does not use self, but overriding functions may
    _default_transforms: Optional[TupleTransforms] = None
    """The default transformation to apply if no other is given.
    Set to ``None`` to apply no transforms by default.

    :meta public:
    """


# DATASET INDEX MANIPULATION
# --------------------------
def cross_validation_splits(train_val_data, num_splits: int
                            ) -> List[Tuple[Subset, Subset]]:
    """Split dataset it into ``num_splits`` and collect tuples of
    ``(rest, split)``.
    This is useful for creating splits for cross-validation, where ``rest``
    would be the training data, and ``split`` would be the validation data
    split.
    """
    if num_splits <= 1:
        raise ValueError("num_splits must be integer > 1 but was {}"
                         .format(num_splits))

    # lengths of the validation splits:
    lengths = [len(train_val_data) // num_splits] * num_splits
    for i in range(len(train_val_data) % num_splits):
        lengths[i] += 1

    # list of index lists for the validation splits
    # noinspection PyTypeChecker
    val_split_idxs = [list(d) for d in
                      torch.utils.data.random_split(
                          np.arange(len(train_val_data)), lengths,
                          generator=None)]

    # Collect the train and validation subset for each split
    splits = []
    for val_id, val in enumerate(val_split_idxs):
        train_data = Subset(train_val_data,
                            [i for v_id, idxs in enumerate(val_split_idxs)
                             for i in idxs if v_id != val_id])
        val_data = Subset(train_val_data, val)
        splits.append((train_data, val_data))

    return splits


def add_gaussian_peak(mask_np: np.ndarray,
                      centroid: Tuple[float, float],
                      binary_radius: Union[float, int],
                      radius_value: float = 0.5,
                      ) -> np.ndarray:
    r"""Add a peak to the heatmap ``mask_np`` as a non-normalized gaussian
    at ``centroid``. The standard deviation is calculated such that the
    value of the gaussian is ``radius_value`` at L2 distance of``binary_radius``
    from the centroid.
    The gaussian is non-normalized, i.e. has maximum 1.
    For adding, each pixel is set to the maximum of the original value or the
    new gaussian.
    The value of a pixel is set to that of its center point, e.g. the pixel
    at index ``[0,0]`` in the image gets the value of the point ``(0.5, 0.5)``.

    :param mask_np: numpy mask of shape ``(width, height)``
    :param centroid: the center point of the peak in ``(x, y)`` in pixels
    :param binary_radius: the radius of the peak if binarized with a threshold
        of ``radius_value``; is related to the standard deviation of the non-normalized
        gaussian via :math:`\sigma=\frac{r}{\sqrt{-2 \ln(\text{radius\_value})}}`
    :param radius_value: the value the point at L2 distance of ``binary_radius``
        from the peak ``centroid`` should have
    :return: copy of ``mask_np`` with new peak added
    """
    assert len(mask_np.shape) == 2
    if not (0 < radius_value < 1):
        raise ValueError("radius_value must be positive float in (0, 1), but was {}".format(radius_value))
    width, height = mask_np.shape
    # Pixel centers
    x_coords: np.ndarray = np.stack([np.arange(width)] * height, axis=1) + 0.5
    y_coords: np.ndarray = np.stack([np.arange(height)] * width, axis=0) + 0.5
    # Distance of pixel centers to centroid
    square_dists = (x_coords - centroid[1]) ** 2 + (y_coords - centroid[0]) ** 2
    std_dev: float = binary_radius / np.sqrt(2 * np.log(1 / radius_value))
    # Non-normalized gaussian:
    gaussian = np.exp(- square_dists / (2 * (std_dev ** 2)))
    mask_np = np.max(np.stack([gaussian, mask_np], axis=0), axis=0)
    return mask_np

# HASHING METHODS
# ---------------
# import hashlib
# def model_parameter_hash(model: torch.nn.Module,
#                          hash_len: Optional[int] = None) -> str:
#     """Produce a truncated hex hash from the state of a pytorch model
#     (and only from that).
#     Guarantee: Two models whose parameters origin from the same file
#     yield the same hash.
#
#     .. note::
#       This can take extremely long (e.g. 60s for Mask R-CNN).
#
#     The hard part is that the hash should really only depend on the model
#     parameter data (and possibly architecture), so
#
#     - no standard hash() can be used: this is session dependent
#     - no hash of pickled models can be used: these save positions in memory
#       which change;
#       Instead, the parameters are parsed to a string representation to be
#       then hashed.
#
#     :param hash_len: Length to which to truncate the hash; no truncation if
#         ``None``
#     :param model: torch model to hash the state of.
#     """
#     if hash_len is not None and hash_len <= 0:
#         raise ValueError("Hash length must >0 but was {}".format(hash_len))
#
#     state_dict: Dict[str, torch.Tensor] = model.state_dict()
#
#     # Apply the magic:
#     # Represent the complete state_dict ({str: tensors}) as encoded
#     # string to make it hashable.
#     # Mind the sorting to make this deterministic.
#     hashable_state_dict: bytes = '\n'.join((
#         "{!s}\t{!s}".format(k, v.tolist())
#         for k, v in sorted(state_dict.items())
#     )).encode()
#
#     # Now create the hex hash.
#     state_dict_hash: int = hashlib.md5(hashable_state_dict).hexdigest()
#
#     # Truncate if necessary
#     format_str = "0x{!s" + ("}" if hash_len is None else
#                             ":0<" + str(hash_len) + "." + str(hash_len) + "}")
#     return format_str.format(state_dict_hash)
