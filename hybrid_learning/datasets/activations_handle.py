"""Wrapper dealing with generating and caching activation maps from a dataset.
Useful for concept embedding analysis for easy reuse of generated activation maps."""

#  Copyright (c) 2022 Continental Automotive GmbH

import os
from typing import Callable, Tuple, Optional, Any, Union, Dict, Sequence

import PIL.Image
import torch
from torch.utils.data import Subset
from tqdm import tqdm

from . import transforms, caching
from .base import BaseDataset


class DatasetWrapper(BaseDataset):
    """Wrap a dataset or subset to add caches or transforms.
    Wrapping a dataset with a :py:class:`DatasetWrapper` has the same effect
    as the in-place version

    .. code:: python

        dataset.transforms_cache = new_cache + cache
        dataset.transforms = transforms + new_transforms

    but without changing the wrapped dataset.
    In case the data is a :py:class:`torch.utils.data.Subset` instance,
    care is taken to correctly retrieve the descriptor of the subsetted
    data (without mixing up indices), see :py:meth:`descriptor`.

    .. note::
        :py:attr:`dataset` is read-only, and the dataset should not be
        changed. Create a new wrapper if this is necessary.
    """

    @property
    def dataset(self) -> Union[Subset, BaseDataset, Sequence]:
        """The wrapped dataset."""
        return self._dataset

    def __init__(self,
                 dataset: Union[Subset, BaseDataset, Sequence],
                 **data_args
                 ):
        """Init.

        :param dataset: Dataset to wrap; must be a sequence of tuples of
            ``(image, ground_truth)`` with both image and ground-truth
            of type :py:class:`torch.Tensor`;
        """
        # region Default values and value checks
        data_args['split'] = \
            data_args.get('split',
                          # dataset.split
                          getattr(dataset, "split", None) or
                          # dataset.dataset.split
                          getattr(getattr(dataset, "dataset", None),
                                  "split", None))
        data_args['dataset_root'] = \
            data_args.get('dataset_root',
                          # dataset.dataset_root
                          getattr(dataset, "dataset_root", None) or
                          # dataset.dataset.dataset_root
                          getattr(getattr(dataset, "dataset", None),
                                  "dataset_root", None))
        if data_args['dataset_root'] is None:
            raise ValueError(("dataset_root is None: not given and not "
                              "specified by dataset {}").format(repr(dataset)))
        # endregion
        super().__init__(**data_args)

        self._dataset: Union[Subset, BaseDataset, Sequence] = dataset
        """The wrapped dataset."""

    def __len__(self) -> int:
        """Length determined by the length of the wrapped dataset.
        See :py:attr:`dataset`."""
        return len(self.dataset)

    def getitem(self, idx: int
                ) -> Tuple[Union[torch.Tensor, PIL.Image.Image],
                           Union[torch.Tensor, PIL.Image.Image,
                                 Dict[torch.Tensor, PIL.Image.Image]]]:
        """Wrap the __getitem__ of the wrapped dataset."""
        return self.dataset[idx]

    def descriptor(self, i: int) -> str:
        """Wrap descriptor method of wrapped dataset.
        It is assumed that either :py:attr:`dataset` or
        ``self.dataset.dataset`` provide a function
        ``descriptor(i: int)``. This is e.g. the case for a
        :py:class:`hybrid_learning.datasets.base.BaseDataset` or a
        :py:class:`torch.utils.data.Subset` instance.
        In case of a subset instance, care is taken to heed the index shuffling.
        """
        desc = None
        # For standard BaseDataset -> use dataset descriptor
        if hasattr(self.dataset, "descriptor") \
                and callable(self.dataset.descriptor):
            # noinspection PyTypeChecker
            desc = self.dataset.descriptor
        # Dataset wrapped in Subset -> take care of index permutation
        if isinstance(self.dataset, Subset) \
                and hasattr(self.dataset.dataset, "descriptor") \
                and callable(self.dataset.dataset.descriptor):
            desc = lambda idx: self.dataset.dataset.descriptor(
                self.dataset.indices[idx])
        if desc is None:
            raise AttributeError(
                ("Could not find descriptor() method within self.dataset "
                 "nor its dataset member; dataset type: {}"
                 ).format(type(self.dataset)))
        return desc(i)


class ActivationDatasetWrapper(DatasetWrapper):
    # noinspection PyUnresolvedReferences
    """Wrapper for image datasets that will generate and yield activation maps.
    Behaves like a sequence of tuples of

    - activation maps (original input transformed by
        :py:attr:`generate_act_map`) and
    - original ground truth (e.g. mask)

    The wrapper can handle :py:class:`torch.utils.data.Subset` and
    :py:class:`hybrid_learning.datasets.base.BaseDataset` instances.

    Features:

    - Option to enable efficient file caching of the generated activation
      maps in order to avoid costly re-evaluations.
      See :py:attr:`act_maps_cache`.
    - Convenience functions for caching:
      :py:meth:`existence checks <act_map_exists>` and
      :py:meth:`cache filling <fill_cache>` with progress bar.
    - Replacement of the :py:attr:`activation generator <generate_act_map>` by
      the cache, i.e. no ``act_map_gen`` must be provided if activation maps
      for all indices are cached (make sure to not clear the cache then though).

    Activation map caching is enabled by setting :py:attr:`activations_root`
    and disabled by setting :py:attr:`activations_root` to ``None``.
    To fill the cache i.e. generate all activation maps, call
    :py:meth:`fill_cache`. But be aware that this can be very time
    consuming depending on the generator.
    """

    @property
    def activations_root(self) -> Optional[str]:
        """The activations root of the file cache if caching is enable.
        Enable caching by setting this to a file path, disable caching
        by setting this attribute to ``None``."""
        if not self.act_maps_cache:
            return None
        return self.act_maps_cache.cache_root

    @activations_root.setter
    def activations_root(self, activations_root: Optional[str]):
        """Disable activation map file caching."""
        self.act_maps_cache: caching.PTCache = \
            caching.PTCache(activations_root) if activations_root else None

    def __init__(self, dataset: BaseDataset,
                 act_map_gen: torch.nn.Module = None,
                 activations_root: str = None,
                 device: Union[str, torch.device] = None,
                 **data_args):
        # pylint: disable=line-too-long
        """Init.

        The base settings (dataset root, split) default to those of the
        wrapped dataset.

        :param dataset: Dataset to wrap; must be a sequence of tuples of
            ``(image, ground_truth)`` with both image and ground-truth
            of type :py:class:`torch.Tensor`;
            the default transformation assumes that the ground truth are masks
            (same sized images)
        :param act_map_gen: torch module that accepts as input a batch of
            images and returns the activation maps to yield
        :param activations_root: root directory under which to store and find
            the activation maps if file caching shall be enabled
        :param device: the device on which to run ``act_map_gen``;
            see :py:class:`hybrid_learning.datasets.transforms.image_transforms.ToActMap`
        """
        # pylint: enable=line-too-long
        super().__init__(dataset, **data_args, device=device)

        self.act_maps_cache: caching.PTCache = \
            caching.PTCache(activations_root) if activations_root else None
        """File cache for caching activations.
        Set to ``None`` in case the activations root is set to ``False``
        during init."""

        self.generate_act_map: Callable[[torch.Tensor], torch.Tensor] = \
            (transforms.ToActMap(act_map_gen, device=device)
             if act_map_gen is not None else None)
        """Transformation that returns an activation map given a valid input
        datum.
        Input data is assumed to origin from the original
        :py:attr:`~DatasetWrapper.dataset`.
        Used to generate missing activation maps in :py:meth:`getitem`."""

    def getitem(self, i: int) -> Tuple[torch.Tensor, Any]:
        """Get activation map and original ground truth for item at index ``i``.

        Used for
        :py:meth:`~hybrid_learning.datasets.base.BaseDataset.__getitem__`.
        If the activation map does not exist and a generator is given in
        :py:attr:`generate_act_map`, generate and save the activation map.

        :return: tuple of the loaded or generated activation map and the
            target of the original dataset for that act map
        """
        # Get mask
        img_t, mask_t = self.dataset[i]

        if self.act_maps_cache:
            desc: str = self.descriptor(i)
            act_map = self.act_maps_cache.load(desc)

            # Get activation map (generate and save lazily if possible)
            if act_map is None:
                if self.generate_act_map is None:
                    raise RuntimeError(
                        ("generate_act_map unset but act map at index {} "
                         "missing from root directory {} (assumed path {})."
                         ).format(i, self.act_maps_cache.cache_root,
                                  self.act_map_filepath(i)))
                act_map = self.generate_act_map(img_t)
                self.act_maps_cache.put(desc, act_map)
        else:
            if self.generate_act_map is None:
                raise RuntimeError("generate_act_map unset but called")
            act_map = self.generate_act_map(img_t)

        return act_map, mask_t

    def act_map_filepath(self, i: int) -> str:
        """Return the path to the activation map file in the cache.
        The base directory is :py:attr:`activations_root`.
        The basename is determined by the :py:attr:`act_maps_cache` from
        the ``descriptor()`` for the index ``i``.

        :param i: index of the image to get activation map for.
        :return: (relative or absolute) path to the activation map for datum
            at index ``i``
        """
        if not self.act_maps_cache:
            raise RuntimeError("act_map_filepath called but act_maps_cache is "
                               "unset")
        return self.act_maps_cache.descriptor_to_fp(self.descriptor(i))

    def act_map_exists(self, i: int) -> bool:
        """Check whether the activation map at index ``i`` is cached.

        :param i: index in :py:attr:`~DatasetWrapper.dataset` for which to check
            whether an activation map was created.
        """
        if not self.act_maps_cache:
            return False
        act_fp: str = self.act_map_filepath(i)
        return os.path.exists(act_fp) and os.path.isfile(act_fp)

    def load_image(self, i: int) -> torch.Tensor:
        """Load the image/original input for index ``i``."""
        return self.dataset[i][0]

    def fill_cache(self, force_rebuild: bool = False,
                   show_progress_bar: bool = True,
                   **kwargs) -> 'ActivationDatasetWrapper':
        """Generate activation maps for all images.

        :param force_rebuild: whether to overwrite existing images or not
        :param show_progress_bar: whether to show the progress using
            :py:class:`tqdm.tqdm`
        :param kwargs: further arguments to the progress bar
        """
        # region Value check
        if not self.act_maps_cache:
            raise ValueError("Act maps cannot be generated while "
                             "act_maps_cache is unset.")
        if not self.generate_act_map:
            raise ValueError("Act maps cannot be generated if "
                             "generate_act_map is unset.")
        # endregion

        act_maps_to_process = [i for i in range(len(self))
                               if force_rebuild or not self.act_map_exists(i)]
        if len(act_maps_to_process) == 0:
            return self

        if show_progress_bar:
            act_maps_to_process = tqdm(
                **{**dict(iterable=act_maps_to_process, unit="act_map",
                          desc="Activation maps newly generated: "),
                   **kwargs})
        for i in act_maps_to_process:
            # Get activation map
            desc: str = self.descriptor(i)
            img_t: torch.Tensor = self.load_image(i)
            act_map: torch.Tensor = self.generate_act_map(img_t)

            # Save
            self.act_maps_cache.put(desc, act_map)

        return self
