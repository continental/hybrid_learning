#  Copyright (c) 2022 Continental Automotive GmbH
"""Base classes and implementations of cache handles.
The common base class is :py:class:`Cache`.

Cache types provided::

- :py:class:`DictCache`: An in-memory cache
- :py:class:`FileCache`: Base class for file system caches
- Combined caches:

  - :py:class:`CacheCascade`: A chain map for caches with different sync modes
  - :py:class:`CacheTuple`: Have each entry of a tuple handled by a separate
    cache, but with shared descriptor key
"""

import abc
import collections
import logging
import os
import shutil
from typing import Hashable, Any, Optional, Dict, Iterable, List, Union, \
    Callable, Tuple, Sequence, Collection

import PIL.Image
import numpy as np
import torch
import torchvision.transforms.functional as tv_functional
from torch import multiprocessing

from .transforms.image_transforms import ToTensor


class Cache(abc.ABC):
    """Caching base handle.
    Put objects into the cache using :py:meth:`put`,
    and load cached objects by their cache descriptor using :py:meth:`load`.
    Derive custom caching handles from this class.
    """

    @abc.abstractmethod
    def put(self, descriptor: Hashable, obj: Any):
        """Store ``obj`` in this cache.
        In case it already exists, the existing object is overwritten.

        :param descriptor: the descriptor key under which to store the
            object; used to access the object later
        :param obj: the object to put into cache; must not be ``None``
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, descriptor: Hashable) -> Optional[Any]:
        """Load the object stored under key ``descriptor`` from cache.
        ``None`` is returned if the object is not in the cache.
        """
        raise NotImplementedError()

    def put_batch(self, descriptors: Iterable[Hashable], objs: Iterable[Any]):
        """Store a batch of ``objs`` in this cache using according ``descriptors``."""
        if not isinstance(objs, collections.Iterable) or not isinstance(descriptors, collections.Iterable):
            raise ValueError(("objs and descriptors must be iterable and of same length,"
                             "but were of type (objs) {} and (descriptors) {}!"
                              ).format(type(objs), type(descriptors)))
        for desc, obj in zip(descriptors, objs):
            self.put(desc, obj)

    def load_batch(self, descriptors: Iterable[Hashable],
                   return_none_if: Union[str, int] = 'any') -> Optional[Collection]:
        """Load a batch of objects. Return ``None`` according to ``return_none_if``.

        :param descriptors: descriptors to load values for
        :param return_none_if: may be ``"any"``, ``"all"``, ``"never"``
        """
        return_none_if = self._standardize_return_none_if(return_none_if)
        return self._to_none([self.load(desc) for desc in descriptors], return_none_if)

    @staticmethod
    def _standardize_return_none_if(return_none_if: Union[str, int]) -> int:
        """Return an int representation of the ``return_none_if`` value.
        Raise if ``return_none_if`` has invalid value."""
        return_none_if_vals: Dict[str, int] = {'any': 1, 'all': 0, 'never': -1}
        if return_none_if not in (*return_none_if_vals.keys(),
                                  *return_none_if_vals.values()):
            raise ValueError(("Expected as value of return_none_if a key or "
                              "value of {}, but was {}."
                              ).format(return_none_if_vals, return_none_if))
        return return_none_if_vals.get(return_none_if, return_none_if)

    @staticmethod
    def _to_none(objs: Collection, return_none_if: int) -> Optional[Collection]:
        """Return ``None`` if the condition encoded by ``return_none_if`` holds,
        else return ``objs`` unchanged."""
        if objs is None:
            return objs
        elif return_none_if > 0 and any(obj is None for obj in objs):
            objs = None
        elif return_none_if == 0 and all(obj is None for obj in objs):
            objs = None
        return objs

    @abc.abstractmethod
    def clear(self):
        """Clear the current cache."""
        raise NotImplementedError()

    @abc.abstractmethod
    def descriptors(self) -> Iterable:
        """Return all descriptors for which an element is cached."""
        raise NotImplementedError()

    def as_dict(self) -> Dict:
        """Return a dict with all cached descriptors and objects.
        Beware: This can be very large!"""
        return {k: self.load(k) for k in self.descriptors()}

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"

    def wrap(self, getitem: Callable[[int], Any],
             descriptor_map: Callable[[int], Hashable] = None
             ) -> Callable[[int], Any]:
        """Add this cache to the deterministic function ``getitem`` (which
        should have no side effects).
        When the wrapped method is called, the desired item from cache is
        returned. Only if it does not exist in the cache, the original
        ``getitem`` is called, and its output cached and returned.
        The optional ``descriptor_map`` function should map a ``getitem``-input
        to the hash value which is to be used for the cache.
        E.g. it could map an index to the underlying file name.

        ``getitem`` should

        - have no side effects, and
        - have deterministic output, i.e. calls with equal input value
          will return equal output values.

        ``descriptor_map`` should

        - accept elements from the same domain as ``getitem``,
        - be injective, i.e. map each ``getitem``-input to a *unique* descriptor
          value.

        :param getitem: the function to wrap; for requirements see above
        :param descriptor_map: optionally a map from ``getitem``-input to
            descriptor
        """

        if descriptor_map is None:
            descriptor_map: Callable[[int], int] = lambda i: i

        def cached_getitem(i: int) -> Any:
            """The cached ``getitem`` method."""
            desc: Hashable = descriptor_map(i)
            out = self.load(desc)
            if out is None:
                out = getitem(i)
                self.put(desc, out)
            return out

        cached_getitem.__name__ = getitem.__name__
        cached_getitem.__doc__ = getitem.__doc__
        return cached_getitem

    def __add__(self, other: Optional['Cache']) -> 'Cache':
        """Return a (cascaded) cache which will first lookup ``self`` then
        ``other`` with default sync mode.
        In case ``other`` is ``None`` or a dummy :py:class:`NoCache`,
        return ``self``.

        :return: one of the summands in case the other is a no-op, else
            a :py:class:`CacheCascade` transforms.
        """
        if not (isinstance(other, Cache) or other is None):
            return NotImplemented
        # no-op:
        if other is None or isinstance(other, NoCache):
            return self
        return CacheCascade(self, other)

    def __radd__(self, other: Optional['Cache']) -> 'Cache':
        """Return a (cascaded) cache which will first lookup ``other`` then
        ``self`` with default sync mode.
        See :py:meth:`__add__`."""
        # no-op:
        if other is None:
            return self
        if not isinstance(other, Cache):
            return NotImplemented
        return other.__add__(self)


# noinspection PyMissingOrEmptyDocstring
class NoCache(Cache):
    """Dummy cache that has no effect."""

    def put(self, descriptor: Hashable, obj: Any):
        pass

    def load(self, descriptor: Hashable) -> Optional[Any]:
        return None

    def descriptors(self) -> Iterable:
        return ()

    def clear(self):
        pass

    def __add__(self, other: Optional[Cache]) -> Cache:
        if other is None:
            return self
        if not isinstance(other, Cache):
            return NotImplemented
        return other

    def __radd__(self, other: Optional[Cache]) -> Cache:
        return self + other


class DictCache(Cache):
    # pylint: disable=line-too-long
    """Cache objects in a (multiprocessing capable) dictionary in memory.
    In case this cache is used, the multiprocessing sharing strategy is
    automatically set to ``'file_system'`` since otherwise the ulimit of
    multiprocessing is exceeded for larger cache sizes.
    See `pytorch issue 973 <https://github.com/pytorch/pytorch/issues/973>`_
    for this, and the `pytorch doc on multiprocessing <https://pytorch.org/docs/stable/multiprocessing.html#file-system-file-system>`_
    for the drawbacks of this sharing strategy.

    .. warning::
        Be sure to have enough RAM!
    """

    # pylint: enable=line-too-long

    def __init__(self, thread_safe: bool = True):
        """Init.

        :param thread_safe: whether to use a multiprocessing-capable dict
        """
        if thread_safe:
            # This is needed to ensure that the ulimit is not exceeded, see here
            # https://github.com/pytorch/pytorch/issues/973
            torch.multiprocessing.set_sharing_strategy('file_system')
            self.cache: Dict = multiprocessing.Manager().dict()
        else:
            self.cache: Dict = {}

    def put(self, descriptor: Hashable, obj: Any):
        """Store ``obj`` under key ``descriptor`` in a in-memory cache.
        In case it already exists, the existing object is overwritten.

        :param descriptor: the descriptor key under which to store the
            object; used to access the object later
        :param obj: the object to put into cache; must not be ``None``
        """
        if obj is None:
            raise ValueError("Cache received None object for descriptor {}"
                             .format(descriptor))
        self.cache[descriptor] = obj

    def load(self, descriptor: Hashable) -> Optional[Any]:
        """Load the object stored under ``descriptor`` from in-memory cache."""
        return self.cache.get(descriptor, None)

    def clear(self):
        """Empty cache dict."""
        self.cache: Dict = {} if isinstance(self.cache, dict) else \
            multiprocessing.Manager().dict()

    def descriptors(self) -> Iterable:
        """Return the keys (descriptors) of the cache dict."""
        return self.cache.keys()


class TensorDictCache(DictCache):
    """In-memory cache specifically for torch tensors.
    Other than a normal :py:class:`DictCache` it takes care to move
    a :py:class:`torch.Tensor` to CPU before saving it to the shared memory,
    since at the time being sharing of CUDA-tensors between sub-processes is
    not supported.

    .. note::
        Do not expect speed improvements if CUDA based tensors are to be
        cached: Copying tensors from and to CPU is quite costly and
        comparable if not less efficient than loading from file.
        Consider using a :py:class:`PTCache` in such cases.
    """

    def __init__(self, sparse: bool = False, thread_safe: bool = None):
        """Init.

        :param sparse: whether tensors should be sparsified before put
            (and de-sparsified afterwards)
        :param thread_safe: whether to use a multiprocessing-capable dict;
            only available if ``sparse`` is not activated
        """
        super().__init__(thread_safe=(not sparse) and
                                     (thread_safe is None or thread_safe))
        # It looks like sparse tensors are for now not pickleable:
        self._to_tens: ToTensor = ToTensor(sparse=sparse, device='cpu')

    def put(self, descriptor: Hashable, obj: torch.Tensor):
        """Store torch ``obj`` under key ``descriptor`` in a in-memory cache.
        In case it already exists, the existing object is overwritten.
        If the tensor device is CPU, the tensor is cached, else a CPU copy
        of it.

        :param descriptor: the descriptor key under which to store the
            object; used to access the object later
        :param obj: the tensor to put into cache; must not be ``None``
        """
        if obj is None:
            raise ValueError("Cache received None object for descriptor {}"
                             .format(descriptor))
        self.cache[descriptor] = self._to_tens(obj)

    def load(self, descriptor: Hashable) -> Optional[Any]:
        """Load and densify tensors from in-memory cache."""
        obj = super().load(descriptor)
        if obj is None:
            return obj
        if isinstance(obj, torch.Tensor) and obj.is_sparse:
            if obj.dtype == torch.bfloat16:
                obj = obj.to(torch.float)
            obj = obj.to_dense()
        return obj


class FileCache(Cache, abc.ABC):
    """Base class to cache objects as files under a cache folder.
    An implementation needs to set the :py:attr:`FILE_ENDING` and
    implement the object type specific :py:meth:`~FileCache.put_file` and
    :py:meth:`~FileCache.load_file` methods.
    Mind that writing to the files is not multiprocess save, so ensure no
    objects in cache are overwritten while other processes are reading
    from cache.

    The descriptors are used to create the filenames by appending the
    :py:attr:`FILE_ENDING`.
    """
    FILE_ENDING = None
    """The file ending to append to descriptors to get the file path.
    See :py:meth:`~FileCache.descriptor_to_fp`."""

    def __init__(self, cache_root: str = None):
        """Init.

        :param cache_root: see :py:attr:`~FileCache.cache_root`
        """
        self.cache_root = cache_root or ".cache"
        """The path to the root folder under which to store cached files."""
        os.makedirs(self.cache_root, exist_ok=True)

    def put(self, descriptor: str, obj: Any):
        """Store ``obj`` under the cache root using
        :py:func:`~FileCache.put_file`.
        The file name is ``descriptor`` + :py:attr:`FILE_ENDING`.

        .. warning::
            This put method is not multiprocessing capable!
            Already created/put files may be overwritten by parallel processes.
            Make sure, no two processes will attempt to put an object to the
            same descriptor (e.g. handled by
            :py:class:`torch.utils.data.DataLoader` for map-style datasets).

        :param descriptor: The (unique) file name to use without
            :py:attr:`FILE_ENDING`; may also be a file path relative to the
            :py:attr:`~FileCache.cache_root`
        :param obj: the object to save; must not be ``None``
        """
        if obj is None:
            raise ValueError("Cache received None object for descriptor {}"
                             .format(descriptor))
        filepath = self.descriptor_to_fp(descriptor)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.put_file(filepath, obj)

    def load(self, descriptor: str) -> Optional[torch.Tensor]:
        """Load object from file ``descriptor`` + :py:attr:`FILE_ENDING`
        under cache root.
        Return ``None`` if file is not in cache.

        :param descriptor: The (unique) file name to use without the
            :py:attr:`FILE_ENDING`; may also be a file path relative to
            the :py:attr:`cache_root`
        """
        filepath = self.descriptor_to_fp(descriptor)
        if not os.path.isfile(filepath):
            return None

        try:
            return self.load_file(filepath)
        except PermissionError:
            logging.getLogger().warning("Cannot access cache file %s",
                                        filepath)
        except EOFError:
            logging.getLogger().warning("Encountered empty cache file: %s",
                                        filepath)
        return None

    def clear(self):
        """Remove all files from cache root.

        .. warning::
            This also removes files which were not created by this cache handle.
        """
        if not os.path.exists(self.cache_root):
            return
        for filelike in os.listdir(self.cache_root):  # os.scandir?
            filepath = os.path.join(self.cache_root, filelike)
            if os.path.isfile(filepath) or os.path.islink(filepath):
                os.unlink(filepath)
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath)

    def descriptors(self) -> Iterable:
        """Provide paths of all cached files with ending stripped and
        relative to cache root.
        These can be used as descriptors for accessing the cached files via
        :py:meth:`load`. The paths are given as normed paths using
        :py:func:`os.path.normpath`."""
        return (
            os.path.normpath(
                os.path.join(
                    os.path.relpath(root, start=self.cache_root), fn
                ).rsplit(self.FILE_ENDING, maxsplit=1)[0]
            )
            for root, _, filenames in os.walk(self.cache_root)
            for fn in filenames
            if fn.endswith(self.FILE_ENDING)
        )

    def descriptor_to_fp(self, descriptor: str) -> str:
        """Return the file path of the cache file for a given ``descriptor``."""
        return os.path.join(self.cache_root, "{}{}".format(
            str(descriptor), self.FILE_ENDING))

    @abc.abstractmethod
    def put_file(self, filepath: str, obj: Any):
        """Save put ``obj`` under ``filepath``."""
        raise NotImplementedError()

    @abc.abstractmethod
    def load_file(self, filepath: str) -> Optional[Any]:
        """Load object from ``filepath``."""
        raise NotImplementedError()

    def __repr__(self) -> str:
        return "{cls}(cache_root={cache_root})".format(
            cls=self.__class__.__name__,
            cache_root=str(self.cache_root)
        )


class PTCache(FileCache):
    """File cache that uses torch saving and loading mechanism.
    All objects are moved to the given :py:attr:`device`
    during loading.
    For further details see super class.

    .. note::
        The file sizes may become quite large for larger tensors.
        Consider a file cache applying compression if saving/loading times
        or storage space get a problem.
    """

    FILE_ENDING = ".pt"
    """The file ending to append to descriptors to get the file path.
    See :py:meth:`~FileCache.descriptor_to_fp`.
    This is the standard for :py:func:`torch.save`."""

    def __init__(self, cache_root: str = None,
                 device: Union[str, torch.device] = None,
                 sparse: Optional[Union[bool, str]] = 'smallest',
                 dtype: Optional[torch.dtype] = None,
                 before_put: Callable[[Any], torch.Tensor] = None,
                 after_load: Callable[[Any], torch.Tensor] = None):
        """Init.

        :param cache_root: see :py:attr:`~FileCache.cache_root`
        :param device: see :py:attr:`device`
        :param sparse: sparse option of the default :py:attr:`before_put`
        :param dtype: dtype option of the default :py:attr:`before_put`
        :param before_put: see :py:attr:`before_put`;
            overrides ``sparse`` and ``dtype``
        :param after_load: see :py:attr:`after_load`
        """
        super().__init__(cache_root=cache_root)
        self.device: Union[str, torch.device] = device or 'cpu'
        """The device to load elements to.
        See :py:meth:`~FileCache.load_file` and
        :py:meth:`~FileCache.put_file`."""
        self.before_put: ToTensor = before_put or ToTensor(
            sparse=sparse, dtype=dtype)
        """The transformation to call to obtain a tensor with desired
        properties for saving."""
        self.after_load: ToTensor = after_load or ToTensor(
            sparse=False, dtype=torch.float, requires_grad=False)
        """The transformation to call on loaded tensors."""

    def __repr__(self):
        return f"{self.__class__.__name__}(" \
               f"cache_root={str(self.cache_root)}, " \
               f"device={repr(self.device)}, " \
               f"before_put={repr(self.before_put)}, " \
               f"after_load={repr(self.after_load)})"

    def put_file(self, filepath: str, obj: torch.Tensor):
        """Save ``obj`` to ``filepath`` using :py:func:`torch.save`.
        Move ``obj`` to :py:attr:`device` before saving."""
        torch.save(self.before_put(obj), filepath)

    def load_file(self, filepath: str) -> Optional[torch.Tensor]:
        """Load ``obj`` from ``filepath`` using :py:func:`torch.load`.
        Move them to :py:attr:`device` before return.
        (Note that the tensors may be sparse.)
        """
        try:
            obj = torch.load(filepath, map_location=self.device)
            return self.after_load(obj)
        except RuntimeError as err:
            logging.getLogger().warning(
                "Received the following RuntimeError for cached file %s:\n%s",
                filepath, repr(err))
            return None


class NPYCache(FileCache):
    """File cache that uses numpy saving and loading mechanism to cache
    :py:class:`torch.Tensor` objects.
    Cannot use sparse tensor representation for saving for now.
    For further details see super class.
    """

    FILE_ENDING = ".npy"
    """The file ending to append to descriptors to get the file path.
    See :py:meth:`FileCache.descriptor_to_fp`.
    This is the standard for :py:func:`numpy.save`."""

    def put_file(self, filepath: str, obj: torch.Tensor):
        """Save ``obj`` to ``filepath`` using :py:func:`numpy.save`."""
        if isinstance(obj, torch.Tensor):
            if obj.is_sparse:
                obj = obj.to_dense()
            obj = obj.cpu().numpy()
        np.save(filepath, obj)

    def load_file(self, filepath: str) -> torch.Tensor:
        """Load ``obj`` from ``filepath`` using :py:func:`numpy.load`."""
        obj = np.load(filepath, allow_pickle=True)
        if np.issubdtype(obj.dtype, np.number) or \
                np.issubdtype(obj.dtype, np.bool_):
            obj = torch.from_numpy(obj)
        return obj


class NPZCache(FileCache):
    """File cache that uses numpy compressed saving and loading mechanism to
    cache :py:class:`torch.Tensor` objects.
    For further details see super class.
    """

    FILE_ENDING = ".npz"
    """The file ending to append to descriptors to get the file path.
    See :py:meth:`FileCache.descriptor_to_fp`.
    This is the standard for :py:func:`numpy.savez_compressed`."""

    OBJ_KEY = 'obj'
    """The key within the compressed ZIP archive under which to store and
    find the object."""

    def put_file(self, filepath: str, obj: torch.Tensor):
        """Save ``obj`` to .npz archive at ``filepath`` using
        :py:func:`numpy.savez_compressed`."""
        if isinstance(obj, torch.Tensor):
            if obj.is_sparse:
                obj = obj.to_dense()
            obj = obj.cpu().numpy()
        np.savez_compressed(filepath, **{self.OBJ_KEY: obj})

    def load_file(self, filepath: str) -> torch.Tensor:
        """Load ``obj`` from .npz archive at ``filepath`` using
        :py:func:`numpy.load`."""
        obj = np.load(filepath, allow_pickle=True)[self.OBJ_KEY]
        if np.issubdtype(obj.dtype, np.number) or \
                np.issubdtype(obj.dtype, np.bool_):
            obj = torch.from_numpy(obj)
        return obj


class JPGCache(FileCache):
    """Cache for JPEG images using :py:mod:`PIL`.
    Non-image tensor or array objects are converted to images and must
    have the shape ``(height, width, channels)``.
    For mode auto-inference see :py:class:`torchvision.transforms.ToPILImage`.

    .. note::
        Will by default return tensors of type :py:class:`torch.float`
        with a value range in ``[0, 1]``.
        Set :py:attr:`after_load` accordingly, if a different type and value
        range is desired.

    .. warning::
        If :py:attr:`mode` is set to ``RGB``, make sure that the dtype of the
        tensors either is


    The pixel value ranges of the JPG images are always in ``[0, 255]``.
    For modes 'L' and 'RGB' with float dtype (not mode 'F' and not for int
    or bool types!), the value range of the tensor is automatically scaled
    from [0, 1] to [0, 255] during put.
    During load, modes 'L' and 'RGB' are automatically downscaled again from
    range [0, 255] to [0, 1].
    This means, the following pixel value ranges of the tensors (after the
    :py:attr:`before_put` transformation) are assumed:

    - mode 'F': [0, 255]
    - mode 'L', dtype int: [0, 255]
    - mode 'L', dtype float: [0, 1] -> automatically scaled to [0, 255]
    - mode 'RGB', dtype int: [0, 255]
    - mode 'RGB', dtype float: [0, 1] -> automatically scaled to [0, 255]

    Note that only integer values can be saved in JPG images, so values of
    mode 'F' get rounded to obtain valid mode 'L' values.
    """

    FILE_ENDING = ".jpg"
    """JPEG file ending. Appended to the descriptors to get the file path
    (may lead to a double ending, this is intentional)."""

    def __init__(self, cache_root: str, mode: Optional[str] = 'RGB',
                 before_put: Callable[[Any], torch.Tensor] = None,
                 after_load: Callable[[Any], torch.Tensor] = None):
        """Init.

        :param cache_root: the cache root directory
        :param mode: see :py:attr:`mode`
        """
        super().__init__(cache_root=cache_root)

        self.mode: str = mode
        """The :py:class:`PIL.Image.Image` mode to be represented by the put
        and loaded tensors (after the :py:attr`before_put` resp. before the
        :py:attr:`after_load` transformation).
        Note that the images are converted to mode ``'RGB'`` (for 3 to 4
        channels) or mode ``'L'`` (for 1 to 2 channels) before saving to JPG.
        Not all dtypes are supported for ``mode==None``."""
        self.before_put: Callable[[Any], torch.Tensor] = \
            before_put or ToTensor(sparse=False)
        """The transformation applied to tensors before turning them into
        images for putting. By default only densifies them."""
        self.after_load: Optional[Callable[[Any], torch.Tensor]] = \
            after_load or ToTensor()
        """The transformation applied to loaded tensors."""

    @staticmethod
    def save_image(img: PIL.Image.Image, filepath: str):
        """Save an image as JPG and take care of necessary conversion."""
        if img.mode in ['P', 'RGBA', 'HSV', 'PA']:
            img = img.convert('RGB')
        elif img.mode in ['I', 'I;16', 'I;16L', 'I;16B', 'I;16N', 'F', 'LA', ]:
            img = img.convert('L')
        img.save(filepath)

    def put_file(self, filepath: str, obj: torch.Tensor):
        """Convert ``obj`` to a py:mod:`PIL` image and save to ``filepath``."""
        obj: torch.Tensor = self.before_put(obj)

        # Auto-conversion of non-floating point images to uint8:
        # 1-channel: int8 and bool aren't supported yet
        # 2-channel, 3-channel: mode==None requires uint8
        # 3-channel RGB: expects uint8 with value range [0, 255]
        num_channels = obj.size()[0] if len(obj.size()) >= 3 else 1
        if not (obj.dtype.is_floating_point or obj.dtype == torch.bool):
            if self.mode == 'RGB' or obj.dtype == torch.int8 \
                    or (self.mode is None and num_channels >= 2):
                obj = obj.to(torch.uint8)
        if obj.dtype == torch.bool and num_channels < 3:
            obj = obj.to(torch.uint8) * 255

        obj: PIL.Image.Image = tv_functional.to_pil_image(obj, mode=self.mode)
        self.save_image(obj, filepath)

    def load_file(self, filepath: str) -> torch.Tensor:
        """Load image and convert to correct mode."""
        img: PIL.Image.Image = PIL.Image.open(filepath)
        if self.mode is not None and img.mode != self.mode:
            img = img.convert(self.mode)
        img: torch.Tensor = self.after_load(img)
        return img


class CacheCascade(Cache):
    """Combine several caches by trying to load from first to last.
    In case of a put, all caches are updated.
    In case of a load, the object is collected from the first cache holding it.
    In case
    and all previous ones are updated to also hold it.
    If :py:attr:`sync_by` is ``True``, then on load the first match is put
    to *all* other cache instances, not only the previous ones. I.e. the order
    of :py:attr:`caches` also determines the precedence.


    Some use-cases:

    - *Combine in-memory and persistent cache*:
      Combine a :py:class:`DictCache` with a :py:class:`FileCache` instance:
      Files are stored in file system for later runs, respectively loaded from
      previous runs, and additionally for even faster access stored in memory.
    - *Cache to different cache locations*:
      Combine several :py:class:`FileCache` caches with ``sync_by=True`` to
      write cache to several locations (can be used as sort of a lazy copy).
    """

    def __init__(self, *caches: Cache,
                 sync_by: Optional[Union[bool, str]] = 'precedence'):
        """Init.

        :param caches: cache instances in the correct load order
        :param sync_by: synchronization mode; must be one of ``'none'`` /
            ``False``, ``'precedence'``, ``'all'``;
            for details see :py:attr:`sync_by`
        """
        sync_by_vals = (False, 'none', 'precedence', 'all')
        if sync_by not in sync_by_vals:
            raise ValueError("sync_by must be one of {}, but was {}"
                             .format(sync_by_vals, sync_by))
        if len(caches) == 0:
            raise ValueError("No cache given")

        self.caches: List[Cache] = list(caches)
        """The list of caches to consult during load, ordered by descending
        precedence."""
        self.sync_by: bool = sync_by if sync_by != 'none' else False
        """Synchronization mode for loading. Update other instances according
        to the following settings:

        - ``False``: no sync; simply load from the first cache holding an
          object without updating the others
        - ``'precedence'``: when loading, update all caches with higher
          precedence (earlier in the :py:attr:`caches` list) that do not
          hold the object
        - ``'all'``: put object to all other caches when a value was loaded
          from one
        """

    def put(self, descriptor: Hashable, obj: Any):
        """Put ``obj`` to all caches under key ``descriptor``.
        Beware that the key may be changed (e.g. transformed to string) in
        sub-caches, leading to non-unique descriptors."""
        for cache in self.caches:
            cache.put(descriptor, obj)

    def load(self, descriptor: Hashable) -> Optional[Any]:
        """Load object stored under ``descriptor`` from the cache with highest
        precedence holding it.
        Possibly update other cache instances according to :py:attr:`sync_by`
        mode.
        """
        # obtain object
        obj = None
        first_hit = -1  # index at which value was found
        for i in range(len(self.caches)):
            obj = self.caches[i].load(descriptor)
            if obj is not None:
                first_hit = i
                break

        # update all other caches by sync_by mode:
        if (first_hit > 0 and self.sync_by == 'precedence') or \
                (first_hit >= 0 and self.sync_by == 'all'):
            caches_to_update = self.caches[:first_hit]
            if self.sync_by == 'all' and first_hit < len(self.caches) - 1:
                caches_to_update += self.caches[first_hit + 1:]
            for cache in caches_to_update:
                cache.put(descriptor, obj)
        return obj

    def clear(self):
        """Clears *all* caches in the cascade."""
        for cache in self.caches:
            cache.clear()

    def descriptors(self) -> Iterable:
        """This returns the united descriptor lists of all sub-caches.

        .. warning::
            This may be very computationally expensive, depending on
            the size of the lists to merge into a set. So use with care!
        """
        return {desc
                for descs in (cache.descriptors() for cache in self.caches)
                for desc in descs}

    def __repr__(self) -> str:
        return "{cls}({caches}{other_setts})".format(
            cls=self.__class__.__name__,
            caches=", ".join([repr(cache) for cache in self.caches]),
            other_setts=(", sync_by='{}'".format(self.sync_by)
                         if self.sync_by != 'precedence' else "")
        )

    def append(self, cache: Cache) -> 'CacheCascade':
        """Append ``cache`` to the cascade and return self."""
        self.caches.append(cache)
        return self

    def insert(self, i: int, cache: Cache) -> 'CacheCascade':
        """Insert ``cache`` at position ``i`` in cascade and return self."""
        self.caches.insert(i, cache)
        return self

    def remove(self, i: int) -> 'CacheCascade':
        """Remove the cache at position ``i`` in cascade and return self."""
        self.caches.pop(i)
        return self


class CacheTuple(Cache):
    """Cache the values of tuples using different caches.
    Given a descriptor and a tuple of objects, each value of the tuple
    is stored in a different cache under the given descriptor.

    Can be used e.g. to store transformed pairs of (input, target) using
    two different caches.
    """

    def __init__(self, *caches: Cache,
                 return_none_if: Union[str, int] = 'any'):
        """Init.

        :param caches: the caches to use to cache the values of given tuples
        :param return_none_if: see
            :py:attr:`~hybrid_learning.datasets.caching.CacheTuple.return_none_if`;
            may be one of ``'any', 'all', 'never', 1, 0, -1``.
        """
        # region value checks
        if len(caches) <= 0:
            raise ValueError("Empty caches given.")
        # endregion

        self.caches: Tuple[Cache] = caches
        """The tuple of caches to handle tuple values."""
        self.return_none_if: int = self._standardize_return_none_if(return_none_if)
        """Mode by which to return ``None`` on :py:meth:`load`.
        Possible modes:

        - ``1`` /``'any'``: Return ``None`` if any cache load returns ``None``.
        - ``0``/``'all'``: Return ``None`` if all cache loads return ``None``.
        - ``-1``/``'never'``: Do not return ``None``, but always a tuple
          (possibly only holding ``None`` values).

        The string specifiers will get mapped to integer values to increase
        speed.
        """

    def __repr__(self) -> str:
        return "{cls}({caches}{other_setts})".format(
            cls=self.__class__.__name__,
            caches=", ".join([repr(cache) for cache in self.caches]),
            other_setts=(", return_none_if={}".format(repr(self.return_none_if))
                         if self.return_none_if != 1 else "")
        )

    def load(self, descriptor: Hashable) -> Optional[Tuple[Any, ...]]:
        """Load all objects stored under ``descriptor`` and return as tuple.
        Return ``None`` according to the setting of :py:attr:`return_none_if`.
        """
        objs: Optional[Tuple[Any, ...]] = \
            tuple(cache.load(descriptor) for cache in self.caches)
        objs = self._to_none(objs, self.return_none_if)
        return objs

    def put(self, descriptor: Hashable, obj: Sequence[Any]):
        """Put ``obj[i]`` into ``caches[i]`` under key ``descriptor``."""
        try:
            for i in range(len(self.caches)):
                self.caches[i].put(descriptor, obj[i])
        except TypeError as exc:
            raise TypeError(("Given object to put must be tuple of length {} "
                             "with puttable data, but was {} (type: {})")
                            .format(len(self.caches), obj, type(obj))) from exc
        except IndexError as exc:
            raise IndexError(("Given object tuple was too short: Required "
                              "tuple length {} but was {}"
                              ).format(len(self.caches), len(obj))) from exc

    def clear(self):
        """Clear all caches in the tuple."""
        for cache in self.caches:
            cache.clear()

    def descriptors(self) -> Iterable:
        """Return all descriptors that occur in any of the given caches.

        .. warning::
            This may be slow, as the descriptor sets need to be united.
            Instead collect the descriptors of one cache if you know the
            caches have the same descriptors:
            ``tuple_cache.caches[0].descriptors()``.
        """
        return {desc
                for descs in (cache.descriptors() for cache in self.caches)
                for desc in descs}


class CacheDict(CacheTuple):
    """Cache the values of dicts using different caches.
    Under the hood this is a :py:class:`CacheTuple` matching keys to caches."""

    @property
    def cache_dict(self) -> Dict[Hashable, Cache]:
        """The dictionary of caches used."""
        return dict(zip(self.keys, self.caches))

    def __init__(self, cache_dict: Dict[Hashable, Cache],
                 return_none_if: Union[str, int] = 'any'):
        keys, caches = zip(*cache_dict.items())
        super().__init__(*caches, return_none_if=return_none_if)
        self.keys: Tuple[Hashable] = keys
        """The keys matching the caches.
        Caches are stored in :py:attr:`~CacheTuple.caches`."""

    def __repr__(self) -> str:
        return "{cls}({caches}{other_setts})".format(
            cls=self.__class__.__name__,
            caches=repr(self.cache_dict),
            other_setts=(", return_none_if={}".format(repr(self.return_none_if))
                         if self.return_none_if != 1 else "")
        )

    def load(self, descriptor: Hashable) -> Optional[Dict]:
        """Load a cached dict."""
        tuple_obj: Tuple = super().load(descriptor)
        if tuple_obj is None:
            return None
        return dict(zip(self.keys, tuple_obj))

    def put(self, descriptor: Hashable, obj: Dict):
        """Cache a dict."""
        tuple_obj: List = [obj[self.keys[i]] for i in range(len(self.keys))]
        super().put(descriptor, tuple_obj)
