Dataset Caching
===============

The base dataset class :py:class:`~hybrid_learning.datasets.base.BaseDataset`
defined in this library supports basic caching functionality.
Respective cache handles are defined in :py:mod:`~hybrid_learning.datasets.caching`.


About Caches
------------

The base class of cache handles is :py:class:`~hybrid_learning.datasets.caching.Cache`.
A cache implementation provides methods
:py:meth:`~hybrid_learning.datasets.caching.Cache.put` and
:py:meth:`~hybrid_learning.datasets.caching.Cache.load`
which will store a given object respectively load a previously pushed object using a
given descriptor. If load is called upon a descriptor for which no object
has been stored so far, ``None`` is returned.

An example is here given for a file cache that stores objects to disk using ``torch.save()``:

>>> import os, torch
>>> from hybrid_learning.datasets.caching import PTCache
>>> mycache = PTCache(cache_root=".pytest_tmpdir")
>>> obj: torch.Tensor = torch.tensor([1,2,3])
>>> descriptor: str = "unique_descriptor"
>>> mycache.put(descriptor, obj)
>>> assert os.path.exists(os.path.join(mycache.cache_root, descriptor + ".pt"))
>>> print(mycache.load(descriptor))
tensor([1., 2., 3.])
>>> print(mycache.load("descriptor_of_not_yet_stored_object"))
None


Adding a Cache to a Dataset
---------------------------

Implementations of :py:class:`~hybrid_learning.datasets.base.BaseDataset`
allow to specify a cache handle in order to cache transformed items.
They feature a :py:meth:`~hybrid_learning.datasets.base.BaseDataset.descriptor`
method that returns the unique descriptor of the sample at an index.
If the dataset is assigned a :py:attr:`~hybrid_learning.datasets.base.BaseDataset.transforms_cache`
handle, these descriptors are used to load or put a transformed sample into the cache.
To apply further transformations to items, independent on whether they were
loaded from cache or newly transformed using :py:attr:`~hybrid_learning.datasets.base.BaseDataset.transforms`,
use the :py:meth:`~hybrid_learning.datasets.base.BaseDataset.after_cache_transforms`.


>>> from hybrid_learning.datasets.custom import coco
>>> concept_data = coco.ConceptDataset(
...     body_parts=[coco.BodyParts.FACE],
...     dataset_root=os.path.join("dataset", "coco_test", "images", "train2017"),
...     transforms_cache=mycache
... )


Also, caching can be applied manually by decorating
``__getitem__``-like functions with a cache's
:py:meth:`~hybrid_learning.datasets.caching.Cache.wrap` method.
