Dataset Tooling
===============
The tooling for datasets is collected in the module :py:mod:`hybrid_learning.datasets`.
For details have a look at the :ref:`apiref/index:API Reference`.

.. contents::
    :depth: 2
    :local:
    :backlinks: top


Base dataset handles
--------------------
.. autosummary::
    :nosignatures:

    ~hybrid_learning.datasets.base.BaseDataset
    ~hybrid_learning.datasets.activations_handle.ActivationDatasetWrapper


Custom dataset handles
----------------------

.. py:currentmodule:: hybrid_learning.datasets.custom
.. autosummary::
    coco
    fasseg
    broden


Caching handles
---------------
Cache handles will allow to insert and read objects into/from a cache.
For details see :ref:`userguide/custom_dataset_handles/caching:Dataset Caching`.

.. automodsumm:: hybrid_learning.datasets.caching
    :skip: Hashable, Any, Optional, Dict, Iterable, List, Union, Callable, Tuple, Collection, Sequence, ToTensor
    :classes-only:
    :nosignatures:


Transformations
---------------
Transformations can be used to modify data tuples or values.

.. rubric:: Transformations for tuples

.. automodsumm:: hybrid_learning.datasets.transforms.tuple_transforms
    :skip: Any, Callable, Dict, Tuple, Optional, Iterable, List, Sequence, Set, Union, Mapping, Transform, Compose, TupleTransforms, TwoTupleTransforms, TwoTuple, TensorTwoTuple, TensorThreeTuple
    :classes-only:
    :nosignatures:

.. rubric:: Transformations for dicts

.. automodsumm:: hybrid_learning.datasets.transforms.dict_transforms
    :skip: Any, Callable, Dict, Tuple, Optional, Iterable, List, Sequence, Set, Union, Mapping, Transform
    :classes-only:
    :nosignatures:

.. rubric:: Transformations for (tensor) images

.. automodsumm:: hybrid_learning.datasets.transforms.image_transforms
    :skip: Any, Callable, Dict, Tuple, Optional, Iterable, Mapping, List, Sequence, Set, Union, BatchBoxBloat, BatchConvOp, BatchIntersectDecode2D, BatchIntersectEncode2D, BatchIoUEncode2D, Transform
    :classes-only:
    :nosignatures:

.. rubric:: Intersection and intersection over union encoders

.. automodsumm:: hybrid_learning.datasets.transforms.encoder
    :skip: Any, Callable, Dict, Tuple, Optional, Iterable, Mapping, List, Sequence, Set, Union
    :classes-only:
    :nosignatures:


Visualization and Utility Functions
-----------------------------------
.. rubric:: From :py:mod:`hybrid_learning.datasets.data_visualization`

.. automodsumm:: hybrid_learning.datasets.data_visualization
    :skip: to_pil_image
    :functions-only:

.. rubric:: From :py:mod:`hybrid_learning.datasets.base`

.. automodsumm:: hybrid_learning.datasets.base
    :functions-only:
