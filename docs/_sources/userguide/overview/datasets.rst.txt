Dataset tooling
===============
The tooling for datasets is collected in the module :py:mod:`hybrid_learning.datasets`.
For details have a look at the :ref:`apiref/index:API Reference`.

.. contents::
    :depth: 2
    :local:
    :backlinks: top


Base Dataset Handles
--------------------
.. autosummary::
    :nosignatures:

    ~hybrid_learning.datasets.base.BaseDataset
    ~hybrid_learning.datasets.activations_handle.ActivationDatasetWrapper


Custom Dataset Handles
----------------------

.. py:currentmodule:: hybrid_learning.datasets.custom
.. autosummary::
    coco
    fasseg
    broden


Transformations
---------------
Transformations can be used to modify data tuples or values.

.. rubric:: Transformations for Tuples

.. automodsumm:: hybrid_learning.datasets.transforms.tuple_transforms
    :skip: Any, Callable, Dict, Sequence, Tuple, List, settings_to_repr
    :nosignatures:

.. rubric:: Transformations for Dicts

.. automodsumm:: hybrid_learning.datasets.transforms.dict_transforms
    :skip: Any, Dict, Optional, Sequence, List, Iterable, Set, Union
    :nosignatures:

.. rubric:: Transformations for (Tensor) Images

.. automodsumm:: hybrid_learning.datasets.transforms.image_transforms
    :skip: Tuple, Callable, Dict, Any, Optional, settings_to_repr
    :nosignatures:

.. rubric:: Intersection and Intersection over Union Encoders

.. automodsumm:: hybrid_learning.datasets.transforms.encoder
    :skip: Tuple, Dict, Any, Sequence, settings_to_repr, WithThresh
    :nosignatures:


Visualization and Utility Functions
-----------------------------------
.. automodsumm:: hybrid_learning.datasets.data_visualization
    :functions-only:

.. automodsumm:: hybrid_learning.datasets.base
    :functions-only:
