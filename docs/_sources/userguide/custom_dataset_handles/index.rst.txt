Custom dataset handles
======================

The custom dataset handles are meant to handle file stored image datasets,
and are all derived from :py:class:`~hybrid_learning.datasets.base.BaseDataset`.
Available custom handles are:

.. py:currentmodule:: hybrid_learning.datasets.custom
.. autosummary::
    :nosignatures:

    coco.keypoints_dataset.KeypointsDataset
    coco.mask_dataset.ConceptDataset
    broden.BrodenHandle
    fasseg.FASSEGHandle

Useful wrappers are:

.. py:currentmodule:: hybrid_learning.datasets.activations_handle
.. autosummary::
    :nosignatures:

    ActivationDatasetWrapper


.. rubric:: Tips and tricks: Merge transformations

Also worth noting are the
:py:class:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge`
``dict`` transformations that allow boolean combination of
masks and (boolean) classification labels. For details see
their class documentation and the example given in
:ref:`userguide/custom_dataset_handles/broden:Boolean Label Combinations`.

.. rubric:: Tips and tricks: Caching

All custom handles offer the option to specify a
:py:attr:`~hybrid_learning.datasets.base.BaseDataset.transforms_cache` handle.
This can be used to cache costly transformation or loading operations, e.g.
in-memory or by dumping intermediate results to files.
For details on available cache types and combination options have a look
at the :py:mod:`~hybrid_learning.datasets.caching` module.
See also :ref:`userguide/custom_dataset_handles/caching:Dataset Caching`


.. rubric:: Examples

In the following, usage examples for some handles are shown.
For details about the dataset formats have a look at the class documentations.

.. toctree::
    :maxdepth: 1

    coco
    broden
    activation_map_handling
    caching
