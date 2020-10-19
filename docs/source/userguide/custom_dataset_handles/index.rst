Custom dataset handles
======================

The custom dataset handles are meant to handle file stored image datasets,
and are all derived from :py:class:`~hybrid_learning.datasets.base.BaseDataset`.
Available custom handles are:

.. py:currentmodule:: hybrid_learning.datasets.custom
.. autosummary::
    :nosignatures:

    coco.KeypointsDataset
    coco.ConceptDataset
    broden.BrodenHandle
    fasseg.FASSEGHandle

Useful wrappers are:

.. py:currentmodule:: hybrid_learning.datasets.activations_handle
.. autosummary::
    :nosignatures:

    ActivationDatasetWrapper


Also worth noting are the
:py:class:`~hybrid_learning.datasets.transforms.dict_transforms.Merge`
``dict`` transformations that allow boolean combination of
masks and (boolean) classification labels. For details see
their class documentation and the example given in
:ref:`userguide/custom_dataset_handles/broden:Boolean Label Combinations`.

In the following, usage examples for some handles are shown.
For details about the dataset formats have a look at the class documentations.

.. toctree::
    :maxdepth: 1

    coco
    broden
    activation_map_handling
