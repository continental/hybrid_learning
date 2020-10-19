Concept analysis and usage tooling
===================================
The tooling for concept analysis is collected in the module :py:mod:`hybrid_learning.concepts`.
For details have a look at the :ref:`apiref/index:API Reference`.

.. contents::
    :depth: 2
    :local:
    :backlinks: top


Analysis Handles
----------------
The actual analysis handle is :py:class:`~hybrid_learning.concepts.analysis.ConceptAnalysis`.


Helper and Utility Classes
--------------------------

Training and Validation Interface
.................................
The following handle classes from :py:mod:`~hybrid_learning.concepts.models` are used to provide a generic
training and validation interface for pytorch models:

.. py:currentmodule:: hybrid_learning.concepts.models.base_handles
.. autosummary::
    :nosignatures:

    ~early_stopping.EarlyStoppingHandle
    ~resettable_optimizer.ResettableOptimizer
    ~train_test_handle.TrainEvalHandle

Training KPIs
..............
Some loss and metric functions are available for different training setups of the concept models.
They all inherit from :py:class:`torch.nn.Module`.

.. automodsumm:: hybrid_learning.concepts.kpis
    :skip: Tuple, Callable, Union, Dict, Sequence, Any
    :nosignatures:

Intermediate Output Retrieval
.............................
To retrieve the intermediate output of pytorch models, the `pytorch hooks mechanism`_ is used.
Wrappers for adding/retrieving intermediate output of DNNs are defined in the module
:py:mod:`~hybrid_learning.concepts.models.model_extension`:

.. automodsumm:: hybrid_learning.concepts.models.model_extension
    :skip: Iterable, Dict, Optional, List, Sequence, Tuple, Any, Callable
    :nosignatures:


Concept and Concept Embedding Modelling
---------------------------------------
A concept is defined by sample data points and is modeled as an instance of
:py:class:`~hybrid_learning.concepts.concepts.Concept`.
A concept can be used to train a model for predicting/detecting it properly.
Here, concepts are used to train a linear *concept model* (e.g. an instance of
:py:class:`~hybrid_learning.concepts.models.concept_segmentation.ConceptSegmentationModel2D`),
which predicts a concept from the intermediate output of a DNN.
The parameters of a trained concept model describe a linear *concept embedding*
of a concept into a layer of a DNN. Embeddings and operations thereon are
modeled in :py:class:`~hybrid_learning.concepts.embeddings.ConceptEmbedding`.
For translation between concept model and concept embedding use

- :py:meth:`~hybrid_learning.concepts.models.concept_detection.ConceptDetectionModel2D.to_embedding` for model to embedding, and
- :py:meth:`~hybrid_learning.concepts.models.concept_detection.ConceptDetectionModel2D.from_embedding` for embedding to model.

The relevant modelling classes are:

.. py:currentmodule:: hybrid_learning.concepts
.. autosummary::
    :nosignatures:

    ~concepts.Concept
    ~concepts.SegmentationConcept2D
    ~embeddings.ConceptEmbedding
    ~models.concept_detection.ConceptDetectionModel2D
    ~models.concept_segmentation.ConceptSegmentationModel2D

The concept model classes
(:py:class:`~hybrid_learning.concepts.models.concept_detection.ConceptDetectionModel2D` and
:py:class:`~hybrid_learning.concepts.models.concept_segmentation.ConceptSegmentationModel2D`)
are accompanied by custom handles for training and testing derived from
:py:meth:`~hybrid_learning.concepts.models.base_handles.train_test_handle.TrainEvalHandle`.
The handles are:

.. py:currentmodule:: hybrid_learning.concepts.models
.. autosummary::
    :nosignatures:

    ~concept_detection.ConceptDetection2DTrainTestHandle
    ~concept_segmentation.ConceptSegmentation2DTrainTestHandle


.. _pytorch hooks mechanism: https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks
