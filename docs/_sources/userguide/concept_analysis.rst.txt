Conducting a Concept Analysis
=============================

Concept analysis refers to

1. *finding* :py:class:`concept embeddings <hybrid_learning.concepts.models.embeddings.ConceptEmbedding>`
   within the latent spaces of a DNN,
2. *assessing* their strength (i.e. prediction performance as
   :py:class:`concept models <hybrid_learning.concepts.models.concept_models.concept_detection.ConceptDetectionModel2D>`),
3. possibly :py:class:`*aggregating* <hybrid_learning.concepts.analysis.analysis_handle.EmbeddingReduction>`
   several candidates to a single one for stabilization, and
4. *selecting* the final embedding for a :py:class:`concept <hybrid_learning.concepts.concepts.Concept>`.

The base class holding functionality for concept analysis and logging is
:py:class:`~hybrid_learning.concepts.analysis.analysis_handle.ConceptAnalysis`:

>>> from hybrid_learning.concepts.analysis import ConceptAnalysis

In the following, an exemplary concept analysis is conducted

- of the concept "face"
- on a AlexNet model
- in the backbone convolutional layer ``features.5``
- with 3 cross-validation splits

.. contents:: :local:


Preparation: Getting the concept
--------------------------------

First initialize the concept and the main model to analyse:

>>> # Main model & layers to analyse
>>> from torchvision.models import alexnet
>>> main_model = alexnet(pretrained=True)
>>> layers = ['features.5']  # more layers can be added
>>>
>>> # Concept data and concept
>>> import os
>>> from hybrid_learning.datasets.custom import coco
>>> from hybrid_learning.datasets import DataTriple
>>> root = os.path.join("dataset", "coco_test", "images", "{}2017")
>>> data = DataTriple(
...   train_val=coco.ConceptDataset([coco.BodyParts.FACE],
...       split="train", dataset_root=root.format("train"), img_size=(224, 224)).subset(num=10),
...   test=coco.ConceptDataset([coco.BodyParts.FACE],
...       split="test", dataset_root=root.format("val"), img_size=(224, 224)).subset(num=3)
... )
>>> from hybrid_learning.concepts.concepts import SegmentationConcept2D
>>> concept=SegmentationConcept2D("face", data, rel_size=(0.05, 0.05))


Collecting embeddings and performances
--------------------------------------

Now one can start an analysis with the desired amounts of runs on
the layers of interest. If the logging level is set to ``INFO``,
intermediate results are logged.

>>> analyser = ConceptAnalysis(
...     concept=concept, model=main_model,
...     # layer info (Iterable with IDs of layers to analyse)
...     layer_infos=layers,
...     # the number of independent cross-validation runs per layer
...     cross_val_runs=1,
...     # the number of runs/splits per cross-validation run
...     num_val_splits=3,
... )
>>> analysis_results = analyser.analysis()

The analysis results format can be turned into a :py:class:`pandas.DataFrame`:

>>> run_info = analysis_results.to_pandas()


Aggregation and selection
-------------------------

The layer-wise best embedding is collected by aggregating the results
for each layer
(see :py:meth:`~hybrid_learning.concepts.analysis.analysis_handle.ConceptAnalysis.embedding_reduction`).
To obtain the final best embedding over all layers, their evaluation results
are compared and the best one is selected
(see :py:meth:`~hybrid_learning.concepts.analysis.analysis_handle.ConceptAnalysis.best_layer_from_stats`).
Both steps are united in :py:meth:`~hybrid_learning.concepts.analysis.analysis_handle.ConceptAnalysis.best_embedding`,
which directly returns
the best embedding. To automate experiment saving use
:py:meth:`~hybrid_learning.concepts.analysis.analysis_handle.ConceptAnalysis.best_embedding_with_logging` instead.

.. note::
    If no analysis results are given, a complete new analysis is conducted.

>>> best_emb = analyser.best_embedding(analysis_results)
>>> type(best_emb)
<class 'hybrid_learning.concepts.models.embeddings.ConceptEmbedding'>


Performance assessment
----------------------

Prediction performance
......................

The performance of an embedding can be (re-)evaluated with respect to
the analysis settings by using
:py:meth:`~hybrid_learning.concepts.analysis.analysis_handle.ConceptAnalysis.evaluate_embedding`.

>>> from hybrid_learning.concepts.analysis.results import ResultsHandle
>>> best_emb_stats = analyser.evaluate_embedding(best_emb)
>>> print(ResultsHandle.emb_info_to_string(best_emb, best_emb_stats))
test_loss         ...
test_set_iou      ...
test_mean_iou     ...
normal vec len    ...
support factor    ...
scaling factor    ...


Variance and standard deviation
...............................

The variance and standard deviation of different runs within one layer
can be obtained via the embedding functionalities:

>>> # Embeddings for this layer from the analysis_results:
>>> embeddings = [e for embs, stats in
...               analysis_results.results[best_emb.layer_id].values()
...               for e in embs]
>>> # Variances of the different embedding aspects:
>>> from hybrid_learning.concepts.models import ConceptEmbedding
>>> std_normal_vec, std_support_factor, std_scaling_factor = \
...     ConceptEmbedding.std_deviation(embeddings)


Cosine distance
...............

Cosine distance between concept embeddings from one layer can be
calculated as follows (including the layer's ``best_emb`` in this case):

>>> import numpy as np
>>> import pandas as pd
>>> # pair-wise cosines with last row and column the best_embedding:
>>> all_vecs = ([e.normal_vec for e in embeddings] + [best_emb.normal_vec])
>>> keys = list(range(len(embeddings))) + ['best_emb']
>>> pairwise_cosines = pd.DataFrame([
...     # cosine dist between two vectors:
...     [np.sum(n1 * n2)/(np.linalg.norm(n1)*np.linalg.norm(n2))
...      for n1 in all_vecs]
...     for n2 in all_vecs], index = keys, columns = keys)
>>> # Mean cosine distance of best to other embeddings in the layer:
>>> mean_pairwise_cosine = pairwise_cosines.iloc[:-1, -1].mean()


Using an embedding
------------------

Finally, the embedding can be used to initialize a concept model
(here fed to visualization):

>>> from hybrid_learning.concepts.models import ConceptDetectionModel2D
>>> from hybrid_learning.concepts.models import ConceptDetection2DTrainTestHandle
>>> best_concept_model = ConceptDetectionModel2D.from_embedding(best_emb)
>>> from hybrid_learning.concepts.analysis import data_for_concept_model, visualization as vis
>>> vis.visualize_concept_model(
...     ConceptDetection2DTrainTestHandle(
...         best_concept_model,
...         data_for_concept_model(best_concept_model),
...         **analyser.train_val_args),
...     save_as=None
... )
