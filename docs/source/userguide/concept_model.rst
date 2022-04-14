Using a concept model to find an embedding
==========================================

:py:class:`Concept embeddings <hybrid_learning.concepts.models.embeddings.ConceptEmbedding>`
in a DNN latent space can serve as the parameters for a linear classifier
that predicts the concept (resp. a segmentation mask of the concept) from
the latent space.
Models can e.g. be linear models or clustering.

The linear model classes
:py:class:`~hybrid_learning.concepts.models.concept_models.concept_detection.ConceptDetectionModel2D` and
:py:class:`~hybrid_learning.concepts.models.concept_models.concept_segmentation.ConceptSegmentationModel2D`
can be used to predict segmentation masks of a
:py:class:`concept <hybrid_learning.concepts.concepts.SegmentationConcept2D>`
respectively its estimated center.
They can be

- *initialized from an embedding* using
  :py:meth:`~hybrid_learning.concepts.models.concept_models.concept_detection.ConceptDetectionModel2D.from_embedding`, and
- used to *find an embedding by training* them on a set of samples
  associated to the :py:class:`concept <hybrid_learning.concepts.concepts.Concept>`, and
  then extracting the embedding using
  :py:meth:`~hybrid_learning.concepts.models.concept_models.concept_detection.ConceptDetectionModel2D.to_embedding`.

The training is done using their custom
:py:class:`~hybrid_learning.concepts.train_eval.base_handles.train_test_handle.TrainEvalHandle`.

In the following we consider an exemplary concept model for the concept
*"face"*.

.. contents:: :local:


Defining a concept model
------------------------

Preparation: Getting concept and main model
...........................................

First prepare the concept with its data.
Note the relative size input to the concept that is later used
to determine the convolution kernel size of the concept model.

>>> import os
>>> from hybrid_learning.datasets.custom import coco
>>> from hybrid_learning.datasets import DataTriple, DatasetSplit
>>> root = os.path.join("dataset", "coco_test", "images", "{}2017")
>>> data = DataTriple(
...   train_val=coco.ConceptDataset([coco.BodyParts.FACE],
...       split=DatasetSplit.TRAIN, dataset_root=root.format("train"),
...       img_size=(400, 400)
...       ).subset(num=10),
...   test=coco.ConceptDataset([coco.BodyParts.FACE],
...       split=DatasetSplit.TEST, dataset_root=root.format("val"),
...       img_size=(400, 400)
...       ).subset(num=3)
... )
>>> from hybrid_learning.concepts.concepts import SegmentationConcept2D
>>> concept = SegmentationConcept2D(name="face", data=data,
...                                 rel_size=(0.05, 0.05))
>>> concept
SegmentationConcept2D(
    name='face',
    data=DataTriple(...),
    rel_size=(0.05, 0.05)
)

For a concept model the main model and the layer to obtain concepts from
is needed, here a standard Mask R-CNN model.

>>> from torchvision.models.detection import maskrcnn_resnet50_fpn
>>> main_model = maskrcnn_resnet50_fpn(pretrained=True)
>>> layer_id = 'backbone.body.layer3'

Note the relative size input to the concept that is later used
to determine the convolution kernel size of the concept model.


Init a concept model
....................

Now the concept model can be defined (since this is exchangeable with
the segmentation equivalent, an alias for the class name is used).
By default, the kernel size is determined from the layer output size and the
concept :py:attr:`~hybrid_learning.concepts.concepts.SegmentationConcept2D.rel_size`.

>>> from hybrid_learning.concepts.models import ConceptDetectionModel2D as ConceptModel # same for segmentation
>>> concept_model = ConceptModel(
...    concept=concept, model=main_model, layer_id=layer_id,
...    # kernel_size=(3,3)
... )
>>> concept_model
ConceptDetectionModel2D(
  (padding): ZeroPad2d(padding=(0, 1, 0, 1), value=0.0)
  (concept_layer_0): Conv2d(1024, 1, kernel_size=(2, 2), stride=(1, 1))
  (activation): Sigmoid()
)
>>> for n, p in concept_model.named_parameters():
...     print("{param}: {size}".format(param=n, size=p.size()))
concept_layer_0.weight: torch.Size([1, 1024, 2, 2])
concept_layer_0.bias: torch.Size([1])


Training a concept model
------------------------

Given the concept model (including its concept) the training handle
can be defined. The data is available from the concept of the concept
model instance.


Defining the training handle
............................

When training or testing, the concept model must be fed with activation
maps derived from this data. Thus, the data splits need to be wrapped
by a :py:class:`~hybrid_learning.datasets.activations_handle.ActivationDatasetWrapper`.
The wrapper datasets are accessible via the data triple of the
training handle.

>>> from hybrid_learning.concepts.models import ConceptDetection2DTrainTestHandle as CMHandle
>>> from hybrid_learning.concepts import kpis
>>> from hybrid_learning.concepts.analysis import data_for_concept_model
>>> concept_model_handle = CMHandle(concept_model,
...                                 data=data_for_concept_model(concept_model),
...                                 loss_fn=kpis.TverskyLoss(),
...                                 metric_fns={'set_iou': kpis.SetIoU(), 'acc': kpis.Accuracy()})
>>> # Filled default values:
>>> for k, v in sorted(concept_model_handle.settings.items()):
...     print(k, ":", repr(v))
batch_size : 8
batch_size_hessian : 8
batch_size_val : 16
callback_context : {}
callbacks : [...LoggingCallback..., ...ProgressBarUpdater...]
data : DataTriple(
    train=ActivationDatasetWrapper(...),
    val=ActivationDatasetWrapper(...),
    test=ActivationDatasetWrapper(...)
)
device : device(...)
loss_fn : ReduceTuple(
    trafo=Compose(transforms=[SameSize(...), OnInput(trafo=Lambda(...))]),
    reduction=TverskyLoss(...)
early_stopping_handle : None
max_epochs : 5
metric_fns : {'set_iou': ReduceTuple(
    trafo=Compose(transforms=[SameSize(...), OnInput(trafo=Lambda(...))]),
    reduction=SetIoU(output_thresh=0.5,...)
), 'acc': ReduceTuple(...)}
model : ConceptDetectionModel2D(...)
nll_fn : ReduceTuple(...),
num_workers : 0
optimizer : ResettableOptimizer(optimizer_type=Adam, lr=0.01, ...)
show_progress_bars : True

Training and evaluation
.......................

Training and evaluation then is as simple as

>>> train_results, val_results = concept_model_handle.train()
>>> print(sorted(train_results.columns))
['train_loss', 'train_mean_iou', 'train_set_iou']
>>> print(sorted(val_results.columns))
['val_loss', 'val_mean_iou', 'val_set_iou']
>>> test_results = concept_model_handle.evaluate(mode=DatasetSplit.TEST)
>>> print(test_results)
test_loss        ...
test_set_iou     ...
test_mean_iou    ...
dtype: float64

Note that

- the training results are batch-wise
  (:py:class:`pandas.DataFrame` with multi-index of ``(epoch, batch``)),
- the validation results are epoch-wise
  (:py:class:`pandas.DataFrame` with index of epochs), and
- the test results are for the complete run (:py:class:`pandas.Series`).

The views can be merged using

>>> tot_batches = len(train_results.loc[(0, slice(None)), :])
>>> train_results_per_epoch = \
...     train_results.loc[(slice(None), tot_batches-1), :].droplevel(-1)
>>> results = {"test": test_results.copy(),
...            "val": val_results.iloc[-1],
...            "train": train_results_per_epoch.iloc[-1]}
>>> for key, split in results.items():
...     split.index = split.index.str.replace(key+"_", "")
>>> import pandas as pd
>>> print(pd.DataFrame(results))
              test       val     train
loss      ...
set_iou   ...
mean_iou  ...

