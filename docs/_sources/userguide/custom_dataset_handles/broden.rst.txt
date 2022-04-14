Broden
======

The :py:class:`~hybrid_learning.datasets.custom.broden.BrodenHandle` is a handle
for datasets following the format of the Broad and Densely Labeled Dataset (Broden)
dataset introduced in `this paper <https://doi.org/10.1109/CVPR.2017.354>`_.

.. note::
    The original Broden dataset is not mandatory for usage of the provided handle.
    Any dataset following the format of the Broden dataset is supported.

The source code for downloading the original Broden dataset including the dataset specification can be found
`here <https://github.com/CSAILVision/NetDissect-Lite/blob/master/script/dlbroden.sh>`_
*(no code was used from this repository)*.

The handle's init options allow for

- *shuffling*,
- *length* restriction,
- *pruning of ``NaN``* entries, and
- restriction to one of the original *Broden splits*.

Further dataset customizations are described below.

The item format of the dataset is (before application of any transformation)
a tuple of the Broden input image and a dictionary ``{label_name: annotation}``
containing the annotations for all specified labels.


.. contents:: :local:


Pruning and Balancing
---------------------

Further *pruning*, e.g. of empty labels/masks, can be achieved using
:py:meth:`~hybrid_learning.datasets.custom.broden.BrodenHandle.prune` with a custom
pruning condition (a function on the dataset outputs with boolean output).
A simple condition for large persons would be:

>>> condition = (lambda d: d["person"] is None  # person annotation exists
...                        or d["person"].sum()/d["person"].numel() < 0.1)

*Balancing* e.g. of (custom) classification labels can be automated
using :py:meth:`~hybrid_learning.datasets.custom.broden.BrodenHandle.balance`.
This accepts a condition for being in the positive class, and a
proportion of positive samples that should be in the final dataset.
Then either positive or negative samples are pruned until the desired
ratio is achieved as best as possible.
A condition to define samples labeled as striped as positive is

>>> condition = lambda d: d["striped"]==True


Boolean Label Combinations
--------------------------

The Broden dataset has some drawbacks which may---depending on the
use-case---require
:py:class:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge`
transformations of several labels (all of union, intersection, and inversion
may be needed):

- One *part* label may belong to different super-*objects* labels,
  e.g. all of *person*, *cat*, *train* have a *head*, and both *person* and
  *chair* feature an *arm*.
- Part-hierarchy only has one level.
  Since *part* labels may not overlap, sub-parts are excluded from
  parent-parts where available. E.g. for Pascal-Part, *heads* are
  segmented excluding any *nose*, *eye*, *ear*.
  However, depending on the underlying *part*-dataset (ADE or Pascal-Part),
  different sub-parts are available, thus ADE does not cut out the *nose*
  from a *head*.

*Boolean combinations* of the labels to overcome above specialties,
can be achieved by :py:mod:`~hybrid_learning.fuzzy_logic.tnorm_connectives.boolean.BooleanLogic`
operations. This is eased by using the
:py:meth:`~hybrid_learning.datasets.custom.broden.BrodenHandle.custom_label`
generator function, which accepts such a
:py:mod:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge`
transformation and returns a dataset instance with the required labels
and transformation. The ground truth output of this dataset then is
only the custom label, no ``dict`` any more (mind this for defining further
pruning rules). More precisely one can automate

- label selection (from a given transformation),
- transforms definition for the dataset to get one combined label output,
- pruning,
- balancing of the one label.

An example label spec for persons in crop gardens would be

>>> from hybrid_learning.fuzzy_logic.tnorm_connectives.boolean import AND, OR
>>> label_spec = AND('person', OR('vegetable_garden-s', 'herb_garden-s'))

For further details on how to specify logical mask combinations of labels
see :ref:`userguide/fuzzy_logic:Fuzzy Logic Operations`.
