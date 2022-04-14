"""Dataset handles and manipulation methods.

All datasets should be derived from
:py:class:`hybrid_learning.datasets.base.BaseDataset` which acts as Sequence
over tuples of ``(input, ground_truth)``.

This module implements a couple of useful

- dataset handles
  (see :py:mod:`~hybrid_learning.datasets.custom`)
- transformations
  (see :py:mod:`~hybrid_learning.datasets.transforms`)
- cache handles
  (see :py:mod:`~hybrid_learning.datasets.caching`)
- visualization helpers
  (see :py:mod:`~hybrid_learning.datasets.data_visualization`)
"""

#  Copyright (c) 2022 Continental Automotive GmbH

from .activations_handle import ActivationDatasetWrapper
# All standard manipulation methods:
from .base import *
from .data_visualization import *
