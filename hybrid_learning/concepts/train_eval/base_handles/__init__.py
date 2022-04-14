"""Basic handles for training and evaluation of pytorch models.
These are wrappers around the training and evaluation functions
from :py:mod:`~hybrid_learning.concepts.train_eval.train_eval_funs`,
with additional support of automated several epoch training."""

#  Copyright (c) 2022 Continental Automotive GmbH

from .early_stopping import EarlyStoppingHandle
from .resettable_optimizer import ResettableOptimizer
from .train_test_handle import TrainEvalHandle
