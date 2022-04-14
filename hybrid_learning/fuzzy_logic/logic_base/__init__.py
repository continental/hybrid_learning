"""Base classes for defining and operating on logics.
This includes defining logical operations, defining a logic,
and basic parsing of logical formulas.
"""
#  Copyright (c) 2022 Continental Automotive GmbH


from .logic import Logic
from .merge_operation import Merge, MergeBuilder, TorchOrNumpyOperation, TorchOperation, stack_tensors
from .parsing import FormulaParser
