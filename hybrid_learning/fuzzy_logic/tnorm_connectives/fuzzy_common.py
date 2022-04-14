"""Common helper functions and connectives for fuzzy logics."""
#  Copyright (c) 2022 Continental Automotive GmbH
from typing import Union, List

import numpy as np
import torch

from hybrid_learning.fuzzy_logic.logic_base.connectives import AbstractNOT


def _tens_to_bool(*inputs: torch.Tensor, thresh: Union[torch.Tensor, float] = 0.5) -> List[torch.BoolTensor]:
    """Make pytorch tensors boolean by the element-wise rule
    ``True if a > 0.5 else False``."""
    return [inp.__ge__(thresh) for inp in inputs]


def _array_to_bool(*inputs: np.ndarray, thresh: Union[np.ndarray, float] = 0.5) -> List[np.ndarray]:
    """Make numpy arrays boolean by the element-wise rule
    ``True if a > 0.5 else False``."""
    return [inp >= thresh for inp in inputs]


class NOT(AbstractNOT):
    r"""Fuzzy logic NOT.
    Calculates as :math:`1-a` for a truth value :math:`a\in[0,1]`."""

    @staticmethod
    def torch_operation(*inputs: torch.Tensor) -> torch.Tensor:
        """Fuzzy NOT operation."""
        assert len(inputs) == 1
        return 1 - inputs[0] if not inputs[0].dtype == torch.bool \
            else torch.logical_not(inputs[0])

    @staticmethod
    def numpy_operation(*inputs: np.ndarray) -> np.ndarray:
        """Fuzzy NOT operation."""
        assert len(inputs) == 1
        return 1 - inputs[0]
