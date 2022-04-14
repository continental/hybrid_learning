"""Implementations of arithmetic comparison predicates for torch and numpy arrays.
A default precedence list for adding arithmetic comparison operations to a logic
is given in :py:data:`ARITHMETIC_OP_PRECEDENCE`.
"""
#  Copyright (c) 2022 Continental Automotive GmbH
from typing import Union, Tuple

import numpy as np
import torch

from ..logic_base.merge_operation import TorchOrNumpyOperation


class GreaterThan(TorchOrNumpyOperation):
    """Predicate of arity 2 that thresholds number values at a given threshold."""
    ARITY: int = 2
    SYMB: str = ">"

    @staticmethod
    def torch_operation(inp: torch.Tensor, thresh: Union[torch.Tensor, float]) -> torch.Tensor:
        """Threshold torch tensor ``inp`` to boolean tensor at ``thresh``."""
        return torch.gt(inp, thresh)

    @staticmethod
    def numpy_operation(inp: Union[np.ndarray, float], thresh: Union[np.ndarray, float]) -> np.ndarray:
        """Threshold array ``inp`` to boolean at ``thresh``."""
        return inp > thresh


class GreaterEqualsThan(TorchOrNumpyOperation):
    """Predicate of arity 2 that thresholds number values at a given threshold."""
    ARITY: int = 2
    SYMB: str = ">="

    @staticmethod
    def torch_operation(inp: torch.Tensor, thresh: Union[torch.Tensor, float]) -> torch.Tensor:
        """Threshold torch tensor ``inp`` to boolean tensor at ``thresh``."""
        return torch.ge(inp, thresh)

    @staticmethod
    def numpy_operation(inp: Union[np.ndarray, float], thresh: Union[np.ndarray, float]) -> np.ndarray:
        """Threshold array ``inp`` to boolean at ``thresh``."""
        return inp >= thresh


class LowerThan(TorchOrNumpyOperation):
    """Predicate of arity 2 that checks whether an input is below a threshold."""
    ARITY: int = 2
    SYMB: str = "<"

    @staticmethod
    def torch_operation(inp: torch.Tensor, thresh: Union[torch.Tensor, float]) -> torch.Tensor:
        """Threshold torch tensor ``inp`` to boolean tensor at ``thresh``."""
        return torch.lt(inp, thresh)

    @staticmethod
    def numpy_operation(inp: Union[np.ndarray, float], thresh: Union[np.ndarray, float]) -> np.ndarray:
        """Threshold array ``inp`` to boolean at ``thresh``."""
        return inp < thresh


class LowerEqualsThan(TorchOrNumpyOperation):
    """Predicate of arity 2 that checks whether an input is below or equals a threshold."""
    ARITY: int = 2
    SYMB: str = "<="

    @staticmethod
    def torch_operation(inp: torch.Tensor, thresh: Union[torch.Tensor, float]) -> torch.Tensor:
        """Threshold torch tensor ``inp`` to boolean tensor at ``thresh``."""
        return torch.le(inp, thresh)

    @staticmethod
    def numpy_operation(inp: Union[np.ndarray, float], thresh: Union[np.ndarray, float]) -> np.ndarray:
        """Threshold array ``inp`` to boolean at ``thresh``."""
        return inp <= thresh


class Equals(TorchOrNumpyOperation):
    """Predicate of arity 2 that checks element-wise equality between two arrays."""
    ARITY: int = 2
    SYMB: str = "=="

    @staticmethod
    def torch_operation(inp_a: Union[torch.Tensor, float], inp_b: Union[torch.Tensor, float]) -> torch.Tensor:
        """Where is ``inp_a`` equal to ``inp_b``?"""
        dtype = torch.promote_types(inp_a.dtype, inp_b.dtype)
        return torch.isclose(inp_a.to(dtype), inp_b.to(dtype))

    @staticmethod
    def numpy_operation(inp_a: Union[np.ndarray, float], inp_b: Union[np.ndarray, float]) -> np.ndarray:
        """Where is ``inp_a`` equal to ``inp_b``?"""
        return inp_a == inp_b


ARITHMETIC_OP_PRECEDENCE: Tuple = (
    GreaterThan,
    LowerThan,
    GreaterEqualsThan,
    LowerEqualsThan,
    Equals
)
