#  Copyright (c) 2022 Continental Automotive GmbH
"""Boolean logic and connectives."""
import abc
import collections
from typing import List, Dict, Any

import numpy as np
import torch

from .fuzzy_common import _tens_to_bool, _array_to_bool
from ..logic_base import Logic as BaseLogic, stack_tensors, TorchOrNumpyOperation
from ..logic_base.connectives import AbstractAND, AbstractOR, AbstractNOT, AbstractIMPLIES, AbstractIMPLIEDBY


class BoolTorchOrNumpyOperation(TorchOrNumpyOperation, abc.ABC):
    """Base class for Boolean operations that threshold their inputs before operation."""

    def __init__(self, *in_keys, bool_thresh: float = None, **kwargs):
        super().__init__(*in_keys, **kwargs)
        self.bool_thresh: float = bool_thresh if bool_thresh is not None else 0.5
        """Threshold in [0,1] to apply to masks for binarization before operation.
        Defaults to ``0.5``."""

    @property
    def settings(self) -> Dict[str, Any]:
        return {**super().settings, 'bool_thresh': self.bool_thresh}

    @property
    def setting_defaults(self) -> Dict[str, Any]:
        return {**super().setting_defaults, 'bool_thresh': 0.5}

    @staticmethod
    @abc.abstractmethod
    def bool_numpy_operation(*inputs):
        """Numpy operation assuming Boolean values."""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def bool_torch_operation(*inputs):
        """Torch operation assuming Boolean values."""
        raise NotImplementedError

    def numpy_operation(self, *inputs: np.ndarray, bool_thresh: float = None) -> np.ndarray:
        """Threshold the ``inputs`` at ``bool_thresh`` and apply operation.
        Default for ``bool_thresh`` first is :py:attr:`bool_thresh` (if available), then 0.5."""
        thresh = bool_thresh if bool_thresh is not None else getattr(self, 'bool_thresh', 0.5)
        inputs: List[np.ndarray] = _array_to_bool(*inputs, thresh=thresh)
        return self.bool_numpy_operation(*inputs)

    def torch_operation(self, *inputs: torch.Tensor, bool_thresh: float = None):
        """AND on pytorch tensors."""
        dtype = inputs[0].dtype
        thresh: float = bool_thresh if bool_thresh is not None else getattr(self, 'bool_thresh', 0.5)
        inputs: List[torch.BoolTensor] = _tens_to_bool(*inputs, thresh=thresh)
        return self.bool_torch_operation(*inputs).to(dtype)


class AND(AbstractAND, BoolTorchOrNumpyOperation):
    """Intersection/AND operation on binary masks and scalars.
    Store intersection of :py:attr:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge.in_keys`
    masks as :py:attr:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge.out_key`.
    AND with just one input key ``x`` is treated like ``x&&True``, i.e. identity."""

    @staticmethod
    def bool_numpy_operation(*inputs: np.ndarray) -> np.ndarray:
        """AND on numpy-like vectors."""
        return np.prod(np.broadcast_arrays(*inputs), axis=0)

    @staticmethod
    def bool_torch_operation(*inputs: torch.Tensor):
        """AND on pytorch tensors."""
        return torch.prod(stack_tensors(*inputs), dim=0)


class OR(AbstractOR, BoolTorchOrNumpyOperation):
    """Union/OR operation on binary masks and scalars.
    Store union of :py:attr:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge.in_keys`
    masks as :py:attr:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge.out_key`.
    OR with just one input key ``x`` is treated like ``x||True``, i.e. identity."""

    @staticmethod
    def bool_numpy_operation(*inputs: np.ndarray) -> np.ndarray:
        """OR on numpy-like vectors."""
        return np.sum(np.broadcast_arrays(*inputs), axis=0) > 0

    @staticmethod
    def bool_torch_operation(*inputs: torch.Tensor) -> torch.Tensor:
        """OR on torch tensors."""
        return torch.clamp(torch.sum(stack_tensors(*inputs), dim=0), min=0, max=1)


class NOT(AbstractNOT, BoolTorchOrNumpyOperation):
    """Inversion/NOT operation on binary masks and scalars.
    Store inverted version of :py:attr:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge.in_keys`
    as :py:attr:`~hybrid_learning.fuzzy_logic.logic_base.merge_operation.Merge.out_key`.
    Only accepts one input key."""

    @staticmethod
    def bool_numpy_operation(inp: np.ndarray) -> np.ndarray:
        """NOT on numpy-like vectors."""
        return np.logical_not(inp)

    @staticmethod
    def bool_torch_operation(inp: torch.Tensor) -> torch.Tensor:
        """NOT on pytorch tensors."""
        return torch.logical_not(inp)


class IMPLIES(AbstractIMPLIES, BoolTorchOrNumpyOperation):
    r"""Pixel-wise implication logical connective on binary masks and scalars.
    It is realized as the binary equivalent
    :math:`\left(a\rightarrow b\right) = \left((\neg a) \vee b\right)`."""

    @staticmethod
    def bool_numpy_operation(inp_a: np.ndarray, inp_b: np.array) -> np.ndarray:
        """IMPLIES on two numpy-like tensors."""
        return np.logical_or(np.logical_not(inp_a), inp_b)

    @staticmethod
    def bool_torch_operation(inp_a: torch.Tensor, inp_b: torch.Tensor) -> torch.Tensor:
        """IMPLIES on two torch tensors.
        The output device is the one of the first tensor."""
        return torch.logical_or(torch.logical_not(inp_a), inp_b)


class IMPLIEDBY(AbstractIMPLIEDBY, BoolTorchOrNumpyOperation):
    r"""Goedel inverted logic IMPLIES connective.
    For details see :py:class:`IMPLIES`."""

    @staticmethod
    def bool_torch_operation(inp_a: torch.Tensor, inp_b: torch.Tensor) -> torch.Tensor:
        """Inverted Goedel logical IMPLIES on tensors."""
        return IMPLIES.bool_torch_operation(inp_b, inp_a)

    @staticmethod
    def bool_numpy_operation(inp_a: np.ndarray, inp_b: np.ndarray) -> np.ndarray:
        """Inverted Goedel logical IMPLIES on numpy arrays."""
        return IMPLIES.bool_numpy_operation(inp_b, inp_a)


class BooleanLogic(BaseLogic):
    """Boolean logic and parser.
    For details and examples see the super-class.
    """
    DEFAULT_CONNECTIVES: collections.OrderedDict = collections.OrderedDict(
        BaseLogic.DEFAULT_CONNECTIVES, **dict(AND=AND, OR=OR, NOT=NOT, IMPLIES=IMPLIES, IMPLIEDBY=IMPLIEDBY))


Logic = BooleanLogic
"""Alias for logic handle."""
