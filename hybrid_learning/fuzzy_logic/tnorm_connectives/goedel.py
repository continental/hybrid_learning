#  Copyright (c) 2022 Continental Automotive GmbH
"""Goedel fuzzy logic and connectives."""
import collections
from typing import Union

import numpy as np
import torch

from .fuzzy_common import NOT
from ..logic_base import Logic as BaseLogic, stack_tensors
from ..logic_base.connectives import AbstractAND, AbstractOR, AbstractIMPLIES, AbstractIMPLIEDBY


class AND(AbstractAND):
    r"""Goedel or Minimum AND operation.
    For a finite amount of truth values :math:`(a_i)_{i=1}^n` it calculates as

    .. math:: \bigwedge_{i=1}^n a_i = \min_i a_i
    """

    @staticmethod
    def torch_operation(*inputs: torch.Tensor):
        """Goedel AND on pytorch tensors."""
        return torch.amin(stack_tensors(*inputs), dim=0)

    @staticmethod
    def numpy_operation(*inputs: Union[bool, np.ndarray, float]) -> Union[np.ndarray, float]:
        """Goedel AND on Booleans, numpy arrays and numbers."""
        return np.amin(np.broadcast_arrays(*inputs), axis=0)


class OR(AbstractOR):
    r"""Goedel or Maximum OR operation.
    For a finite amount of truth values :math:`(a_i)_{i=1}^n` it calculates as

    .. math:: \bigvee_{i=1}^n a_i = \max_i a_i
    """

    @staticmethod
    def torch_operation(*inputs: torch.Tensor) -> torch.Tensor:
        """Goedel OR on pytorch tensors."""
        return torch.amax(stack_tensors(*inputs), dim=0)

    @staticmethod
    def numpy_operation(*inputs: Union[bool, np.ndarray, float]) -> Union[np.ndarray, float]:
        """Goedel OR on Booleans, numpy arrays and numbers."""
        return np.amax(np.broadcast_arrays(*inputs), axis=0)


class IMPLIES(AbstractIMPLIES):
    r"""Goedel logic IMPLIES connective.
    The connective is of arity 2 and calculates as
    :math:`a\rightarrow b = 1 if a < b else b`.

    .. note::
        For Goedel logic, implication is not the same as :math:`\neg a \vee b`.
    """

    @staticmethod
    def torch_operation(inp_a: torch.Tensor, inp_b: torch.Tensor) -> torch.Tensor:
        """Goedel logic IMPLIES on tensors."""
        one = torch.ones(1, dtype=inp_a.dtype, device=inp_a.device)
        return torch.where(torch.logical_or(inp_a <= inp_b, torch.isclose(inp_a, inp_b)).bool(),
                           one, inp_b).to(inp_a.dtype)

    @staticmethod
    def numpy_operation(inp_a: np.ndarray, inp_b: np.ndarray) -> np.ndarray:
        """Goedel logic IMPLIES on numpy arrays."""
        return np.where(np.logical_or(inp_a <= inp_b, np.isclose(inp_a, inp_b)),
                        1, inp_b)


class IMPLIEDBY(AbstractIMPLIEDBY):
    r"""Goedel inverted logic IMPLIES connective.
    For details see :py:class:`IMPLIES`."""

    @staticmethod
    def torch_operation(inp_a: torch.Tensor, inp_b: torch.Tensor) -> torch.Tensor:
        """Inverted Goedel logical IMPLIES on tensors."""
        return IMPLIES.torch_operation(inp_b, inp_a)

    @staticmethod
    def numpy_operation(inp_a: np.ndarray, inp_b: np.ndarray) -> np.ndarray:
        """Inverted Goedel logical IMPLIES on numpy arrays."""
        return IMPLIES.numpy_operation(inp_b, inp_a)


class GoedelLogic(BaseLogic):
    """Goedel fuzzy logic parser. For details see the super class."""
    DEFAULT_CONNECTIVES: collections.OrderedDict = collections.OrderedDict(
        BaseLogic.DEFAULT_CONNECTIVES, **dict(AND=AND, OR=OR, NOT=NOT, IMPLIES=IMPLIES, IMPLIEDBY=IMPLIEDBY))


Logic = GoedelLogic
"""Alias for logic handle."""
