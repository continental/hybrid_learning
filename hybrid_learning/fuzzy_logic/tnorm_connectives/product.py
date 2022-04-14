#  Copyright (c) 2022 Continental Automotive GmbH
"""Product fuzzy logic and connectives."""
import collections
from functools import reduce
from typing import Union

import numpy as np
import torch

from .fuzzy_common import NOT
from ..logic_base import Logic as BaseLogic, stack_tensors
from ..logic_base.connectives import AbstractAND, AbstractOR, AbstractIMPLIES, AbstractIMPLIEDBY


class AND(AbstractAND):
    r"""Product AND operation.
    For a finite amount of truth values :math:`(a_i)_{i=1}^n` it calculates as

    .. math:: \bigwedge_{i=1}^n a_i = \prod_i a_i
    """

    @staticmethod
    def torch_operation(*inputs: torch.Tensor):
        """Product AND on pytorch tensors."""
        return torch.prod(stack_tensors(*inputs), dim=0)

    @staticmethod
    def numpy_operation(*inputs: Union[bool, np.ndarray, float]) -> Union[np.ndarray, float]:
        """Product AND on Booleans, numpy arrays and numbers."""
        return np.prod(np.broadcast_arrays(*inputs), axis=0)


class OR(AbstractOR):
    r"""Product OR operation.
    For two fuzzy truth values it calculates as

    .. math:: a \vee b = a + b - a\cdot b

    For a finite amount of truth values :math:`(a_i)_{i=1}^n` it calculates as

    .. math::
        \bigvee_{i=1}^n a_i
        = \sum_{i=1}^n (-1)^{i-1} \sum_{S\subset\{1,\dots,n\}, |S|=i} \prod_{j\in S} a_j
        = (a_1 + \dots + a_n) - (a_1a_2 + a_1a_3 + \dots + a_na_{n-1}) + (a_1a_2a_3 + \dots) - \dots

    To minimize the number of operations, the value for :math:`n>2`
    is calculated in a tail recursion instead of the global sum formula.
    """

    @staticmethod
    def general_or_binary(inp1: Union[torch.Tensor, bool, np.ndarray, float],
                          inp2: Union[torch.Tensor, bool, np.ndarray, float]):
        """Product OR formula for two fuzzy truth values.
        Works for both tensors and numpy types."""
        return inp1 + inp2 - (inp1 * inp2)

    @classmethod
    def general_or(cls, *inputs: Union[torch.Tensor, bool, np.ndarray, float]):
        """Product OR formula for arbitrary amount of fuzzy truth values
        as tail recursion. Works for both tensors and numpy types."""
        return reduce((lambda x, y: cls.general_or_binary(x, y)), inputs, 0)

    @classmethod
    def torch_operation(cls, *inputs: torch.Tensor) -> torch.Tensor:
        """Product OR on pytorch tensors."""
        return cls.general_or(*inputs)

    @classmethod
    def numpy_operation(cls, *inputs: Union[bool, np.ndarray, float]) -> Union[np.ndarray, float]:
        """Product OR on Booleans, numpy arrays and numbers."""
        return cls.general_or(*inputs)


class IMPLIES(AbstractIMPLIES):
    r"""Product logic IMPLIES connective.
    The connective is of arity 2 and calculates as
    :math:`a\rightarrow b = \min(1, \frac{b}{a})`.

    .. note::
        For product logic, implication is not the same as :math:`\neg a \vee b`.
    """

    @staticmethod
    def torch_operation(inp_a: torch.Tensor, inp_b: torch.Tensor) -> torch.Tensor:
        """Product logic IMPLIES on tensors."""
        one: torch.Tensor = torch.ones(1, dtype=inp_a.dtype, device=inp_a.device)
        zero: torch.Tensor = torch.zeros(1, dtype=inp_a.dtype, device=inp_a.device)
        # Calc min(1, b/a); Note that in torch: x/0 = inf and 0/0 = nan
        b_over_a: torch.Tensor = torch.nan_to_num(torch.clamp(inp_b / inp_a, min=0, max=1), nan=1.0)
        # Filter the cases where both a and b are close to 0
        return torch.where(torch.logical_or(torch.isclose(inp_a, inp_b), torch.isclose(inp_a, zero)),
                           one, b_over_a).to(inp_a.dtype)

    @staticmethod
    def numpy_operation(inp_a: np.ndarray, inp_b: np.ndarray) -> np.ndarray:
        """Product logic IMPLIES on numpy arrays."""
        # Calc min(1, b/a); Note that in numpy: x/0 = inf and 0/0 = nan (and both raise warnings)
        with np.errstate(divide='ignore', invalid='ignore'):
            b_over_a: np.ndarray = np.nan_to_num(np.clip(inp_b / inp_a, a_min=0, a_max=1), nan=1.0)
        # Filter the cases where both a and b are close to 0
        return np.where(np.logical_or(np.isclose(inp_a, inp_b), np.isclose(inp_a, 0)),
                        1, b_over_a)


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


class ProductLogic(BaseLogic):
    """Product fuzzy logic parser. For details see the super class."""
    DEFAULT_CONNECTIVES: collections.OrderedDict = collections.OrderedDict(
        BaseLogic.DEFAULT_CONNECTIVES, **dict(AND=AND, OR=OR, NOT=NOT, IMPLIES=IMPLIES, IMPLIEDBY=IMPLIEDBY))


Logic = ProductLogic
"""Alias for logic handle."""
