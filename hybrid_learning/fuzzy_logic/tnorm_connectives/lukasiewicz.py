#  Copyright (c) 2022 Continental Automotive GmbH
"""Lukasiewicz fuzzy logic and connectives."""
import collections
from typing import Union

import numpy as np
import torch

from .fuzzy_common import NOT
from ..logic_base import Logic as BaseLogic, stack_tensors
from ..logic_base.connectives import AbstractAND, AbstractOR, AbstractIMPLIES, AbstractIMPLIEDBY


class AND(AbstractAND):
    r"""Lukasiewicz AND operation.
    For a finite amount of truth values :math:`(a_i)_{i=1}^n` it calculates as

    .. math:: \bigwedge_{i=1}^n a_i = \max\left(0, \sum_{i=1}^n a_i - (n-1) \right)
    """

    @staticmethod
    def torch_operation(*inputs: torch.Tensor):
        """Lukasiewicz AND on pytorch tensors."""
        return torch.clamp(torch.sum(stack_tensors(*inputs), dim=0) - (len(inputs) - 1), min=0)

    @staticmethod
    def numpy_operation(*inputs: Union[bool, np.ndarray, float]) -> Union[np.ndarray, float]:
        """Lukasiewicz AND on Booleans, numpy arrays and numbers."""
        return np.clip(np.sum(np.broadcast_arrays(*inputs), axis=0) - (len(inputs) - 1), a_min=0, a_max=1)


class OR(AbstractOR):
    r"""Lukasiewicz OR operation.
    For a finite amount of truth values :math:`(a_i)_{i=1}^n` it calculates as

    .. math:: \bigvee_{i=1}^n a_i = \min\left(1, \sum_{i=1}^n a_i \right)
    """

    @staticmethod
    def torch_operation(*inputs: torch.Tensor) -> torch.Tensor:
        """Lukasiewicz OR on pytorch tensors."""
        return torch.clamp(torch.sum(stack_tensors(*inputs), dim=0), max=1)

    @staticmethod
    def numpy_operation(*inputs: Union[bool, np.ndarray, float]) -> Union[np.ndarray, float]:
        """Lukasiewicz OR on Booleans, numpy arrays and numbers."""
        return np.clip(np.sum(np.broadcast_arrays(*inputs), axis=0), a_min=0, a_max=1)


class IMPLIES(AbstractIMPLIES):
    r"""Lukasiewicz logic IMPLIES connective.
    The connective is of arity 2 and calculates as
    :math:`a\rightarrow b = \min(1, 1 - a + b)`.
    This is equivalent to :math:`\neg a \vee b`:

    >>> inp = [(.5, .75), (.75, .5), (.25, 1), (1, .25)]
    >>> a_implies_b, not_a_or_b = IMPLIES("a", "b"), OR(NOT("a"), "b")
    >>> for a, b in inp:
    ...     a_implies_b_out = a_implies_b({'a': a, 'b': b})[a_implies_b.out_key]
    ...     not_a_or_b_out = not_a_or_b({'a': a, 'b': b})[not_a_or_b.out_key]
    ...     assert not_a_or_b_out == a_implies_b_out, "a->b: {}\n~a||b: {}".format(a_implies_b_out, not_a_or_b_out)
    """

    @staticmethod
    def torch_operation(inp_a: torch.Tensor, inp_b: torch.Tensor) -> torch.Tensor:
        """Lukasiewicz logic IMPLIES on tensors."""
        a_implies_b: torch.Tensor = torch.clip(1 - inp_a + inp_b, min=0, max=1).to(inp_a.dtype)
        # Catch cases of a, b being very close:
        one = torch.ones(1, dtype=inp_a.dtype, device=inp_a.device)
        return torch.where(torch.isclose(inp_a, inp_b), one, a_implies_b)

    @staticmethod
    def numpy_operation(inp_a: np.ndarray, inp_b: np.ndarray) -> np.ndarray:
        """Lukasiewicz logic IMPLIES on numpy arrays."""
        a_implies_b: np.ndarray = np.clip(1 - inp_a + inp_b, a_min=0, a_max=1)
        # Catch cases of a, b being very close:
        return np.where(np.isclose(inp_a, inp_b), 1, a_implies_b)


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


class LukasiewiczLogic(BaseLogic):
    """Lukasiewicz logic parser. For details see the super class."""
    DEFAULT_CONNECTIVES: collections.OrderedDict = collections.OrderedDict(
        BaseLogic.DEFAULT_CONNECTIVES, **dict(AND=AND, OR=OR, NOT=NOT, IMPLIES=IMPLIES, IMPLIEDBY=IMPLIEDBY))


Logic = LukasiewiczLogic
"""Alias for logic handle."""
