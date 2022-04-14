"""Base classes for some standard logical connectives with symbols and arity."""
#  Copyright (c) 2022 Continental Automotive GmbH

import abc
from typing import Union

from .merge_operation import Merge, TorchOrNumpyOperation

STANDARD_PRECEDENCE = ('IMPLIES', 'AND', 'OR', 'NOT')


class AbstractAND(TorchOrNumpyOperation, abc.ABC):
    """Base class for intersection/AND operations."""
    SYMB = "&&"
    IS_COMMUTATIVE = True


class AbstractOR(TorchOrNumpyOperation, abc.ABC):
    """Base class for union/OR operations."""
    SYMB = "||"
    IS_COMMUTATIVE = True


class AbstractNOT(TorchOrNumpyOperation, abc.ABC):
    """Base class for inversion/NOT operation.
    Only accepts one input key."""
    SYMB: str = "~"
    ARITY: int = 1

    @property
    def in_key(self) -> Union[str, Merge]:
        """The only operational input key."""
        return self.in_keys[0]


class AbstractIMPLIES(TorchOrNumpyOperation, abc.ABC):
    """Base class for pixel-wise implication logical connective.
    The connective ``a->b`` should return the following (by decreasing precedence of rules):

    - a == b (within tolerance): 1
    - b < a: < 1
    """
    SYMB: str = ">>"
    ARITY: int = 2


class AbstractIMPLIEDBY(TorchOrNumpyOperation, abc.ABC):
    """Base class for pixel-wise reverse implication logical connective."""
    SYMB: str = "<<"
    ARITY: int = 2
