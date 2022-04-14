#  Copyright (c) 2022 Continental Automotive GmbH
"""Base classes and simple implementations for standard quantifiers."""

import abc
from typing import Union, Sequence, Callable, Literal, Optional, Tuple, get_args

import numpy as np
import torch

from .logic_base.merge_operation import TorchOrNumpyOperation, TorchOperation, Merge, _NumericType

_REDUCTION_TYPE_KEYS = Literal['min', 'max', 'mean']


class AbstractQuantifier(Merge, abc.ABC):
    """Abstract base class for quantifiers accepting one tensor/list that is to reduce along a dimension.
    To instantiate, create a sub-class specifying the ``SYMB`` and ``DEFAULT_REDUCTION`` variables."""
    ARITY: int = 1
    DEFAULT_REDUCTION: Union[str, Merge] = None

    @property
    def settings(self):
        return {**super().settings, **dict(reduction=self.reduction, dim=self.dim)}

    @property
    def setting_defaults(self):
        return {**super().setting_defaults, **dict(reduction=self.DEFAULT_REDUCTION, dim=None)}

    def __init__(self, *in_keys,
                 reduction: Union[str, Merge] = None,
                 dim: Union[Sequence[int], int] = None, **kwargs):
        super().__init__(*in_keys, **kwargs)
        self.reduction: Union[str, Merge] = reduction \
            if reduction is not None else self.DEFAULT_REDUCTION
        """The variadic logical operation used to define the reduction along one axis.
        If set to string ``'max'``, a simple maximum is used (i.e. Goedel OR)."""
        self.dim: Optional[Tuple[int]] = (dim,) if isinstance(dim, int) else (tuple(dim) if dim else None)
        """The dimensions along which to reduce the given input."""


class AbstractTorchOrNumpyQuantifier(TorchOrNumpyOperation, AbstractQuantifier, abc.ABC):
    """Abstract base class for reducing torch tensors along given dimensions."""

    def reduction(self, tens: _NumericType) -> _NumericType:
        """Reduce the torch tensor."""
        if self.reduction in get_args(_REDUCTION_TYPE_KEYS):
            self.reduction: _REDUCTION_TYPE_KEYS
            if isinstance(tens, torch.Tensor):
                return self.torch_reduction_by_key(self.reduction, tens, dim=self.dim)
            return self.numpy_reduction_by_key(self.reduction, tens, dim=self.dim)
        return self.reduction_by_op(self.reduction, tens, dim=self.dim)

    torch_operation = reduction
    numpy_operation = reduction

    @staticmethod
    def reduction_by_op(reduction: Callable[[_NumericType], _NumericType],
                        tens: torch.Tensor, dim: Sequence[int]):
        """Iteratively reduce all dimensions ``dim`` of ``tens`` using the callable ``reduction``.
        In case of an empty tensor, the tensor is return unchanged."""
        # Determine and validate dimensions
        tens_size: Sequence[int] = tens.size() if isinstance(tens, torch.Tensor) \
            else (tens.shape if isinstance(tens, np.ndarray) else 1)
        if np.prod(tens_size) == 0:  # Empty tensor?
            return tens
        num_tens_dims: int = len(tens_size)
        dims: Sequence[int] = dim if dim is not None else range(num_tens_dims)
        assert all(dim < num_tens_dims if dim >= 0 else -dim <= num_tens_dims
                   for dim in dims)

        # Iterate over dimensions to reduce
        for num_processed, dim in enumerate(sorted(dims)):
            dim -= num_processed
            if dim != 0:
                tens = tens.moveaxis(dim, 0) if isinstance(tens, torch.Tensor) \
                    else np.moveaxis(tens, dim, 0)
            tens: _NumericType = reduction(tens)
        return tens

    @staticmethod
    def torch_reduction_by_key(reduction_key: _REDUCTION_TYPE_KEYS, tens: torch.Tensor,
                               dim: Union[Sequence[int], int] = None):
        """Apply a simple reduction of torch ``tens`` along dimension ``dim``.
        The reduction type is specified by ``reduction``."""
        if tens.numel() == 0:
            return tens
        if reduction_key == 'max':
            return tens.amax(dim=dim)
        elif reduction_key == 'min':
            return tens.amin(dim=dim)
        elif reduction_key == 'mean':
            kwargs = dict(dim=dim) if dim is not None else {}
            return torch.mean(tens.float(), **kwargs)

    @staticmethod
    def numpy_reduction_by_key(reduction_key: _REDUCTION_TYPE_KEYS, tens: np.ndarray,
                               dim: Union[Sequence[int], int] = None):
        """Apply a simple reduction of numpy ``tens`` along dimension ``dim``.
        The reduction type is specified by ``reduction``."""
        if tens.size == 0:
            return tens
        dim = dim if dim is None or isinstance(dim, int) else tuple(dim)
        if reduction_key == 'max':
            return np.amax(tens, axis=dim)
        elif reduction_key == 'min':
            return np.amin(tens, axis=dim)
        elif reduction_key == 'mean':
            return np.mean(tens, axis=dim)


class ANY(AbstractTorchOrNumpyQuantifier):
    """Reduce a single tensor along given dimensions using logical OR."""
    SYMB: str = "Any"
    DEFAULT_REDUCTION: _REDUCTION_TYPE_KEYS = 'max'


class ALL(AbstractTorchOrNumpyQuantifier):
    """Reduce a single tensor along given dimensions using logical AND."""
    SYMB: str = "All"
    DEFAULT_REDUCTION: _REDUCTION_TYPE_KEYS = 'min'


class WHERE(TorchOperation):
    """Filter the dimensions in tensor ``a`` by the boolean values given in ``cond``.
    If ``a`` is to be filtered in dimension ``d``, the size of ``b`` must be
    1 in all dimensions except for ``d``, where it has the same dimensionality as ``a``
    (may be e.g. the output of an ANY operation).
    If ``cond`` has dimensionality 1 in all dimensions, either the mask itself
    (``cond==True``) or a zero-shaped tensor (``cond==True``)."""

    SYMB: str = "Where"
    ARITY: int = 2

    def __init__(self, tens_in_key, cond_in_key, dim: Optional[int] = None, **kwargs):
        super().__init__(tens_in_key, cond_in_key, **kwargs)
        self.dim: Optional[int] = int(dim) if dim is not None else None
        """If given, the dimension in which to filter.
        If not given, the dimension is automatically determined from
        the dimensionality of ``cond``."""

    def torch_operation(self, tens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Filter one dimension of ``tens`` by ``cond``."""
        tens_size, cond_size = tens.size(), cond.size()
        dim: Optional[int] = self._validated_not_one_dim(cond_size, tens_size) or 0  # cond all 1 dim -> assume 0th dim
        cond = cond.view(*([1] * dim), -1, *([1] * (len(tens_size) - dim - 1)))  # size (1,..1,filter,1,..1)
        masked: torch.Tensor = torch.masked_select(tens, cond.to(torch.bool))
        return masked.view(*tens_size[:dim], -1, *tens_size[dim + 1:])

    def _validated_not_one_dim(self, cond_size: Sequence[int], tens_size: Sequence[int]
                               ) -> Optional[int]:
        """Validate and return the single positive dimension index in which to filter.
        If no unique dimension can be found (all dim entries 1), ``None`` is returned."""
        # Default: (positive version of) self.dim
        dim: Optional[int] = self.dim + len(tens_size) \
            if self.dim and self.dim < 0 else self.dim
        # region Validation
        # Validate tens_size
        if dim is not None and (dim >= len(tens_size) or dim < 0):
            raise ValueError("Cannot filter tensor of size {} in dimension self.dim={} (0-indexed)!"
                             .format(tens_size, self.dim))
        # Validate cond_size
        elif dim is None and len(tens_size) != len(cond_size):
            raise IndexError(("If the filtering dimension dim isn't specified, the condition (size: {}) "
                              "must have the same size as the input tensor (size: {})!"
                              ).format(cond_size, tens_size))
        non_one_dims = [i for i in range(len(cond_size)) if cond_size[i] != 1]
        if dim is None and len(non_one_dims) > 1:
            raise ValueError("The condition may only feature one dimension of size > 1! Found size {}"
                             .format(cond_size))
        # endregion
        # Choose dim from cond_size if self.dim is None
        dim = dim if dim is not None else \
            (non_one_dims[0] if len(non_one_dims) == 1 else None)

        if np.prod(cond_size) != 1 and np.prod(cond_size) != tens_size[dim]:
            raise ValueError(("Non-matching dimensions to filter tens (size {}) by condition (size {})"
                              " in dim {}").format(tens_size, cond_size, dim))
        return dim


QUANTIFIER_OP_PRECEDENCE: Tuple = (
    ALL,
    ANY,
    WHERE
)
