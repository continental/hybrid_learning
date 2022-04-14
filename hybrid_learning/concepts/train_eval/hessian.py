#  Copyright (c) 2022 Continental Automotive GmbH

##########################################################################
#
#  Courtesy of Felix Dangel: https://github.com/f-dangel/backpack
#
##########################################################################
"""Exact computation of full Hessian using autodiff for calibration."""

# pylint: disable=invalid-name

from typing import Union, Iterable, Optional, Tuple, Sequence, List

import torch
from torch.autograd import grad


def grad_2nd(dtheta: torch.Tensor, params: Sequence[torch.Tensor]
             ) -> torch.Tensor:
    """Calculate the contiguous gradients of ``params`` wrt ``dtheta``
    as 2D matrix."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    fn_grads2: Tuple[torch.Tensor, ...] = grad(dtheta, params,
                                               create_graph=True)
    fn_thetas2 = None
    for fn_grad2 in fn_grads2:
        fn_thetas2 = (
            fn_grad2.contiguous().view(-1)
            if fn_thetas2 is None
            else torch.cat([fn_thetas2, fn_grad2.contiguous().view(-1)])
        )
    return fn_thetas2


def exact_hessian(fn: torch.Tensor,
                  params: Iterable[torch.Tensor],
                  device: Union[str, torch.device] = None,
                  ) -> torch.Tensor:
    r"""Compute all second derivatives of a scalar w.r.t. `parameters`.

    The order of parameters corresponds to a one-dimensional
    vectorization followed by a concatenation of all tensors in
    `parameters`.

    :param fn: Scalar PyTorch function/tensor.
    :param params: iterable object containing all tensors acting as
        variables of ``f``
    :param device: the torch device to use for the hessian
    :return: Hessian of ``f`` with respect to the concatenated version
        of all flattened quantities in ``parameters``

    .. note::
        The parameters in the list are all flattened and concatenated
        into one large vector ``theta``.
        Return the matrix :math:`d^2 E / d \theta^2` with

        .. math::(d^2E / d \theta^2)[i, j] = (d^2E / d \theta[i] d \theta[j]).

    Related work::
    The code is a modified version of
    https://discuss.pytorch.org/t/compute-the-hessian-matrix-of-a-network/15270/3
    """
    params: List[torch.Tensor] = list(params)
    if not all(p.requires_grad for p in params):
        raise ValueError("Ensure all parameters have grad!")
    fn_grads: Tuple[torch.Tensor, ...] = grad(fn, params, create_graph=True)
    # flatten all parameter gradients and concatenate into a vector
    fn_thetas: Optional[torch.Tensor] = None
    for fn_grad in fn_grads:  # TODO: use grad_2nd routine here
        fn_thetas = (
            fn_grad.contiguous().view(-1)
            if fn_thetas is None
            else torch.cat([fn_thetas, fn_grad.contiguous().view(-1)])
        )
    # compute second derivatives
    hessian_dim: int = fn_thetas.size(0)
    hessian = torch.zeros(hessian_dim, hessian_dim, device=device)
    for idx in range(hessian_dim):
        hessian[idx] = grad_2nd(fn_thetas[idx], params)
    return hessian
