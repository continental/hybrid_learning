"""Basic methods for training and evaluation of models with callback calls."""

#  Copyright (c) 2022 Continental Automotive GmbH

# Paper implementations using variable names from the paper:
# pylint: disable=invalid-name,not-callable,too-many-arguments

import itertools
import logging
from typing import Dict, Union, Any, Mapping, List, Iterable, TYPE_CHECKING, \
    Sequence, Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .callbacks import CallbackEvents, run_callbacks
from .hessian import exact_hessian
from .kpis.aggregating_kpis import filter_aggregating_kpi_keys
from ...datasets import BaseDataset

if TYPE_CHECKING:
    from .base_handles import ResettableOptimizer

LOGGER = logging.getLogger()


def device_of(model: torch.nn.Module) -> torch.device:
    """Return the device of the given pytorch model.
    Distributed models are not supported."""
    device = next(model.parameters()).device \
        if len(list(model.parameters())) > 0 else 'cpu'
    return device


def loader(data: Union[torch.utils.data.dataset.Dataset,
                       BaseDataset] = None,
           *, batch_size: Optional[int] = None, shuffle: bool = False,
           device: torch.device = None, model: torch.nn.Module = None,
           num_workers: Optional[int] = None):
    """Prepare and return a torch data loader with device-dependent
    multi-processing settings.

    .. note::
        If using ``num_workers > 1`` the dataset must be pickleable,
        including caches and all transformations.
        This is usually no problem except for datasets or transformations
        holding references to models:
        These will not be pickleable once the forward method was called
        without disabling the gradients
        (``with torch.set_grad_enabled(False): ...``).
        Make sure this is not the case.

    :param data: data to obtain loader for
    :param batch_size: the batch size to apply
    :param shuffle: Whether the loader should shuffle the data or not;
        e.g. shuffle training data and do not shuffle evaluation data
    :param device: the desired device to work on
        (determines whether to pin memory); defaults to cuda if it is available
    :param model: if device is not given, the device is determined from
        ``model`` if this is given
    :param num_workers: if a positive integer, multi-process data loading
        is done with this amount of worker processes
        (otherwise a blocking single-process is used)
    :return: a data loader for the given data
    """
    device: Union[str, torch.device] = device or (device_of(model) if model is not None
                                                  else ('cuda' if torch.cuda.is_available() else 'cpu'))
    device: Optional[torch.device] = torch.device(device)
    if device.type == 'cuda' and device.index:
        LOGGER.warning(
            "You tried to run training with a cuda device (%s) other "
            "than the default one (index 0). Currently, the default "
            "collate_fn of the DataLoader does always move tensors to "
            "the default cuda device, which may lead to a RuntimeError "
            "if the tensors produced by the dataset are on the "
            "non-default cuda device."
            "Consider instead exporting"
            "CUDA_VISIBLE_DEVICES=<your device index>"
            "on the command line before calling the script and set"
            "the device simply to the default 'cuda' one.",
            device
        )
    worker_kwargs: Dict[str, Any] = {}
    if num_workers and num_workers > 0:
        worker_kwargs.update(num_workers=num_workers)
        # Cuda tensors require specific multiprocessing context:
        if device.type == 'cuda':
            worker_kwargs.update(dict(
                # pin_memory=True,  # only works for CPU-tensor datasets
                persistent_workers=True,
                multiprocessing_context='spawn',
            ))

    return DataLoader(data, shuffle=shuffle,
                      **worker_kwargs,
                      batch_size=batch_size)


def _average_disagreement(outputs: torch.Tensor) -> torch.Tensor:
    """Collect a list of boolean pairwise disagreements of the outputs.
    Disagreement here means inequality after binarizing with a threshold of 0.5.
    """
    pairwise_disagreement = []
    for pair in list(itertools.combinations(range(len(outputs)), 2)):
        probs_1 = outputs[pair[0]]
        probs_2 = outputs[pair[1]]

        preds_1 = probs_1 > 0.5
        preds_2 = probs_2 > 0.5
        pairwise_disagreement.append(torch.not_equal(preds_1, preds_2))

    return torch.mean(torch.stack(pairwise_disagreement).float())


def _to_ens_loss_fn(loss_fn: Callable[[torch.Tensor, torch.Tensor],
                                      torch.Tensor]
                    ) -> Callable[[Iterable[torch.Tensor], torch.Tensor],
                                  torch.Tensor]:
    """Decorate a loss function on one output to work on an ensemble of
    outputs."""

    def ens_loss_fn(outs: Iterable[torch.Tensor], targ: torch.Tensor) -> torch.Tensor:
        """Apply a loss to all outputs for a common target."""
        return torch.sum(
            torch.stack([loss_fn(output, targ) for output in outs]))

    return ens_loss_fn


def train_one_epoch(
        model: torch.nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        metric_fns: Dict[str, Callable[[torch.Tensor, torch.Tensor],
                                       torch.Tensor]],
        train_loader: DataLoader,
        optimizer: Union[Optimizer, 'ResettableOptimizer'],
        callbacks: List[Mapping[CallbackEvents, Callable]] = None,
        callback_context: Dict[str, Any] = None,
        ensemble_count: int = None,
        prefix: str = 'train',
        loss_key: str = 'loss'
) -> pd.DataFrame:
    """Train for one epoch, evaluate, and return history and test results.
    History and test results are stored in a :py:class:`pandas.DataFrame`.
    The device used is the one the model lies on. Distributed models are
    not supported.

    Additional features:

    - ``ensemble_count``: if set and >0, it is assumed the model is an
      ensemble and returns a stack of result tensors (stacked in dim 0).
      The additional metric ``'disag@0.5'`` (average disagreement of
      ensemble outputs if binarized at a threshold of 0.5) is calculated
      and added to the results.

    :meta public:

    :param model: model to train
    :param loss_fn: function that calculates the optimization objective
        value
    :param metric_fns: further KPI functions to gather training stats
    :param train_loader: train data loader
    :param optimizer: optimizer to use for weight update steps
        initialized with model's weights
    :param callbacks: callbacks to feed with callback context after each batch,
        and before and after training epoch
    :param callback_context: dict with any additional context to be handed
        over to the callbacks as keyword arguments
    :param ensemble_count: if set to a value >0 treat the output
        of the model as ``ensemble_count`` outputs stacked in dim 0
    :param prefix: prefix to prepend to KPI names for the final
        :py:class:`pandas.Series` naming
    :param loss_key: key the loss should have in the output
    :return: training history as :py:class:`pandas.DataFrame` with

        :columns:
            ``loss`` and the KPI names (keys from dict ``metric_fns``),
        :index: the batch indices,
        :items:
            the results of KPI evaluations of the output on the training
            batch (i.e. *before* back-propagation step)
    """
    if len(train_loader) == 0:
        raise ValueError("Empty training loader (no batches)! "
                         "Batch size too large?")
    train_ = (lambda x: f'{prefix}_{x}')
    ensemble_count: int = ensemble_count if ensemble_count is not None \
        else getattr(model, "ensemble_count", 0)
    if ensemble_count > 0:
        loss_fn = _to_ens_loss_fn(loss_fn)

    # very simple way to find out the correct device for
    # non-distributed models:
    device = device_of(model)
    for kpi_fn in [*list(metric_fns.values()), loss_fn]:
        getattr(kpi_fn, 'to', lambda *x: None)(device)
    # Training
    model.train()
    train_kpi_vals: pd.DataFrame = pd.DataFrame(
        columns=[train_(s) for s in
                 [loss_key,
                  *(list(metric_fns.keys())
                    + (['disag@0.5'] if ensemble_count > 1 else []))]],
        index=list(range(len(train_loader))))

    callback_context = callback_context or {}
    callbacks = callbacks or []
    callback_context.update(
        model=model, device=device, train_loader=train_loader, run_type=prefix,
        batches=len(train_loader),
        aggregating_kpis=filter_aggregating_kpi_keys(metric_fns))
    run_callbacks(callbacks, CallbackEvents.BEFORE_EPOCH_TRAIN,
                  callback_context)
    for batch_idx, (data, target) in enumerate(train_loader):
        data: torch.Tensor = data.to(device)
        target: torch.Tensor = target.to(device)

        # Reset optimizer gradients
        optimizer.zero_grad()

        # Calculate & collect loss tensor & metric(s) values
        kpi_tensors: Dict[str, Union[float, None, Any]] = {}
        outputs: torch.Tensor = model(data)
        loss_tensor = loss_fn(outputs, target)
        kpi_tensors[train_(loss_key)] = loss_tensor
        output = outputs

        if ensemble_count > 0:
            if output.shape[0] > 1:
                kpi_tensors[train_('disag@0.5')] = \
                    _average_disagreement(output)
            output = output.mean(dim=0)

        kpi_tensors.update({train_(m): fn(output, target)
                            for m, fn in metric_fns.items()})
        train_kpi_vals.loc[batch_idx] = {
            key: value.item() if isinstance(value, torch.Tensor) else value
            for key, value in kpi_tensors.items()}

        # Do back-propagation and apply to weight
        loss_tensor.backward()
        optimizer.step()

        # Logging
        callback_context.update(batch=batch_idx,
                                kpi_train=train_kpi_vals, )
        run_callbacks(callbacks, CallbackEvents.AFTER_BATCH_TRAIN,
                      callback_context)

    # Obtain values for aggregating kpis
    batch_idx = len(train_loader) - 1
    for kpi_name in filter_aggregating_kpi_keys(metric_fns):
        value = metric_fns[kpi_name].value()
        train_kpi_vals.loc[batch_idx][train_(kpi_name)] = \
            value.item() if isinstance(value, torch.Tensor) else value

    run_callbacks(callbacks, CallbackEvents.AFTER_EPOCH_TRAIN,
                  callback_context)

    return train_kpi_vals.infer_objects()


def evaluate(model: torch.nn.Module,
             kpi_fns: Dict[str, Callable],
             val_loader: DataLoader,
             prefix: str = 'val',
             callbacks: List[Mapping[CallbackEvents, Callable]] = None,
             callback_context: Dict[str, Any] = None,
             ensemble_count: int = None,
             ) -> pd.Series:
    """Evaluate the model wrt loss and ``metric_fns`` on the test data.
    The reduction method for the KPI values is ``mean``.
    The device used is the one of the model lies on (see
    :py:func:`device_of`). Distributed models are not supported.

    :meta public:
    :param model: the model to evaluate;
        must return a single tensor or sequence of tensors on call to forward
    :param kpi_fns: dictionary with KPI IDs and evaluation functions for
        the KPIs to evaluate
    :param val_loader: data loader with data to evaluate on
    :param prefix: prefix to prepend to KPI names for the final
        :py:class:`pandas.Series` naming
    :param callbacks: callbacks to feed with callback context after each batch
        and after finishing evaluation
    :param callback_context: dict with any additional context to be handed
        over to the callbacks as keyword arguments
    :param ensemble_count: if set to a value >0 treat the output
        of the model as ``ensemble_count`` outputs stacked in dim 0
    :return: Dictionary of all KPI values in the format:
        ``{<KPI-name>: <KPI value as float>}``
    """
    ensemble_count: int = ensemble_count if ensemble_count is not None \
        else getattr(model, "ensemble_count", 0)

    # very simple way to find out the correct device for
    # non-distributed models:
    prefix_ = (lambda x: f'{prefix}_{x}')
    device = device_of(model)
    _ = [getattr(kpi_fn, 'to', lambda *x: None)(device)
         for kpi_fn in kpi_fns.values()]

    # Value check and defaults
    assert len(val_loader) != 0, ("Empty evaluation data loader (no "
                                  "batches)! Batch size too large?")

    # Combine loss and metrics as general KPI measures
    eval_kpi_vals: Dict[str, float] = {prefix_(m): 0.
                                       for m in kpi_fns.keys()}

    aggregating_kpis: Sequence[str] = filter_aggregating_kpi_keys(kpi_fns)

    model.eval()
    num_batches: int = len(val_loader)
    callback_context = callback_context or {}
    callbacks = callbacks or []
    callback_context.update(
        model=model, device=device, val_loader=val_loader, run_type=prefix,
        batches=num_batches, aggregating_kpis=aggregating_kpis)
    with torch.no_grad():
        # Gather KPI values from all batches
        for batch_idx, (data, target) in enumerate(val_loader):
            data: torch.Tensor = data.to(device)
            target: torch.Tensor = target.to(device)
            outputs: torch.Tensor = model(data)

            if ensemble_count > 0:
                if outputs.shape[0] > 1:
                    eval_kpi_vals[prefix_('disag@0.5')] = \
                        eval_kpi_vals.get(prefix_('disag@0.5'), 0) + \
                        _average_disagreement(outputs).item()
                output = outputs.mean(dim=0)
            else:
                output = outputs

            for kpi_name in aggregating_kpis:
                kpi_fns[kpi_name](output, target)
            for kpi_name in set(kpi_fns.keys()) - set(aggregating_kpis):
                eval_kpi_vals[prefix_(kpi_name)] += \
                    kpi_fns[kpi_name](output, target).item()

            callback_context.update(kpi_val=eval_kpi_vals, batch=batch_idx)
            run_callbacks(callbacks, CallbackEvents.AFTER_BATCH_EVAL,
                          callback_context)

        # Aggregate KPI values of all batches via mean.
        # Obtain final values for aggregating kpis.
        for kpi_name, kpi_fn in kpi_fns.items():
            if kpi_name not in aggregating_kpis:
                eval_kpi_vals[prefix_(kpi_name)] /= num_batches
            else:
                # noinspection PyUnresolvedReferences
                value = kpi_fn.value()
                eval_kpi_vals[prefix_(kpi_name)] = value.item() \
                    if isinstance(value, torch.Tensor) else value

    eval_kpi_vals: pd.Series = pd.Series(eval_kpi_vals)
    callback_context.update(kpi_val=eval_kpi_vals)
    run_callbacks(callbacks, CallbackEvents.AFTER_EPOCH_EVAL,
                  callback_context)

    return eval_kpi_vals.infer_objects()


def predict_laplace(model: torch.nn.Module,
                    data: torch.Tensor,
                    device: Union[torch.device, str] = None,
                    var0: float = None):
    """
    Performs prediction with probit approximation of the Bayesian posterior.
    See [MacKay1992]_ for details.

    Assumptions on the layers to process
    (currently:: layers are assumed to be named ``"concept_layer_{i}"``)

    - ALl layers are convolutional layers with attributes ``kernel`` and
      ``bias``.
    - Hessian available for each layer as attribute ``layer.hessian``
    - If ``var0`` is unset, variance is available for each layer as
      attribute ``layer.var0``

    .. note::
        Implementation note:: Currently, the kernel is flattened out for the
        calculation, leading to a considerable consumption of memory.
        Make sure enough memory is available.

    .. [MacKay1992] MacKay, David JC.
        "The evidence framework applied to classification networks."
        Neural computation 4.5 (1992): 720-736.
        https://github.com/wiseodd/last_layer_laplace/blob/master/paper/laplace/llla_binary.py

    :param model: The pytorch model
    :param data: The input data to predict for
    :param device: The device to use for computations; defaults to model device
    :param var0: The var0 constant. If not given,
        uses ``layer.var0``
    :return: The predictions
    """
    device = device or device_of(model)
    outputs: List[torch.Tensor] = []
    # TODO: hand-over the layers to process
    for concept_layer_idx in range(model.ensemble_count):

        concept_layer = getattr(model, f"concept_layer_{concept_layer_idx}")

        # region: Value checks
        assert hasattr(concept_layer, 'hessian'), \
            ("Make sure the layer {} has registered a buffer 'hessian'"
             ).format(concept_layer)
        assert (hasattr(concept_layer, 'bias') and
                hasattr(concept_layer, 'weight') and
                hasattr(concept_layer, 'kernel_size')), \
            "Given layer {} no Conv layer?".format(concept_layer)
        # endregion

        if var0 is None:
            assert hasattr(concept_layer, 'var0')
            var0 = concept_layer.var0

        if concept_layer.bias:
            mu = torch.cat(
                [concept_layer.bias, concept_layer.weight.flatten()])
        else:
            mu = concept_layer.weight.flatten()

        output_dim = [0, 0]
        output_dim[0] = int(
            (data.shape[2]
             - concept_layer.kernel_size[0] + 2 * concept_layer.padding[0])
            / concept_layer.stride[0]
        ) + 1
        output_dim[1] = int(
            (data.shape[3]
             - concept_layer.kernel_size[1] + 2 * concept_layer.padding[1])
            / concept_layer.stride[1]
        ) + 1

        # https://discuss.pytorch.org/t/manual-implementation-of-unrolled-3d-convolutions/91021/3
        unfolded_input_tensor = torch.nn.functional.unfold(
            data,
            kernel_size=concept_layer.kernel_size,
            padding=concept_layer.padding,
            stride=concept_layer.stride)

        if concept_layer.bias:
            phi = torch.cat([torch.ones(unfolded_input_tensor.shape[0], 1,
                                        unfolded_input_tensor.shape[2],
                                        device=device),
                             unfolded_input_tensor], dim=1)
        else:
            phi = unfolded_input_tensor

        output = mu.view(1, -1) @ phi

        S = torch.inverse(
            concept_layer.hessian[concept_layer_idx]
            + (1 / var0 * torch.eye(mu.shape[0], device=device)))
        v = torch.stack(
            [torch.diag(phi[:, :, i] @ S @ phi[:, :, i].t())
             for i in range(phi.shape[-1])], -1)
        output = output * (1 + np.pi * v.view(output.shape) / 8) ** (-1 / 2)

        output = output.view(data.shape[0], 1, output_dim[0], output_dim[1])
        outputs.append(output)

    output = torch.stack(outputs)
    return output


def second_stage_train(model: torch.nn.Module,
                       nll_fn: Callable[[torch.Tensor, torch.Tensor],
                                        torch.Tensor],
                       train_loader: DataLoader,
                       val_loader: DataLoader,
                       ):
    """Evaluate the model wrt loss and ``metric_fns`` on the test data.
    The reduction method for the KPI values is ``mean``.
    The device used is the one of the model lies on (see
    :py:func:`device_of`). Distributed models are not supported.
    This method is used for different post training optimization procedures.

    :meta public:

    :param model: the model to evaluate
    :param nll_fn: The loss function, usually negative log likelihood
        (cross entropy) or a different, proper scoring function
    :param train_loader: data loader with train data
    :param val_loader: data loader with validation data
    :return: Dictionary of all KPI values in the format:
        ``{<KPI-name>: <KPI value as float>}``
    """
    assert nll_fn is not None
    device = device_of(model)

    if model.use_laplace:
        # Compute the laplace approximation of the posterior
        mu: List[torch.Tensor] = _initialize_hessian(model)

        for data, target in tqdm(train_loader, mininterval=30):
            data: torch.Tensor = data.to(device)
            target: torch.Tensor = target.to(device)

            _approximate_hessian(model, data, target, mu, nll_fn)

        with torch.no_grad():

            _optimize_posterior_variance(model, val_loader, nll_fn)


def _initialize_hessian(model: torch.nn.Module) -> List[torch.Tensor]:
    """
    Initializes/resets the Hessians and the mean vector of the last layer
    (consistent of weights and bias).

    :return: A list of mean tensors
    """
    mu = []
    # TODO: hand-over the layers to process
    for concept_layer_idx in range(model.ensemble_count):
        concept_layer = getattr(model, f"concept_layer_{concept_layer_idx}")
        if concept_layer.bias:
            mu.append(torch.cat(
                [concept_layer.bias, concept_layer.weight.flatten()]))
        else:
            mu.append(concept_layer.weight.flatten())
        concept_layer.hessian[:, :] = 0
        concept_layer.var0[:] = 0
    if not torch.cuda.is_available():
        LOGGER.info(f'Starting Laplace '
                    f'(Hessian {model.concept_layer_0.hessian.shape})')
    else:
        torch.cuda.empty_cache()
        LOGGER.info(
            f'Starting Laplace '
            f'(Hessian {model.concept_layer_0.hessian.shape}): '
            f'GPU Memory {torch.cuda.memory_allocated() / 1024 ** 2}/'
            f'{torch.cuda.memory_reserved() / 1024 ** 2} allocated/reserved')
    return mu


def _approximate_hessian(model: torch.nn.Module,
                         x_data: torch.Tensor,
                         y_data: torch.Tensor,
                         mu: List[torch.Tensor],
                         nll_fn: Callable[[torch.Tensor, torch.Tensor],
                                          torch.Tensor]):
    """
    Step for iteratively approximating the Hessian by computing a running
    mean of the hessian.

    :param model: The concept model with initialized Hessian
    :param x_data: The input data (of the current batch)
    :param y_data: The target data (of the current batch)
    :param mu: The mean vectors
        (from :py:meth:`~TrainEvalHandle._initialize_hessian`)
    :param nll_fn: The loss function, usually negative log likelihood
        (cross entropy) or a different, proper scoring function
    """
    rho = 0.95
    device = device_of(model)
    losses: List[torch.Tensor] = []
    # TODO: hand-over layers to process
    for concept_layer_idx in range(model.ensemble_count):

        concept_layer = getattr(model, f"concept_layer_{concept_layer_idx}")
        output_dim = [0, 0]
        output_dim[0] = int((x_data.shape[2]
                             - concept_layer.kernel_size[0]
                             + 2 * concept_layer.padding[0])
                            / concept_layer.stride[0]) + 1
        output_dim[1] = int((x_data.shape[3]
                             - concept_layer.kernel_size[1]
                             + 2 * concept_layer.padding[1])
                            / concept_layer.stride[1]) + 1

        # https://discuss.pytorch.org/t/manual-implementation-of-unrolled-3d-convolutions/91021/3
        unfolded_input_tensor = torch.nn.functional.unfold(
            x_data,
            kernel_size=concept_layer.kernel_size,
            padding=concept_layer.padding,
            stride=concept_layer.stride)
        if concept_layer.bias:
            phi = torch.cat([torch.ones(unfolded_input_tensor.shape[0], 1,
                                        unfolded_input_tensor.shape[2],
                                        device=device),
                             unfolded_input_tensor], dim=1)
        else:
            phi = unfolded_input_tensor
        output = mu[concept_layer_idx].view(1, -1) @ phi

        nll = nll_fn(
            output.view(x_data.shape[0], 1, output_dim[0], output_dim[1]),
            y_data)
        losses.append(nll)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Sum reduction instead mean, hence len(data)
        H_ = exact_hessian(nll * len(x_data), [mu[concept_layer_idx]],
                           device=device).detach()
        concept_layer.hessian[:, :] = \
            rho * concept_layer.hessian + (1 - rho) * H_


def _optimize_posterior_variance(
        model: torch.nn.Module,
        val_loader: DataLoader,
        nll_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
    """
    Determines the posterior variance of the Laplace approximation.
    Has to be called after :py:meth:`~TrainEvalHandle._approximate_hessian`.
    """
    device = device_of(model)
    # Gridsearch over precision/variance values of the posterior
    pbar = tqdm(torch.tensor(np.logspace(-4, 2, 70)).to(device),
                mininterval=30)

    nlls: List[torch.Tensor] = []
    val: List[torch.Tensor] = []
    for var0 in pbar:
        nll = torch.zeros(1, device=device)
        for data, target in val_loader:
            data: torch.Tensor = data.to(device)
            target: torch.Tensor = target.to(device)
            output = predict_laplace(model, data, device, var0)
            nll += nll_fn(output.mean(0), target)
        val.append(var0.item())
        nlls.append((nll / len(val_loader)).item())
        pbar.set_postfix(
            {'v': var0.item(), 'nll': (nll / len(val_loader)).item(),
             'best_v': val[int(np.argmin(nlls))], 'best_nll': np.min(nlls)},
            refresh=False)
        if np.argmin(nlls) < len(nlls) - 5:
            LOGGER.info(f'Posterior variance tuning stopped early after'
                        f' {len(nlls)} tests.')
            pbar.close()
            break

    best = np.argmin(nlls)
    LOGGER.info(f'Best var0: {val[int(best)]} @ {nlls[int(best)]}')
    for i in range(model.ensemble_count):
        concept_layer = getattr(model, f'concept_layer_{i}')
        concept_layer.var0[:] = val[int(best)]
