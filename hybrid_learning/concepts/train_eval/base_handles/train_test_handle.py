"""Abstract handle for standard training and testing of pytorch models."""

#  Copyright (c) 2022 Continental Automotive GmbH

import abc
import copy
import logging
from typing import Optional, Dict, Any, Callable, Union, Iterable, Tuple, \
    List, Mapping

import pandas as pd
import torch
import torch.nn
import torch.nn.functional
import torch.utils
from torch.optim.optimizer import Optimizer  # pylint: disable=no-name-in-module
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm

from hybrid_learning.datasets import BaseDataset, DataTriple, DatasetSplit
from hybrid_learning.datasets import cross_validation_splits
from hybrid_learning.datasets.transforms import ReduceTuple
from hybrid_learning.datasets.transforms import TupleTransforms
from .early_stopping import EarlyStoppingHandle
from .resettable_optimizer import ResettableOptimizer
from ..callbacks import run_callbacks, CallbackEvents, _validate_templ, LoggingCallback, ProgressBarUpdater
from ..train_eval_funs import train_one_epoch, evaluate, second_stage_train, device_of, loader
from ..kpis import aggregating_kpis

LOGGER = logging.getLogger(__name__)


class TrainEvalHandle(abc.ABC):  # pylint: disable=too-many-instance-attributes
    # pylint: disable=line-too-long
    """Handle for training and evaluation of pytorch models.
    The model base class should be :py:class:`torch.nn.Module`.

    The main functions are :py:meth:`train` and :py:meth:`evaluate`.
    Metrics and loss functions must be given on initialization.
    Training and evaluation results are returned as
    :py:class:`pandas.DataFrame` resp. :py:class:`pandas.Series` with columns
    the metric keys (prefixed according to the mode).
    Modes can be train, test, or validation (see instances of
    :py:class:`~hybrid_learning.datasets.base.DatasetSplit` enum).
    The non-prefixed loss key is saved in :py:const:`LOSS_KEY`.

    For a usage example see
    :py:class:`~hybrid_learning.concepts.models.concept_models.concept_detection.ConceptDetection2DTrainTestHandle`.
    """
    # pylint: enable=line-too-long
    LOSS_KEY = 'loss'
    """Key for the loss evaluation results."""
    NLL_KEY = 'NLL'
    """Key for the proper scoring evaluation function used for second stage
    training. Typically a negative log-likelihood."""

    @classmethod
    def prefix_by(cls, mode: DatasetSplit, text: str) -> str:
        """Prefix ``s`` with the given mode."""
        return "{}_{}".format(mode.value, text)

    @classmethod
    def test_(cls, metric_name: str) -> str:
        """Get name of metric for testing results."""
        return cls.prefix_by(DatasetSplit.TEST, metric_name)

    @classmethod
    def train_(cls, metric_name: str) -> str:
        """Get name of metric for training results."""
        return cls.prefix_by(DatasetSplit.TRAIN, metric_name)

    @classmethod
    def val_(cls, metric_name: str) -> str:
        """Get name of metric for validation results."""
        return cls.prefix_by(DatasetSplit.VAL, metric_name)

    @property
    def kpi_fns(self) -> Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
        """Metric *and* loss functions.
        Nomenclature: :py:attr:`metric_fns` holds all metrics meant for
        evaluation, while :py:attr:`kpi_fns` also encompasses the losses."""
        losses = {self.LOSS_KEY: self.loss_fn} if self.loss_fn else {}
        if self.nll_fn != self.loss_fn and self.nll_fn is not None:
            losses.update({self.NLL_KEY: self.nll_fn})
        return {**losses, **self.metric_fns}

    @property
    def settings(self) -> Dict[str, Any]:
        """The current training settings as dictionary."""
        return dict(
            model=self.model,
            data=self.data,
            device=self.device,
            batch_size=self.batch_size,
            batch_size_val=self.batch_size_val,
            batch_size_hessian=self.batch_size_hessian,
            max_epochs=self.epochs,
            num_workers=self.num_workers,
            loss_fn=self.loss_fn,
            nll_fn=self.nll_fn,
            metric_fns=self.metric_fns,
            early_stopping_handle=self.early_stopping_handle,
            optimizer=self.optimizer,
            callbacks=self.callbacks,
            callback_context=self.callback_context,
            show_progress_bars=self.show_progress_bars,
        )

    def __repr__(self):
        return "{}({})".format(str(self.__class__.__name__),
                               ', '.join(['='.join([str(k), repr(v)])
                                          for k, v in self.settings.items()]))

    def _show_progress_bars_for(self, mode: Union[str, DatasetSplit]):
        """Whether to show the progress for the respective run mode.
        See :py:attr:`show_progress_bars`."""
        mode = self._to_validated_split(mode)
        return self.show_progress_bars == 'always' or mode.value in self.show_progress_bars.split(',')

    def __init__(self,
                 model: torch.nn.Module,
                 data: DataTriple,
                 device: torch.device = None,
                 batch_size: int = None,
                 batch_size_val: int = None,
                 batch_size_hessian: int = None,
                 max_epochs: int = None,
                 num_workers: int = None,
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
                 nll_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
                 metric_fns: Dict[str, Union[aggregating_kpis.AggregatingKpi, Callable[
                     [torch.Tensor, torch.Tensor], torch.Tensor]]] = None,
                 early_stopping_handle: EarlyStoppingHandle = None,
                 optimizer: Callable[..., Optimizer] = None,
                 model_output_transform: TupleTransforms = None,
                 metric_input_transform: TupleTransforms = None,
                 callbacks: List[Mapping[CallbackEvents, Callable]] = None,
                 callback_context: Dict[str, Any] = None,
                 show_progress_bars: Union[bool, str] = True,
                 ):  # pylint: disable=too-many-arguments
        # pylint: disable=line-too-long
        """Init.

        :param model: model to train/eval
        :param device: device on which to load the data and the model parameters
        :param num_workers: number of workers to use for data loading;
            see :py:meth:`loader`; single process loading is used if unset or <2
        :param optimizer: callable that yields a fresh optimizer instance
            when called on the model's trainable parameters
        :param early_stopping_handle: handle for early stopping;
            defaults to default
            :py:class:`~hybrid_learning.concepts.train_eval.base_handles.early_stopping.EarlyStoppingHandle`
            if ``None``; set to ``False`` to disable early stopping;
        :param loss_fn: differentiable metric function to use as loss
        :param nll_fn: Negative log likelihood (or other proper scoring
            function) for use as Laplace approximation
        :param metric_fns: Dictionary of metric functions, each accepting

            - the batch model output tensor, and
            - the batch ground truth tensor

            and yields the value of the specified metric.
        :param model_output_transform: transformation applied to
            the tuples of ``(model output, target)`` before applying loss
            functions or metric functions;
            the functions are wrapped correspondingly;
        :param metric_input_transform: transformation applied to
            the tuples of ``(model output, target)`` before applying metric
            functions only (not the loss and scoring functions), after
            model_output_transform is applied;
            the functions are wrapped correspondingly;
            meant as convenient way to modify metrics simultaneously
        :param callbacks: see :py:attr:`callbacks`
        :param show_progress_bars: see :py:attr:`show_progress_bars`
        """
        # pylint: enable=line-too-long

        # The model to train/eval
        self.model: torch.nn.Module = model
        """The model to work on."""

        # General args
        self.device: torch.device = device or device_of(model)
        """Device to run training and testing on (this is where the data
        loaders are put)."""
        self.model.to(device)

        self.batch_size: int = batch_size or 8
        """Default training batch size."""
        self.batch_size_val: int = batch_size_val or self.batch_size * 2
        """Default validation batch size."""
        self.batch_size_hessian: int = batch_size_hessian or 8
        """Default batch size for calculating the hessian."""
        self.epochs: int = max_epochs or 5
        """Default maximum number of epochs.
        May be reduced by :py:attr:`early_stopping_handle`."""
        self.num_workers: int = num_workers or 0
        """The default number of workers to use for data loading.
        See :py:meth:`hybrid_learning.concepts.train_eval.base_handles.train_test_handle.TrainEvalHandle.loader`.
        """  # pylint: disable=line-too-long
        show_progress_bars: str = show_progress_bars or ''
        self.show_progress_bars: str = show_progress_bars \
            if isinstance(show_progress_bars, str) else 'train'
        """Whether to show progress bars for batch-wise operations.
        Value must be a comma-separated concatenation of run types
        (the values of dataset splits) for which to show progress,
        or ``'always'``."""

        # KPI functions
        self.loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = loss_fn
        """Loss function callable.
        Defaults to a balanced binary cross-entropy assuming on average 1%
        positive px per img.
        Must be wrapped into a tuple to hide the parameters, since these are
        not to be updated."""
        self.nll_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = \
            nll_fn if nll_fn is not None else self.loss_fn
        """Proper scoring function callable used as loss in second stage
        training for Laplace approximation.
        Usually is chosen as negative log-likelihood, defaults to
        :py:attr:`loss_fn`."""
        self.metric_fns: Dict[
            str, Union[aggregating_kpis.AggregatingKpi,
                       Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]] \
            = dict(metric_fns) if metric_fns is not None else {}
        """Dictionary of metric functions to apply for evaluation and logging.
        Each function must have a signature of
        ``(output, target) -> metric_value``. See also :py:attr:`kpi_fns`.
        Keys must not contain :py:const:`LOSS_KEY` or :py:const:`NLL_KEY`."""
        if model_output_transform:
            if self.loss_fn:
                self.loss_fn = ReduceTuple(model_output_transform, self.loss_fn)
            if self.nll_fn:
                self.nll_fn = ReduceTuple(model_output_transform, self.nll_fn)
        if model_output_transform is not None or metric_input_transform is not None:
            transformer = model_output_transform + metric_input_transform
            for name in self.metric_fns:
                self.metric_fns[name] = ReduceTuple(transformer,
                                                    self.metric_fns[name])

        # Additional handles
        self.optimizer: ResettableOptimizer = \
            optimizer or ResettableOptimizer(torch.optim.Adam,
                                             lr=0.01,
                                             weight_decay=0.
                                             # do not add L2 regularization
                                             )
        """Optimizer and learning rate scheduler handle."""
        self.early_stopping_handle: Optional[EarlyStoppingHandle] = \
            early_stopping_handle
        """Handle that is stepped during training and indicates need for
        early stopping.
        To disable early stopping, set :py:attr:`early_stopping_handle` to
        ``None`` resp. specify ``early_stopping_handle=False`` in
        ``__init__`` arguments."""

        # Data loaders
        self.data: DataTriple = data
        """Train, validation, and test data splits to use.
        Must be converted to data loaders before usage."""

        self.callbacks: List[Mapping[CallbackEvents, Callable]] = \
            callbacks or []
        """A dictionary mapping events to a list of callables that are called
        every time the event occurs with the current state.
        Some default logging callbacks are defined.

        For details on available events, see
        :py:class:`hybrid_learning.concepts.train_eval.callbacks.CallbackEvents`.
        After the event, all callbacks for this event are called in order
        with keyword arguments from a
        :py:attr:`hybrid_learning.concepts.train_eval.base_handles.train_test_handle.TrainEvalHandle.callback_context`.
        The base context is dependent on the event and includes e.g. the model,
        and can be extended by specifying the callback context during
        function call or in the default callback context.
        Note that side effects on objects in the callback context (e.g. the
        model) will influence the training.
        Callback application examples:

        - Logging
        - Storing of best n models
        - Results saving, e.g. to tensorboard or sacred log
        - Forced weight normalization

        Callbacks can be
        :py:meth:`added <hybrid_learning.concepts.train_eval.base_handles.train_test_handle.TrainEvalHandle.add_callbacks>`
        and
        :py:meth:`removed <hybrid_learning.concepts.train_eval.base_handles.train_test_handle.TrainEvalHandle.remove_callback>`.
        """
        # always ensure a logger and pbar updater are available:
        if not any([isinstance(cb, LoggingCallback) for cb in self.callbacks]):
            self.add_callbacks([LoggingCallback(LOGGER, log_per_batch=10 if not self.show_progress_bars else False)])
        if not any([isinstance(cb, ProgressBarUpdater)
                    for cb in self.callbacks]):
            self.add_callbacks([ProgressBarUpdater()])

        self.callback_context: Dict[str, Any] = callback_context or {}
        """The default callback context values to use.
        In any training run where context is used, the context can either be
        handed over or it defaults to a copy of this dict."""

    def add_callbacks(
            self, callbacks: Iterable[Mapping[CallbackEvents, Callable]]):
        """Append the given callbacks."""
        self.callbacks += list(callbacks)

    def remove_callback(self, callback: Callable):
        """Remove a single given callback."""
        for i in range(len(self.callbacks)):
            if self.callbacks[i] == callback:
                self.callbacks.pop(i)

    @staticmethod
    def _to_validated_split(mode: Union[str, DatasetSplit]) -> DatasetSplit:
        """Turn a string specifier to a dataset split if necessary."""
        translations = [d for d in DatasetSplit if
                        d.name == mode or d.name.lower() == mode or d.value == mode]
        if len(translations) > 0:
            if not all(t == translations[0] for t in translations):
                raise ValueError(("Ambiguous mode specification {}: found splits {}"
                                  ).format(mode, translations))
            mode = translations[0]

        if mode not in DatasetSplit:
            raise ValueError("Invalid mode {} given. Accepting {}".format(
                mode, [*DatasetSplit, *[d.name for d in DatasetSplit]]))
        return mode

    def loader(self,
               data: Union[torch.utils.data.dataset.Dataset, BaseDataset] = None,
               *, mode: Union[str, DatasetSplit] = None,
               batch_size: Optional[int] = None, shuffle: bool = False,
               device: torch.device = None, model: torch.nn.Module = None,
               num_workers: Optional[int] = None, **_) -> DataLoader:
        """Prepare and return a torch data loader from the dataset
        according to settings.
        For details see :py:meth:`~hybrid_learning.concepts.train_eval.train_eval_funs.loader`.

        :param data: data to obtain loader for; defaults to
            :py:attr:`data` of respective ``mode``
        :param mode: which :py:attr:`data` split to use by default;
            specify as instance of
            :py:class:`~hybrid_learning.datasets.base.DatasetSplit`
            or the name of one;
        :param batch_size: defaults to :py:attr:`batch_size`
        :param device: defaults to :py:attr:`device`
        :param num_workers: defaults to :py:attr:`num_workers`
        """
        if data is None and mode is None:
            raise ValueError("Either data or mode must be given.")
        data = data or self.data[self._to_validated_split(mode)]
        return loader(data, device=device, model=model, shuffle=shuffle,
                      batch_size=(self.batch_size if batch_size is None else batch_size),
                      num_workers=(self.num_workers if num_workers is None else num_workers))

    def reset_optimizer(self, optimizer: ResettableOptimizer = None,
                        device: torch.device = None,
                        model: torch.nn.Module = None):
        """Move model to correct device, init optimizer to parameters of model.
        By default apply to :py:attr:`optimizer`, :py:attr:`device`,
        :py:attr:`model`."""
        optimizer = optimizer or self.optimizer
        model = model or self.model
        device = device or self.device

        model.to(device)
        if optimizer:
            optimizer.init([p for p in model.parameters() if p.requires_grad])
        return optimizer

    def reset_training_handles(self,
                               optimizer: ResettableOptimizer = None,
                               device: torch.device = None,
                               model: torch.nn.Module = None,
                               ) -> None:
        """(Re)set all handles associated with training, and move to ``device``.
        These are: :py:attr:`optimizer`, :py:attr:`early_stopping_handle`,
        and the data loaders.
        The argument values default to the corresponding attributes of this
        instance.
        """
        self.reset_optimizer(optimizer=optimizer, device=device, model=model)
        if self.early_stopping_handle:
            self.early_stopping_handle.reset()

    def reset_kpis(self,
                   kpis: Dict[str, Union[aggregating_kpis.AggregatingKpi,
                                         Callable[[torch.Tensor, torch.Tensor],
                                                  torch.Tensor]]] = None
                   ):  # pylint: disable=no-self-use
        """
        Resets aggregating kpis

        :param kpis: All metric functions and classes
        """
        kpis = kpis or self.kpi_fns
        for metric_fn in [fn for name, fn in kpis.items()
                          if name in aggregating_kpis.filter_aggregating_kpi_keys(kpis)]:
            metric_fn.reset()

    def train(self,
              train_loader: DataLoader = None,
              val_loader: DataLoader = None,
              pbar_desc_templ: str = "Epoch {epoch}/{epochs}",
              callback_context: Dict[str, Any] = None,
              ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train the model according to the specified training parameters.
        Defaults are taken from :py:attr:`settings`.
        To override specify ``custom_args`` (compare arguments to
        :py:attr:`train_val_one_epoch`).

        :param callback_context: see :py:attr:`callback_context`
        :param pbar_desc_templ: template for the progress bar description;
            must accept as substring

            - ``{epoch}``: the current epoch number
            - ``{tot_epoch}``: the total number of epochs
        :param train_loader: see :py:meth:`train_one_epoch`
        :param val_loader: see :py:meth:`evaluate`
        :return: Two `pandas.DataFrame` with history information on

            - *training*: the epoch- and batch-wise loss and KPI results on the
              training data,
              index is a multi-index of ``(epoch, batch)``;
            - *test*: the epoch-wise evaluation results on the test set;
              index is the epoch index;

            columns for both are ``loss`` and KPI names
            (keys of :py:attr:`metric_fns`)
        """

        # region Default values and value checks
        _validate_templ(pbar_desc_templ, ('epoch', 'epochs'))

        callback_context = callback_context or {**self.callback_context}
        # Push model to correct device and reset training handles
        self.reset_training_handles()

        # Shared loaders
        train_loader: DataLoader = \
            train_loader or self.loader(self.data.train, shuffle=True)
        val_loader: DataLoader = \
            val_loader or self.loader(self.data.val,
                                      batch_size=self.batch_size_val)
        # endregion

        history_train = {}
        history_val = {}
        for epoch in range(0, self.epochs):
            callback_context.update(
                history_train=history_train, history_val=history_val,
                epoch=epoch, epochs=self.epochs, model=self.model,
                aggregating_kpis=aggregating_kpis.filter_aggregating_kpi_keys(self.metric_fns))
            # Train and evaluate (with nice progress bar if requested)
            kpis_train, kpis_val = self.train_val_one_epoch(
                train_loader=train_loader, val_loader=val_loader,
                callback_context=callback_context,
                pbar_desc=pbar_desc_templ.format(epoch=epoch + 1,
                                                 epochs=self.epochs), )
            self.optimizer.epoch_end()

            history_train[epoch] = kpis_train
            history_val[epoch] = kpis_val
            callback_context.update(history_train=history_train,
                                    history_val=history_val)
            # Stop early if necessary
            if (self.early_stopping_handle and  # TODO: integrate into callbacks
                    self.early_stopping_handle.step(
                        kpis_val[self.val_(self.LOSS_KEY)])):
                LOGGER.info("Stopped early after %d epochs.", epoch + 1)
                break
            if epoch < (self.epochs - 1):
                run_callbacks(self.callbacks, CallbackEvents.BETWEEN_EPOCHS,
                              callback_context)

        run_callbacks(self.callbacks, CallbackEvents.AFTER_TRAIN,
                      callback_context)

        return pd.concat(history_train), pd.DataFrame(history_val).transpose()

    def cross_validate(self, num_splits: int = 5,
                       train_val_data=None,
                       run_info_templ: str = 'Run {run}/{runs}',
                       callback_context: Dict[str, Any] = None,
                       pbar_desc_templ: str = "epoch {epoch}/{epochs}",
                       ) -> List[Tuple[Dict[str, torch.Tensor],
                                       pd.DataFrame,
                                       pd.DataFrame]]:
        """Record training results for ``num_splits`` distinct val splits.
        The original model state dict is restored after training runs.
        The model must feature a ``reset_parameters()`` method to reinitialize
        between the runs.

        :param run_info_templ: template containing as substring placeholders
            ``{run}`` (the number of the current run) and ``runs``
            (the total number of runs);
            the template is used as prefix for logging and progress bars
        :param num_splits: number of equal-sized, distinct validation splits
            to use
        :param train_val_data: optional given dataset to split into train and
            validation dataset splits; defaults to the ``train_val`` split in
            :py:attr:`data`
        :param callback_context: current callback context to use;
            defaults to copy of :py:attr:`callback_context`
        :param pbar_desc_templ: template for progress bar description
            (prefixed by run information); see :py:meth:`train`
        :return: list of tuples of the form

            | (
            |     final ``state_dict``,
            |     epoch- and batch-wise train history as
                  :py:class:`pandas.DataFrame`,
            |     epoch-wise validation history as :py:class:`pandas.DataFrame`
            | )
        """
        if not ("{run}" in run_info_templ and "{runs}" in run_info_templ):
            raise ValueError("run_info_templ must contain formatting strings "
                             "{run} and {runs} but was " + run_info_templ)
        # region Default values
        if not hasattr(self.model, 'reset_parameters'):
            raise ValueError("model must feature a 'reset_parameters()' method"
                             "for cross-validation")
        train_val_data = train_val_data if train_val_data is not None \
            else self.data.train_val
        callback_context = callback_context or {**self.callback_context}

        # end region

        # Save original model parameters:
        if self.device:
            self.model.to(self.device)
        orig_state_dict = copy.deepcopy(self.model.state_dict())

        splits: List[Tuple[Subset, Subset]] = \
            cross_validation_splits(train_val_data, num_splits=num_splits)
        results: List[Tuple[Dict[str, torch.Tensor],
                            pd.DataFrame, pd.DataFrame]] = []
        for run, (train_data, val_data) in enumerate(splits):
            self.model.reset_parameters()
            callback_context.update(run=run, runs=len(splits))
            history_train, history_val = self.train(
                # Use the created train val split
                train_loader=self.loader(train_data),
                val_loader=self.loader(val_data,
                                       batch_size=self.batch_size_val),
                callback_context=callback_context,
                # TODO: integrate pbar_desc_templ into callback_context
                pbar_desc_templ=", ".join([(
                    run_info_templ.format(run=run + 1, runs=num_splits)),
                    pbar_desc_templ]),
            )
            # collect the state dict
            # properly copy it and move to CPU to not overload GPU:
            # save results:
            results.append((self.detached_state_dict(self.model),
                            history_train, history_val))

        # Restore original model parameters:
        self.model.load_state_dict(orig_state_dict)

        return results

    @staticmethod
    def detached_state_dict(model: torch.nn.Module,
                            device: Union[str, torch.device] = 'cpu'
                            ) -> Dict[str, torch.Tensor]:
        """Return a properly detached copy of the state dict of ``model``
        on ``device``.
        By default, the copy is created on ``cpu`` device to avoid
        overloading the GPU memory."""
        state_dict: Dict[str, torch.Tensor] = model.state_dict()
        for name, param in state_dict.items():
            state_dict[name].data = \
                param.data.detach().to(device).clone().requires_grad_(
                    param.data.requires_grad)
        return state_dict

    def train_val_one_epoch(self,
                            pbar_desc: str = None,
                            callback_context: Dict[str, Any] = None,
                            train_loader: DataLoader = None,
                            val_loader: DataLoader = None,
                            ) -> Tuple[pd.DataFrame, pd.Series]:
        """Train for one epoch, evaluate, and return history and test results.
        This is a wrapper around :py:meth:`train_one_epoch` and
        :py:meth:`evaluate` with nice progress bar printing and logging after
        the epoch.
        History and test results are stored in a :py:class:`pandas.DataFrame`.
        The device used for training is that of the parameters of the used
        model (see :py:func:`~hybrid_learning.concepts.train_eval.train_eval_funs.device_of`).

        :param pbar_desc: leading static description text for the progress bar
        :param callback_context: see :py:attr:`callback_context`
        :param train_loader: see :py:meth:`train_one_epoch`
        :param val_loader: see :py:meth:`evaluate`
        :return: tuple of training history and test results; columns resp.
            index are ``loss`` and the KPI names
            (keys from dict :py:attr:`metric_fns`).

            - :py:class:`pandas.DataFrame`: index are the batch indices,
              items are the results of KPI evaluations of the output on the
              training batch (i.e. before back-propagation step)
            - :py:class:`pandas.Series`: the items are the final evaluations of
              the KPIs on the validation set
        """
        callback_context = callback_context or {**self.callback_context}
        pbar = None if not self._show_progress_bars_for(DatasetSplit.TRAIN) else \
            tqdm(desc=pbar_desc or "Epoch", total=len(train_loader))

        # Training
        callback_context.update(
            pbar=pbar, batches=len(train_loader), train_loader=train_loader)
        train_kpi_vals: pd.DataFrame = self.train_one_epoch(
            train_loader=train_loader,
            callback_context=callback_context,
            pbar_desc=pbar_desc
        )
        if pbar and self._show_progress_bars_for(DatasetSplit.VAL):
            pbar.close()
            callback_context.update(pbar=None)

        # Evaluation
        val_kpi_vals: pd.Series = pd.Series(
            self.evaluate(mode=DatasetSplit.VAL,
                          val_loader=val_loader,
                          callback_context=callback_context,
                          pbar_desc=pbar_desc))

        # Logging
        callback_context.update(
            kpi_train=train_kpi_vals, kpi_val=val_kpi_vals)
        run_callbacks(self.callbacks, CallbackEvents.AFTER_EPOCH,
                      callback_context)
        if pbar:
            pbar.close()

        return train_kpi_vals, val_kpi_vals

    def train_one_epoch(self,
                        train_loader: DataLoader = None,
                        callback_context: Dict[str, Any] = None,
                        pbar_desc: str = "Train progress",
                        ) -> pd.DataFrame:
        """Train for one epoch and return history results as
        :py:class:`pandas.DataFrame`.
        This is a wrapper around
        :py:func:`hybrid_learning.concepts.train_eval.train_eval_funs.train_one_epoch`
        that uses defaults from :py:attr:`settings`.

        :param train_loader: the training loader to use;
            defaults to a shuffled one with training :py:attr:`data` of ``self``
        :param callback_context: see :py:attr:`callback_context`
        :param pbar_desc: leading static description text for the progress bar
            if newly created
        :return: tuple of training history and test results as
            :py:class:`pandas.DataFrame` with:

            :columns:
                ``loss`` and the KPI names
                (keys from dict :py:attr:`metric_fns`),
            :index: the batch indices,
            :items:
                the results of KPI evaluations of the output on the training
                batch (i.e. *before* back-propagation step)
        """
        if self.loss_fn is None:
            raise ValueError("Cannot train without loss_fn (was not given during init)!")
        callback_context = callback_context or {**self.callback_context}
        self.reset_kpis()
        train_loader = train_loader or self.loader(self.data.train,
                                                   shuffle=True)
        # Progressbar handling
        if not callback_context.get('pbar', None) and self._show_progress_bars_for(DatasetSplit.TRAIN):
            callback_context['pbar'] = tqdm(desc=pbar_desc,
                                            total=len(train_loader))

        # Actual training
        return train_one_epoch(
            model=self.model,
            loss_fn=self.loss_fn,
            metric_fns={k: fn for k, fn in self.kpi_fns.items()
                        if k != self.LOSS_KEY},  # add nll_fn to metrics
            train_loader=train_loader,
            optimizer=self.optimizer,
            callbacks=self.callbacks,
            callback_context=callback_context,
            # ensemble_count=self.ensemble_count,  # TODO: support ensembles
            loss_key=self.LOSS_KEY,
        )

    def evaluate(self,
                 mode: Union[str, DatasetSplit] = DatasetSplit.TEST,
                 val_loader: DataLoader = None,
                 prefix: str = None,
                 callback_context: Dict[str, Any] = None,
                 pbar_desc: str = "Progress"
                 ) -> pd.Series:
        """Evaluate the model wrt. :py:attr:`settings`.
        This is a wrapper around
        :py:func:`~hybrid_learning.concepts.train_eval.train_eval_funs.evaluate` which
        uses the defaults given by :py:attr:`settings`.
        Override them by specifying them as ``custom_args``.
        The device used for evaluation is :py:attr:`device` or the one of the
        model.

        :param mode: see :py:attr:`loader`
        :param prefix: see
            :py:func:`~hybrid_learning.concepts.train_eval.train_eval_funs.evaluate`
        :param val_loader: the evaluation dataset loader;
            defaults to one with :py:attr:`data` of respective ``mode``
        :param callback_context: see :py:attr:`callback_context`
        :param pbar_desc: leading static description text for the progress bar
            if newly created
        :return: Dictionary of all KPIs, i.e. of ``loss`` and each metric
            in :py:attr:`metric_fns`;
            format: ``{<KPI-name>: <KPI value as float>}``
        """
        callback_context = callback_context or {**self.callback_context}
        mode: DatasetSplit = self._to_validated_split(mode)
        val_loader = val_loader or self.loader(mode=mode,
                                               batch_size=self.batch_size_val)
        prefix = prefix or mode.value

        # Progressbar handling
        pbar_eval = None
        if self._show_progress_bars_for(mode):
            pbar_eval = tqdm(desc=f'{pbar_desc} ({mode.value})', total=len(val_loader))
            callback_context.update(pbar_eval=pbar_eval)

        self.reset_kpis()
        results: pd.Series = evaluate(
            model=self.model,
            kpi_fns=self.kpi_fns,
            val_loader=val_loader,
            prefix=prefix,
            callbacks=self.callbacks,
            callback_context=callback_context,
            # ensemble_count = self.ensemble_count,  # TODO: support ensembles
        )

        if pbar_eval:
            pbar_eval.close()

        return results

    def second_stage_train(self,
                           callback_context: Dict[str, Any] = None,
                           ) -> pd.Series:
        """Do a second stage training for calibration using Laplace
        approximation. This is a wrapper around
        :py:func:`hybrid_learning.concepts.train_eval.train_eval_funs.second_stage_train`
        that uses defaults from :py:attr:`settings`.
        Before and after the second stage training process one epoch
        on the test and the validation set to enable logging of metrics
        for comparison.

        .. note::
            Evaluation runs on validation and test split are conducted
            before (epoch 0) and after (epoch 1) the second stage training.

        :param callback_context: see :py:attr:`callback_context`.
        :return: py:class:`pandas.Series` with the final evaluation results
            on validation and test splits of :py:attr:`data`
        """

        callback_context = callback_context or {**self.callback_context}
        # region Run callbacks for validation results BEFORE second stage train
        callback_context.update(
            epoch=0, epochs=self.epochs,  # log before status as epoch 0
            model=self.model)
        val_kpi_vals: pd.Series = self.evaluate(
            mode=DatasetSplit.VAL,
            callback_context=callback_context,
            prefix=f"{DatasetSplit.VAL.value}Cal")

        test_kpi_vals: pd.Series = self.evaluate(
            mode=DatasetSplit.TEST,
            callback_context=callback_context,
            prefix=f"{DatasetSplit.TEST.value}Cal")

        val_kpi_vals = pd.concat([val_kpi_vals, test_kpi_vals])
        callback_context.update(kpi_val=val_kpi_vals)
        # endregion

        second_stage_train(
            model=self.model,
            nll_fn=self.nll_fn,
            train_loader=self.loader(
                self.data.train,
                batch_size=self.batch_size_hessian,
                device=device_of(self.model),
                num_workers=0),
            val_loader=self.loader(
                self.data.val,
                batch_size=self.batch_size_val,
                device=device_of(self.model),
                num_workers=0)
        )

        # region: Run callbacks for validation results AFTER second stage train
        callback_context.update(
            epoch=1, epochs=self.epochs,  # log after status as epoch 1
            model=self.model)
        val_kpi_vals: pd.Series = self.evaluate(
            mode=DatasetSplit.VAL,
            callback_context=callback_context,
            prefix=f"{DatasetSplit.VAL.value}Cal")

        test_kpi_vals: pd.Series = self.evaluate(
            mode=DatasetSplit.TEST,
            callback_context=callback_context,
            prefix=f"{DatasetSplit.TEST.value}Cal")

        val_kpi_vals = pd.concat([val_kpi_vals, test_kpi_vals])
        callback_context.update(kpi_val=val_kpi_vals)

        run_callbacks(self.callbacks, CallbackEvents.AFTER_SECOND_STAGE_TRAIN,
                      callback_context=callback_context)
        # endregion

        return val_kpi_vals
