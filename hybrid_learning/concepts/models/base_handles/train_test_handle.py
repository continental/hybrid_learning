"""Abstract handle for standard training and testing of pytorch models."""

#  Copyright (c) 2020 Continental Automotive GmbH

# pylint: enable=no-name-in-module

import abc
import copy
import logging
from typing import Optional, Dict, Any, Callable, Union, Iterable, Tuple, \
    List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.utils
# pylint: disable=no-name-in-module
from torch.optim.optimizer import Optimizer
# pylint: enable=no-name-in-module
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import Subset
from tqdm import tqdm

from hybrid_learning.datasets import BaseDataset, DataTriple, DatasetSplit
from hybrid_learning.datasets import cross_validation_splits
from hybrid_learning.datasets.transforms import ReduceTuple
from hybrid_learning.datasets.transforms import TupleTransforms
from .early_stopping import EarlyStoppingHandle
from .resettable_optimizer import ResettableOptimizer
from ...kpis import BalancedBCELoss, SetIoU, IoU

LOGGER = logging.getLogger(__name__)


def _validate_templ(templ: str, var_names: Iterable[str]):
    """Raise if the template is missing a variable name."""
    format_vars = ['{' + str(var_name) + '}' for var_name in var_names]
    if any([s not in templ for s in format_vars]):
        raise ValueError(("string template must contain all formatting strings "
                          "{} but was {}").format(format_vars, templ))


class TrainEvalHandle(abc.ABC):
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
    :py:class:`~hybrid_learning.concepts.models.concept_detection.ConceptDetection2DTrainTestHandle`.
    """
    # pylint: enable=line-too-long
    LOSS_KEY = 'loss'
    """Key for the loss evaluation results."""

    @classmethod
    def prefix_by(cls, mode: DatasetSplit, string: str) -> str:
        """Prefix ``s`` with the given mode."""
        return "{}_{}".format(mode.value, string)

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
    def settings(self) -> Dict[str, Any]:
        """The current training settings as dictionary."""
        return dict(
            model=self.model,
            data=self.data,
            device=self.device,
            batch_size=self.batch_size,
            max_epochs=self.epochs,
            loss_fn=self.loss_fn,
            metric_fns=self.metric_fns,
            early_stopping_handle=self.early_stopping_handle,
            optimizer_creator=self.optimizer,
        )

    def __init__(self,
                 model: torch.nn.Module,
                 data: DataTriple,
                 device: torch.device = None,
                 batch_size: int = None,
                 max_epochs: int = None,
                 loss_fn: Callable[
                     [torch.Tensor, torch.Tensor], torch.Tensor] = None,
                 metric_fns: Dict[str, Callable[
                     [torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 early_stopping_handle: EarlyStoppingHandle = None,
                 optimizer: Callable[..., Optimizer] = None,
                 model_output_transform: TupleTransforms = None
                 ):  # pylint: disable=too-many-arguments
        # pylint: disable=line-too-long
        """Init.

        :param model: model to train/eval
        :param device: device on which to load the data and the model parameters
        :param optimizer: callable that yields a fresh optimizer instance
            when called on the model's trainable parameters
        :param early_stopping_handle: handle for early stopping;
            defaults to default
            :py:class:`~hybrid_learning.concepts.models.base_handles.early_stopping.EarlyStoppingHandle`
            if ``None``; set to ``False`` to disable early stopping;
        :param loss_fn: differentiable metric function to use as loss
        :param metric_fns: Dictionary of metric functions, each accepting

            - the batch model output tensor, and
            - the batch ground truth tensor

            and yields the value of the specified metric.
        :param model_output_transform: transformation applied to the tuples of
            ``(model output, target)`` before applying loss functions or
            metric functions;
            the functions are wrapped correspondingly
        """
        # pylint: enable=line-too-long

        # The model to train/eval
        self.model: torch.nn.Module = model
        """The model to work on."""

        # General args
        self.device: torch.device = device or torch.device("cpu")
        """Device to run training and testing on (this is where the data
        loaders are put)."""
        self.model.to(device)
        self.batch_size: int = batch_size or 8
        """Default batch size."""
        self.epochs: int = max_epochs or 5
        """Default maximum number of epochs.
        May be reduced by :py:attr:`early_stopping_handle`."""

        # KPI functions
        self.loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = \
            loss_fn if loss_fn is not None else BalancedBCELoss(
                factor_pos_class=0.99)
        """Loss function callable.
        Defaults to a balanced binary cross-entropy assuming on average 1%
        positive px per img.
        Must be wrapped into a tuple to hide the parameters, since these are
        not to be updated."""
        self.metric_fns: Dict[
            str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = \
            metric_fns or {"set_iou": SetIoU(), "mean_iou": IoU()}
        """Dictionary of metric functions to apply for evaluation and logging.
        Each function must have a signature of
        ``(output, target) -> metric_value``."""
        if model_output_transform is not None:
            self.loss_fn = ReduceTuple(model_output_transform, self.loss_fn)
            for name in self.metric_fns:
                self.metric_fns[name] = ReduceTuple(model_output_transform,
                                                    self.metric_fns[name])

        # Additional handles
        self.optimizer: ResettableOptimizer = \
            optimizer or ResettableOptimizer(torch.optim.Adam,
                                             lr=0.01,
                                             weight_decay=0.
                                             # do not add L2 regularization
                                             )
        """Optimizer and learning rate scheduler handle."""
        self.early_stopping_handle: Optional[EarlyStoppingHandle] = None \
            if early_stopping_handle is False else (
                early_stopping_handle or EarlyStoppingHandle())
        """Handle that is stepped during training and indicates need for
        early stopping.
        To disable early stopping, set :py:attr:`early_stopping_handle` to
        ``None`` resp. specify ``early_stopping_handle=False`` in
        ``__init__`` arguments."""

        # Data loaders
        self.data: DataTriple = data
        """Train, validation, and test data splits to use.
        Must be converted to data loaders before usage."""

    def loader(self, data: Union[torch.utils.data.dataset.Dataset, BaseDataset],
               batch_size: Optional[int] = None, shuffle: bool = False,
               device: torch.device = None) -> DataLoader:
        """Prepare and return a torch data loader from the dataset
        according to settings.
        The settings include the device and batch size.

        :param data: data to obtain loader for
        :param batch_size: the batch size to apply; defaults to
            :py:attr:`batch_size`
        :param shuffle: Whether the loader should shuffle the data or not;
            e.g. shuffle training data and do not shuffle evaluation data
        :param device: the desired device to work on
            (determines whether to pin memory); currently unused
        :return: a data loader for the given data
        """
        # It looks like pinning and threading causes issues, so skip this:
        # noinspection PyUnusedLocal
        device = device or self.device
        # loader_kwargs = {}
        # dict(num_workers=1, pin_memory=True) if device.type == 'cuda' else {}
        return DataLoader(data, shuffle=shuffle,
                          batch_size=(self.batch_size if batch_size is None
                                      else batch_size))

    def train_loader(self,
                     device: torch.device = None,
                     batch_size: int = None) -> DataLoader:
        """Return a loader for the train data with default settings.
        Train data is retrieved from the :py:attr:`data` triple."""
        batch_size = batch_size or self.batch_size
        device = device if device is not None else self.device
        return self.loader(self.data.train, shuffle=True,
                           device=device, batch_size=batch_size)

    def disable_early_stopping(self):
        """Disable early stopping.
        This is done by setting :py:attr:`early_stopping_handle` to ``None``."""
        self.early_stopping_handle = None

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

    def train(self,
              epochs: int = None,
              early_stopping_handle: EarlyStoppingHandle = None,
              batch_info_templ: str = "Epoch {epoch}/{tot_epochs}\t"
                                      "Batch {batch}/{tot_batches}\t",
              show_progress_bars: bool = True,
              pbar_desc_templ: str = "Epoch {epoch}/{tot_epochs}",
              device: torch.device = None,
              **custom_args
              ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train the model according to the specified training parameters.
        Defaults are taken from :py:attr:`settings`.
        To override specify ``custom_args`` (compare arguments to
        :py:attr:`train_val_one_epoch`).

        :param early_stopping_handle: handle for early stopping;
            set to ``False`` to disable;
            make sure it is reset if handle given and this is required!
        :param epochs: maximum number of epochs to run
        :param device: the device on which to run training and evaluation
        :param show_progress_bars: whether to verbosely show the progress using
            :py:class:`tqdm.tqdm` progress bar
        :param batch_info_templ: string template for logging epoch and batch,
            including as substring

            - ``{epoch}``: the current epoch number
            - ``{tot_epoch}``: total number of epochs
            - ``{batch}``: the current batch number
            - ``{tot_batches}``: total number of batches

        :param pbar_desc_templ: template for the progress bar description;
            must accept as substring

            - ``{epoch}``: the current epoch number
            - ``{tot_epoch}``: the total number of epochs

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
        _validate_templ(batch_info_templ,
                        ('epoch', 'tot_epochs', 'batch', 'tot_batches'))
        _validate_templ(pbar_desc_templ, ('epoch', 'tot_epochs'))

        epochs = epochs or self.epochs
        # Push model to correct device and reset training handles
        optimizer = custom_args.get('optimizer', self.optimizer)
        self.reset_training_handles(optimizer=optimizer,
                                    device=device,
                                    model=custom_args.get('model', None))
        if early_stopping_handle is None:
            early_stopping_handle = self.early_stopping_handle
        # endregion

        history_train = {}
        history_val = {}
        for epoch in range(0, epochs):
            # Train and evaluate (with nice progress bar if requested)
            kpis_train, kpis_val = self.train_val_one_epoch(
                **custom_args,
                show_progress_bar=show_progress_bars,
                pbar_desc=pbar_desc_templ.format(epoch=epoch + 1,
                                                 tot_epochs=epochs),
                batch_info_templ=batch_info_templ.format(
                    epoch=epoch + 1, tot_epochs=epochs,
                    batch='{batch}', tot_batches='{tot_batches}'))
            optimizer.epoch_end()

            history_train[epoch] = kpis_train
            history_val[epoch] = kpis_val
            # Stop early if necessary
            if (early_stopping_handle and
                    early_stopping_handle.step(
                        kpis_val[self.val_(self.LOSS_KEY)])):
                LOGGER.info("Stopped early after %d epochs.", epoch + 1)
                break
        return pd.concat(history_train), pd.DataFrame(history_val).transpose()

    @staticmethod
    def _log_range(min_val: float, max_val: float):  # TODO: test
        r"""Iterate in :math:`\times 10` steps from ``min_val`` to ``max_val``.
        """
        if max_val < min_val:
            raise ValueError("max ({}) was smaller than min ({})."
                             .format(max_val, min_val))
        if min_val <= 0:
            raise ValueError("Only min >0 allowed, but min was {}"
                             .format(min_val))
        return [min_val * (10 ** a) for a in
                range(0, int(np.log10(max_val / min_val)))]

    def cross_validate(self, num_splits: int = 5,
                       train_val_data=None, batch_size: Optional[int] = None,
                       run_info_templ: str = 'Run {run}/{runs}',
                       **custom_args
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
            the template is prefixed to ``batch_info_templ`` and
            ``pbar_desc_templ``
        :param batch_size: optional ``batch_size`` to use for the loaders;
            defaults to :py:attr:`batch_size`
        :param num_splits: number of equal-sized, distinct validation splits
            to use
        :param train_val_data: optional given dataset to split into train and
            validation dataset splits; defaults to the ``train_val`` split in
            :py:attr:`data`
        :param custom_args: further custom training args overriding defaults
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
        model: torch.nn.Module = custom_args.get('model', self.model)
        train_val_data = train_val_data if train_val_data is not None else \
            ConcatDataset([self.data.train, self.data.val])

        def logging_settings(curr_run: int) -> Dict[str, str]:
            """Return logging template settings for given lr and run.
            Helper function."""
            epoch_info = ", epoch {epoch}/{tot_epochs}"
            run_info = run_info_templ.format(run=curr_run + 1, runs=num_splits)
            return dict(
                pbar_desc_templ=run_info + custom_args.get(
                    'pbar_desc_templ', epoch_info),
                batch_info_templ=run_info + custom_args.get(
                    'batch_info_templ',
                    epoch_info + ", batch {batch}/{tot_batches}"))

        # end region

        # Save original model parameters:
        orig_state_dict = copy.deepcopy(model.state_dict())

        splits: List[Tuple[Subset, Subset]] = \
            cross_validation_splits(train_val_data, num_splits=num_splits)
        results: List[Tuple[Dict[str, torch.Tensor],
                            pd.DataFrame,
                            pd.DataFrame]] = []
        for run, (train_data, val_data) in enumerate(splits):
            model.reset_parameters()
            history_train, history_val = self.train(
                # Use the created train val split
                train_loader=self.loader(
                    train_data, batch_size=batch_size,
                    device=custom_args.get('device', None)),
                val_loader=self.loader(
                    val_data, batch_size=batch_size,
                    device=custom_args.get('device', None)),
                # Use the amended log infos
                **{**custom_args, **logging_settings(run), 'model': model}
            )
            # collect the state dict
            curr_state_dict: Dict[str, torch.Tensor] = model.state_dict()
            # properly copy it and move to CPU to not overload GPU:
            for name, param in curr_state_dict.items():
                curr_state_dict[name].data = \
                    param.data.detach().cpu().clone().requires_grad_(
                        param.data.requires_grad)
            # save results:
            results.append((curr_state_dict, history_train, history_val))

        # Restore original model parameters:
        model.load_state_dict(orig_state_dict)

        return results

    def assess_learning_rates(self,
                              min_lr: float = 1e-8,
                              max_lr: float = 1,
                              lr_candidates: Sequence[float] = None,
                              runs_per_lr: int = 5,
                              **train_args):
        r"""Collect mean and std deviation of loss for lr candidates via
        cross-validation.
        If no ``lr_candidates`` are given, candidates are chosen in a
        logarithmic range from ``min_lr`` to ``max_lr``, i.e.
        ``min_lr * (10**a)`` for ``a=0`` to ``int(log10(max_lr / min_lr))``.

        :param lr_candidates: learning rate candidate values to collect
            performance values for
        :param min_lr: minimum learning rate to automatically determine
            candidates; overridden by ``lr_candidates``
        :param max_lr: maximum learning rate to automatically determine
            candidates; overridden by ``lr_candidates``
        :param runs_per_lr: number of cross-validation splits to make for each
            learning rate candidate
        :param train_args: further arguments to override instance defaults for
            the training runs, like ``epochs``, ``early_stopping_handle``
            etc. (see :py:meth:`train`);
            must not encompass ``train_loader`` or ``val_loader``
            (specify ``train_val_data`` instead)
        :return: :py:class:`pandas.DataFrame` indexed by the learning rate
            candidates with columns for mean and standard deviation of the
            last epoch's validation loss
        """
        # region Defaults and value checks
        if runs_per_lr <= 0:
            raise ValueError("runs_per_lr must be >0, but was {}"
                             .format(runs_per_lr))
        lr_candidates = lr_candidates if lr_candidates is not None else \
            self._log_range(min_lr, max_lr)
        for l_rate in (lr for lr in lr_candidates if lr <= 0):
            raise ValueError("Learning rate candidates must be >0, but got {}"
                             .format(l_rate))
        # We are only interested interested in the loss results by default:
        train_args['metric_fns'] = train_args.get('metric_fns', {})
        # The optimizer is replaced by one with the same settings
        # but controlled learning rate:
        optim: ResettableOptimizer = train_args.pop('optimizer', self.optimizer)
        optim_args = {**optim.settings,
                      **dict(lr_scheduler_type=None, batch_update_lr=False)}
        # endregion

        # Collect cross-validation results for each learning rate
        lr_results = []
        for l_rate in lr_candidates:
            cv_results = self.cross_validate(
                # Here is the magic: provide optimizer w/ fixed learning rate.
                optimizer=self.reset_optimizer(
                    ResettableOptimizer(**{**optim_args, 'lr': l_rate})),
                run_info_templ="LR {: <.2e}".format(
                    l_rate) + ", run {run}/{runs}",
                **train_args
            )
            # Select validation results from last epoch:
            lr_results.append(pd.DataFrame(
                [hist_val.iloc[-1] for _, _, hist_val in cv_results]))

        # Collect the mean and standard deviation of the loss results
        loss_key = self.val_(self.LOSS_KEY)
        return pd.DataFrame({
            'lr': lr_candidates,
            'mean': [float(res[loss_key].mean(axis=0)) for res in lr_results],
            'std.dev': [float(res[loss_key].std(axis=0)) for res in lr_results]
        }).set_index('lr')

    # def cyclic_lr_upper_bound(self):
    #     """Conduct a cross-validation to find the upper bound for a cyclic
    #     learning rate.
    #     https://towardsdatascience.com/2bf904d18dee
    #     """
    #     # TODO: algorithm to find bounds for cyclic learning rate
    #     raise NotImplementedError()

    def train_val_one_epoch(self,
                            show_progress_bar: bool = True,
                            pbar_desc: str = "Epoch ??/??",
                            batch_info_templ: str = "Epoch ??/??\tBatch "
                                                    "{batch}/{tot_batches}",
                            **custom_args
                            ) -> Tuple[pd.DataFrame, pd.Series]:
        """Train for one epoch, evaluate, and return history and test results.
        This is a wrapper around :py:meth:`train_one_epoch` and
        :py:meth:`evaluate` with nice progress bar printing and logging after
        the epoch.
        History and test results are stored in a :py:class:`pandas.DataFrame`.
        The device used for training is that of the parameters of the used
        model (see :py:meth:`device_of`).

        :param show_progress_bar: whether to log the batch progress, training
            latest loss and metrics using :py:class:`tqdm.tqdm`
        :param batch_info_templ: formatting template for logging that
            contains as substring ``{batch}`` and ``{tot_batches}``
        :param pbar_desc: leading static description text for the progress bar
        :return: tuple of training history and test results; columns resp.
            index are ``loss`` and the KPI names
            (keys from dict :py:attr:`metric_fns`).

            - :py:class:`pandas.DataFrame`: index are the batch indices,
              items are the results of KPI evaluations of the output on the
              training batch (i.e. before back-propagation step)
            - :py:class:`pandas.Series`: the items are the final evaluations of
              the KPIs on the validation set
        """
        # Progress bar preparations
        # set the default train_loader
        train_loader = custom_args.setdefault(
            'train_loader',
            self.train_loader(
                batch_size=custom_args.get('batch_size', None),
                device=self.device_of(custom_args.get('model', self.model)))
        )
        pbar = None if not show_progress_bar else \
            tqdm(desc=pbar_desc, total=len(train_loader))

        # Training
        train_kpi_vals = self.train_one_epoch(
            show_progress_bar=show_progress_bar, pbar=pbar,
            batch_info_templ=batch_info_templ,
            **custom_args
        )

        # Evaluation
        val_kpi_vals: pd.Series = pd.Series(
            self.evaluate(mode=DatasetSplit.VAL, **custom_args))

        # Logging
        self._log_after_epoch(train_kpi_vals.mean(), val_kpi_vals, pbar=pbar)
        if show_progress_bar:
            pbar.close()

        return train_kpi_vals, val_kpi_vals

    def train_one_epoch(self,
                        show_progress_bar: bool = True,
                        pbar: tqdm = None,
                        pbar_desc: str = "Epoch ??/??",
                        batch_info_templ: str = "Epoch ??/??\t"
                                                "Batch {batch}/{tot_batches}",
                        batch_size: int = None,
                        **custom_args
                        ) -> pd.DataFrame:
        """Train for one epoch and return history results as
        :py:class:`pandas.DataFrame`.
        This is a wrapper around :py:meth:`_train_one_epoch` that uses
        defaults from :py:attr:`settings`.
        Override the defaults by specifying ``custom_args`` (but watch out
        that model and optimizer fit together, if any of them is overridden!).

        :param pbar: optional progress bar; hand over if more control over
            the progress bar is required
        :param show_progress_bar: whether to log the batch progress,
            training latest loss and metrics using :py:class:`tqdm.tqdm`
        :param batch_info_templ: formatting template for logging that contains
            as substring ``{batch}`` and ``{tot_batches}``
        :param pbar_desc: leading static description text for the progress bar
            if newly created
        :param batch_size: batch size for the ``train_loader`` if this is not
            given in ``custom_args``
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
        model: torch.nn.Module = custom_args.setdefault('model', self.model)
        train_args = {**dict(loss_fn=self.loss_fn,
                             metric_fns=self.metric_fns,
                             train_loader=self.train_loader(
                                 batch_size=batch_size,
                                 device=self.device_of(model)),
                             optimizer=self.optimizer,
                             pbar=pbar,
                             batch_info_templ=batch_info_templ),
                      **custom_args}
        # Progress bar handling
        if show_progress_bar and pbar is None:
            train_args.update(pbar=tqdm(desc=pbar_desc,
                                        total=len(train_args['train_loader'])))
        # Actual training
        return self._train_one_epoch(**train_args)

    @classmethod
    def _train_one_epoch(cls,
                         model,
                         loss_fn,
                         metric_fns,
                         train_loader: DataLoader,
                         optimizer: Union[Optimizer, ResettableOptimizer],
                         pbar: tqdm = True,
                         batch_info_templ: str = "Epoch ??/??\t"
                                                 "Batch {batch}/{tot_batches}",
                         **_unused_args) -> pd.DataFrame:
        """Train for one epoch, evaluate, and return history and test results.
        History and test results are stored in a :py:class:`pandas.DataFrame`.
        The device used is the one the model lies on. Distributed models are
        not supported.

        :meta public:
        :param model: model to train
        :param loss_fn: function that calculates the optimization objective
            value
        :param metric_fns: further KPI functions to gather training stats
        :param train_loader: train data loader
        :param optimizer: optimizer to use for weight update steps
            initialized with model's weights
        :param pbar: optional :py:class:`tqdm.tqdm` progress bar which is
            updated after each batch
        :param batch_info_templ: formatting template for logging
            that contains as substring ``{batch}`` and ``{tot_batches}``
        :return: training history as :py:class:`pandas.DataFrame` with

            :columns:
                ``loss`` and the KPI names
                (keys from dict :py:attr:`metric_fns`),
            :index: the batch indices,
            :items:
                the results of KPI evaluations of the output on the training
                batch (i.e. *before* back-propagation step)
        """
        if len(train_loader) == 0:
            raise ValueError("Empty training loader (no batches)! "
                             "Batch size too large?")
        # very simple way to find out the correct device for
        # non-distributed models:
        device = cls.device_of(model)
        # Training
        model.train()
        train_kpi_vals: pd.DataFrame = pd.DataFrame(
            columns=[cls.train_(s) for s in
                     [cls.LOSS_KEY, *list(metric_fns.keys())]])
        for batch_idx, (data, target) in enumerate(train_loader):
            data: torch.Tensor = data.to(device)
            target: torch.Tensor = target.to(device)

            # Reset optimizer gradients
            optimizer.zero_grad()

            # Calculate loss tensor & metric(s) values
            output: torch.Tensor = model(data)
            loss_tensor = loss_fn(output, target)
            train_kpi_vals.loc[batch_idx] = {
                cls.train_(cls.LOSS_KEY): loss_tensor.item(),
                **{cls.train_(m): fn(output, target).item()
                   for m, fn in metric_fns.items()}}

            # Do back-propagation and apply to weight
            loss_tensor.backward()
            optimizer.step()

            # Logging
            cls._log_after_batch(
                train_kpi_vals.loc[batch_idx], pbar=pbar,
                batch_info=batch_info_templ.format(batch=batch_idx + 1,
                                                   tot_batches=len(train_loader)
                                                   ))

        return train_kpi_vals

    @staticmethod
    def _log_after_batch(batch_kpi_vals: pd.Series, batch_info: str,
                         pbar: tqdm = None) -> None:
        """Logging of training KPI values after one batch.
        Optionally also append information to a progress bar.

        :param batch_info: information about batch and epoch to prefix to the
            KPI information
        :param batch_kpi_vals: :py:class:`pandas.Series` indexed by the
            KPI names for training with the KPI values over the last training
            batch
        :param pbar: progressbar to update the postfix of if given
        """
        # log message content
        message = '\r {batch_info}\t{kpi_info}'.format(
            batch_info=batch_info,
            kpi_info='\t'.join(["{}: {:.6f}".format(m, v)
                                for m, v in batch_kpi_vals.items()]))
        # actual logging
        LOGGER.info(message)

        # progress bar update
        if pbar is not None:
            pbar.set_postfix(**batch_kpi_vals)
            pbar.update()

    @staticmethod
    def _log_after_epoch(train_kpi_vals: pd.Series, val_kpi_vals: pd.Series,
                         pbar: Optional[tqdm] = None) -> None:
        """Logging of mean training and validation KPI values after one epoch.
        Optionally also append information to a progress bar.

        :param train_kpi_vals: :py:class:`pandas.Series` indexed by the KPI
            names for training with the mean KPI values over the last
            training epoch
        :param val_kpi_vals: :py:class:`pandas.Series` indexed by the KPI
            names for testing with the KPI values over the validation set
            after the epoch
        :param pbar: progressbar to update the postfix of if given
        """
        # Logging of evaluation results
        # normal logging
        LOGGER.info("Mean train results: %s",
                    '\t'.join(["     {}={:.6f}".format(m, v) for m, v in
                               train_kpi_vals.items()]))
        LOGGER.info("Test eval results:  %s",
                    '\t'.join(["{}={:.6f}".format(m, v) for m, v in
                               val_kpi_vals.items()]))
        # progress bar logging
        if pbar is not None:
            pbar.set_postfix(
                **{"{}".format(m): v for m, v in
                   [*train_kpi_vals.items(), *val_kpi_vals.items()]})
            pbar.refresh()

    def evaluate(self,
                 mode: Union[str, DatasetSplit] = DatasetSplit.TEST,
                 batch_size: int = None,
                 **custom_args) -> pd.Series:
        """Evaluate the model wrt. :py:attr:`settings`.
        This is a wrapper around :py:meth:`_evaluate` which uses the defaults
        given by :py:attr:`settings`.
        Override them by specifying them as ``custom_args``.
        The device used for evaluation is the one of the model determined using
        :py:meth:`device_of`.

        :param mode: which data set to use; specify as instance of
            :py:class:`~hybrid_learning.datasets.base.DatasetSplit`
            or the name of one
        :param batch_size: batch size used for the ``val_loader`` if that is
            not given within ``custom_args``
        :return: Dictionary of all KPIs, i.e. of ``loss`` and each metric
            in :py:attr:`metric_fns`;
            format: ``{<KPI-name>: <KPI value as float>}``
        """
        # region Value check
        if mode in [d.name for d in DatasetSplit]:
            mode = DatasetSplit[mode]
        if mode not in (DatasetSplit.VAL, DatasetSplit.TEST):
            raise ValueError("Invalid mode {}; accepting one of {}".format(
                mode, (DatasetSplit.VAL, DatasetSplit.TEST,
                       DatasetSplit.VAL.value, DatasetSplit.TEST.value)))
        # endregion

        model = custom_args.get('model', self.model)
        default_eval_args = dict(
            model=model,
            kpi_fns={self.LOSS_KEY: custom_args.get('loss_fn', self.loss_fn),
                     **custom_args.get('metric_fns', self.metric_fns)},
            prefix_=(lambda s: self.prefix_by(mode, s)),
            val_loader=self.loader(
                self.data.test if mode == DatasetSplit.TEST else self.data.val,
                batch_size=batch_size, device=self.device_of(model)))
        return self._evaluate(**{**default_eval_args, **custom_args})

    @classmethod
    def _evaluate(cls,
                  model: torch.nn.Module,
                  kpi_fns: Dict[str, Callable],
                  val_loader: DataLoader,
                  prefix_: Callable[[str], str] = None,
                  **_unused_args) -> pd.Series:
        """Evaluate the model wrt. loss and :py:attr:`metric_fns`
        on the test data.
        The reduction method for the KPI values is ``mean``.
        The device used is the one of the model lies on (see
        :py:meth:`device_of`). Distributed models are not supported.

        :meta public:
        :param model: the model to evaluate
        :param kpi_fns: dictionary with KPI IDs and evaluation functions for
            the KPIs to evaluate
        :param val_loader: data loader with data to evaluate on
        :param prefix_: wrapper to prefix KPI names for the final
            :py:class:`pandas.Series` naming
        :return: Dictionary of all KPI values in the format:
            ``{<KPI-name>: <KPI value as float>}``
        """
        # very simple way to find out the correct device for
        # non-distributed models:
        prefix_ = prefix_ if prefix_ is not None else cls.val_
        device = cls.device_of(model)
        # Value check and defaults
        if len(val_loader) == 0:
            raise ValueError("Empty evaluation data loader (no batches)! "
                             "Batch size too large?")

        # Combine loss and metrics as general KPI measures
        eval_kpi_vals: Dict[str, float] = {prefix_(m): 0. for m in
                                           kpi_fns.keys()}

        model.eval()
        with torch.no_grad():
            # Gather KPI values from all batches
            for data, target in val_loader:
                data: torch.Tensor = data.to(device)
                target: torch.Tensor = target.to(device)

                # Add metric from batch
                output = model(data)
                for kpi in kpi_fns.keys():
                    eval_kpi_vals[prefix_(kpi)] += kpi_fns[kpi](output,
                                                                target).item()

            # Aggregate KPI values of all batches via mean
            num_batches = len(val_loader)
            for eval_kpi in eval_kpi_vals.keys():
                eval_kpi_vals[eval_kpi] /= num_batches

        return pd.Series(eval_kpi_vals)

    @staticmethod
    def device_of(model: torch.nn.Module) -> torch.device:
        """Return the device of the given pytorch model.
        Distributed models are not supported."""
        device = next(model.parameters()).device \
            if len(list(model.parameters())) > 0 else 'cpu'
        return device
