#  Copyright (c) 2022 Continental Automotive GmbH
"""Callback functionality and typical callbacks for training.
A callback simply is a mapping from a :py:class:`CallbackEvents` instance
to a callable that accepts the callback context.
The :py:func:`run_callbacks` method can be used to run those callables
associated with a given event from a list of callbacks.
See :py:class:`~hybrid_learning.concepts.train_eval.callbacks.Callback` for details.
"""

import enum
import logging
import os
import string
from typing import Dict, Callable, Any, Iterable, Mapping, List, Optional, \
    Sequence, Tuple, Union

import matplotlib.lines
import numpy as np
import pandas as pd
import torch.utils.data
import torch.utils.tensorboard as tb
import torchvision as tv
from matplotlib import pyplot as plt
from tqdm import tqdm

LOGGER = logging.getLogger()
"""The default logger in this module."""


class CallbackEvents(enum.Enum):
    """Supported callback events for training and evaluation.
    Events can be:

        - ``before_epoch_train``: right before iterating one epoch of the
            training set
        - ``after_batch_train``, ``after_epoch_eval``:
            after every batch of an epoch of training
        - ``after_epoch_train``: right after iterating one epoch of the
            training set; provides kpis of the training set of current epoch,
            and the ``train_loader``
        - ``after_epoch_eval``: right after iterating one epoch of evaluation
            (validation or test set);
            provides evaluation kpis of the current epoch and the ``val_loader``
        - ``after_epoch``: after an epoch of training: provides kpis of
            current epoch
        - ``between_epochs``: after an epoch of a training run with several
            epochs: provides epoch context (total KPI history, current and
            total epoch number)
        - ``after_train``: after the training of several epochs;
            provides full training and validation history as well as total
            epoch number
        - ``after_second_stage_train``: like ``after_train`` only after the
            second stage training
    """
    BEFORE_EPOCH_TRAIN = "before_epoch_train"
    AFTER_BATCH_TRAIN = "after_batch_train"
    AFTER_EPOCH_TRAIN = "after_epoch_train"
    AFTER_BATCH_EVAL = "after_batch_eval"
    AFTER_EPOCH_EVAL = "after_epoch_eval"
    AFTER_EPOCH = "after_epoch"
    BETWEEN_EPOCHS = "between_epochs"
    AFTER_TRAIN = "after_train"
    AFTER_SECOND_STAGE_TRAIN = "after_second_stage_train"


def run_callbacks(callbacks: Iterable[Mapping[CallbackEvents, Callable]],
                  event: CallbackEvents,
                  callback_context: Dict[str, Any] = None):
    """Run the given callback collection in order."""
    callback_context = callback_context or {}
    for callback in callbacks:
        cb_fun: Optional[Callable] = callback.get(event, None)
        if cb_fun:
            LOGGER.debug("Passing callback event %s to %s", str(event), repr(callback))
            cb_fun(**callback_context)


def _validate_templ(templ: str, var_names: Iterable[str], mode: str = 'all'):
    """Raise if the template is missing a variable name."""
    format_keys = [i[1] for i in string.Formatter().parse(templ)
                   if i[1] is not None]
    if mode not in ['all', 'only', 'exact']:
        raise ValueError("Unknown validation mode {}".format(mode))
    if mode in ['all', 'exact'] and \
            any([s not in format_keys for s in var_names]):
        raise ValueError(("string template must contain all formatting strings "
                          "{} but was {}").format(format_keys, templ))
    if mode in ['only', 'exact']:
        unknown_keys = [k for k in format_keys if k not in var_names]
        if len(unknown_keys) > 0:
            raise ValueError(("string template {} contained unknown format "
                              "keys {}; allowed: {}").format(
                templ, unknown_keys, format_keys))


class Callback(Mapping):  # pylint: disable=too-many-ancestors
    """A callback base class that eases implementing a custom callback handle.

    A callback handle simply is a mapping of :py:class:`CallbackEvents` to
    callback functions.
    One callback can define several pairs of event and callback function.
    A callback function is any function that can be called via
    ``callback_fun(**callback_context)``, where ``callback_context``
    is a mapping of key-value pairs.
    The correct callback function for a given event and callback context
    can be called using :py:func:`run_callbacks`.

    To add a callback function to an implementation of this class,
    just implement it as method with the same name as the callback
    event's value.
    Make sure to add a `**_` catch-all argument to each defined
    callback function to allow ignoring unused key-value pairs from
    the callback context.
    """

    def keys(self) -> List[CallbackEvents]:
        """Collect a list of all supported callback events."""
        return [event for event in CallbackEvents if event.value in dir(self)]

    def __getitem__(self, event: CallbackEvents):
        """Get the method for the specified event or a pass lambda."""
        return getattr(self, event.value, lambda **cb_context: None)

    def __len__(self):
        """Number of mapped events."""
        return len(self.keys())

    def __iter__(self):
        """Iterate over mapped events."""
        return iter(self.keys())

    @staticmethod
    def to_descriptor(log_prefix: str = None, run: str = None,
                      epoch: int = None, batch: int = None) -> str:
        """Merge logging information into file name descriptor.
        ``log_dir`` is prepended as path, the rest is merged with ``_``."""
        components_groups: List[List[Tuple[str, Any]]] = [
            [('', log_prefix), ('run', run)],
            [('epoch', epoch)], [('batch', batch)]]
        sub_descs: List[str] = []
        for components in components_groups:
            substrings: List[str] = [f"{pref}{val}" for pref, val in components
                                     if val is not None]
            if len(substrings) == 0:
                continue
            sub_descs.append("_".join(substrings).replace("/", "_").replace(" ", "_"))
        return os.path.join(*sub_descs) if len(sub_descs) > 0 else ""

    @staticmethod
    def from_descriptor(desc: str) -> Dict[str, str]:
        """Given a descriptor, extract contained information."""
        if not desc:
            return {}
        info = {}
        component_keys = ('epoch', 'batch', 'run')
        substrings: List[str] = [s for d in os.path.normpath(desc).split(os.sep)
                                 for s in d.split('_')]
        log_prefix_list = []
        for substr in substrings:
            # Collect substrings with component key:
            key_list = [k for k in component_keys if substr.startswith(k)]
            if len(key_list) > 0:
                info[key_list[0]] = substr.split(key_list[0], maxsplit=1)[-1]
                continue
            # After first substring with component key, all must have one:
            if len(info) > 0:
                raise ValueError("Invalid descriptor {}: Found undefined substring {}"
                                 .format(desc, substr))
            # First substrings belong to log_prefix:
            log_prefix_list.append(substr)
        if len(log_prefix_list) > 0:
            info['log_prefix'] = "_".join(log_prefix_list)
        return info


class LoggingCallback(Callback):  # pylint: disable=too-many-ancestors
    """Log batch and epoch KPI results."""

    def __init__(self, logger: logging.Logger = None,
                 log_level: int = logging.INFO, log_per_batch: Union[bool, int] = False):
        self.logger: Optional[logging.Logger] = logger
        """The default logger to use."""
        self.log_level: int = log_level
        """The logging level to use. Defaults to :py:const:`logging.INFO`."""
        self.log_per_batch: int = max(0, int(log_per_batch))
        """Log after each ith batch.
        If set to ``False`` or 0 don't log batches at all, if ``True`` or 1 log each batch."""
        self._batches_until_next_log: int = 0
        """Number of batches until the next logging of a batch."""

    @staticmethod
    def _default_batch_info_templ_for(batch, batches,
                                      epoch, epochs, run, runs):
        """Get the default ``batch_info_templ`` for the given params."""
        batch_info_templ = ", ".join(part for part in [
            "Run {run}/{runs}" if any([run, runs]) else None,
            "Epoch {epoch}/{epochs}" if any([epoch, epochs]) else None,
            "Batch {batch}/{batches}" if any(
                [batch, batches]) else None,
        ] if part is not None)
        return batch_info_templ

    def after_batch_train(self,
                          # pylint: disable=unused-argument,too-many-arguments
                          kpi_train: pd.DataFrame, batch: int, batches: int,
                          epoch: int = None, epochs: int = None,
                          run: int = None, runs: int = None,
                          batch_info_templ: str = None, log_prefix: str = "",
                          logger: logging.Logger = None,
                          **_unused_args
                          ) -> None:
        """Logging of training KPI values after one batch.

        :param batch: the current batch index
        :param epoch: the current epoch index
        :param run: the current cross-validation run index
        :param batches: the total number of batches
        :param epochs: the total number of epochs
        :param runs: the total number of cross-validation runs
        :param log_prefix: text to prepend to log messages
        :param batch_info_templ: string template for logging epoch and batch,
            which may include as substring

            - ``{run}``/``{epoch}``/``{batch}``: the current
              cross-validation run/epoch/batch number
            - ``{runs}``/``{epochs}``/``{batches}``: the total number of
              cross-validation runs/epochs/batches

        :param kpi_train: :py:class:`pandas.Series` indexed by the
            KPI names for training with the KPI values over the last training
            batch
        :param logger: do not use this instances logger but a different one
        """
        logger = logger or self.logger
        if logger is None or logger.level > logging.INFO or \
                not self.log_per_batch or self._batches_until_next_log > 0:
            self._batches_until_next_log -= 1
            return
        self._batches_until_next_log = self.log_per_batch - 1

        batch_kpi_vals = kpi_train.loc[batch]

        # log message content
        if batch_info_templ is None:
            batch_info_templ = self._default_batch_info_templ_for(
                batch, batches, epoch, epochs, run, runs)
        else:
            _validate_templ(batch_info_templ, ('epoch', 'epochs', 'batch',
                                               'batches', 'runs', 'runs'),
                            mode='only')
        log_msg_vals = dict(
            run=run + 1 if run is not None else "??",
            runs=runs if runs is not None else "??",
            epoch=epoch + 1 if epoch is not None else "??",
            epochs=epochs if epochs is not None else "??",
            batch=batch + 1 if batch is not None else "??",
            batches=batches if batches is not None else "??")

        message = '{batch_info}: \t{kpi_info}'.format(
            batch_info=(batch_info_templ.format(**{
                k: val for k, val in log_msg_vals.items()
                if "{" + k + "}" in batch_info_templ})),
            kpi_info='\t'.join([
                "{}: {:.6f}".format(m, v)
                for m, v in batch_kpi_vals.items()
                if not isinstance(v, plt.Figure) and not np.isnan(v)]))
        # actual logging
        logger.log(self.log_level, "%s %s", log_prefix, message)

    def after_epoch(self,  # pylint: disable=unused-argument
                    kpi_train: pd.DataFrame,
                    kpi_val: pd.Series,
                    logger: logging.Logger = None,
                    log_prefix: str = "",
                    **_unused_args) -> None:
        """Logging of mean training and validation KPI values after one epoch.

        :param kpi_train: :py:class:`pandas.Series` indexed by the KPI
            names for training with the mean KPI values over the last
            training epoch
        :param kpi_val: :py:class:`pandas.Series` indexed by the KPI
            names for testing with the KPI values over the validation set
            after the epoch
        :param log_prefix: text to prepend to log messages
        :param logger: do not use this instances logger but a different one
        """
        logger = logger or self.logger
        self._batches_until_next_log = 0
        if logger is None:
            return

        # Logging of evaluation results
        logger.log(self.log_level, "%s Mean train results: %s", log_prefix,
                   '\t'.join(["{}={:.6f}".format(m, v) for m, v in
                              kpi_train.mean().items()
                              if not isinstance(v, plt.Figure)]))
        logger.log(self.log_level, "%s Test eval results:  %s", log_prefix,
                   '\t'.join(["{}={:.6f}".format(m, v) for m, v in
                              kpi_val.items()
                              if not isinstance(v, plt.Figure)]))


class ProgressBarUpdater(Callback):  # pylint: disable=too-many-ancestors
    """Update the progress bar postfix after each batch and epoch.
    For batch-wise updates, only :py:attr:`train_kpis` are considered."""

    def __init__(self, train_kpis: Sequence[str] = ('loss',)):
        super().__init__()
        self.train_kpis: Sequence[str] = train_kpis
        """The list of KPI-names to include into the training logging.
        A KPI is included if it includes one of the ``train_kpis`` strings."""

    def after_batch_train(self,
                          kpi_train: pd.DataFrame, batch: int,
                          pbar: tqdm = None, **_unused_args):
        """Update the progress bar with the training KPI values after a
        batch."""
        if pbar is None:
            return
        to_print: Dict[str, float] = {
            name: scalar for name, scalar in dict(kpi_train.loc[batch]).items()
            if any(n in name for n in self.train_kpis) and not isinstance(
                scalar, plt.Figure) and not np.isnan(scalar)}
        pbar.set_postfix(**to_print)
        pbar.update()

    @staticmethod
    def after_batch_eval(pbar_eval: tqdm = None, **_unused_args):
        """Update the progress bar with the evaluation KPI values
        after a batch."""
        if pbar_eval is None:
            return
        pbar_eval.update()

    @staticmethod
    def after_epoch(
            kpi_train: pd.DataFrame = None,
            kpi_val: pd.Series = None,
            pbar: Optional[tqdm] = None, **_unused_args):
        """Update the progress bar with the train and validation KPI values
        after an epoch."""
        if pbar is None:
            return
        kpi_train_mean = kpi_train.infer_objects().mean() if kpi_train is not None else {}
        kpi_val = kpi_val if kpi_val is not None else {}
        pbar.set_postfix(
            **{"{}".format(m): v for m, v in
               [*kpi_train_mean.items(), *kpi_val.items()]
               if not isinstance(v, plt.Figure)})
        pbar.refresh()

    def after_epoch_train(self, kpi_train: pd.Series = None,
                          pbar: Optional[tqdm] = None, **_unused_args):
        """Update the evaluation progress bar with the final validation KPI values."""
        self.after_epoch(kpi_train=kpi_train, pbar=pbar)

    def after_epoch_eval(self, kpi_val: pd.Series = None,
                         pbar_eval: Optional[tqdm] = None, **_unused_args):
        """Update the evaluation progress bar with the final validation KPI values."""
        self.after_epoch(kpi_val=kpi_val, pbar=pbar_eval)


class TensorboardLogger(Callback):  # pylint: disable=too-many-ancestors
    """Write batch and epoch KPI results to a tensorboard log directory."""

    def __init__(self, log_dir: str = "runs",
                 log_sample_inputs: bool = False,
                 log_sample_targets: bool = False):
        self.log_dir: str = log_dir
        """The root logging directory."""
        self.log_sample_inputs: bool = log_sample_inputs
        """Whether to interpret the model inputs as image and log some."""
        self.log_sample_targets: bool = log_sample_targets
        """Whether to interpret the model targets as image and log some."""
        self.writers: Dict[str, tb.SummaryWriter] = {}
        """A mapping of logdir subdirectory to cached writers."""

    def flush_writers(self):
        """Close all writers to free their threads."""
        for writer in self.writers.values():
            writer.flush()
        LOGGER.debug("%s, after_epoch_train: Flushed writers.")

    def _get_writer_for(self, log_prefix=None, run=None) -> tb.SummaryWriter:
        """Collect writer for the (sub)folder determined by prefix and run."""
        logdir = self.log_dir
        super_group = self.to_descriptor(log_prefix, run)

        # Collect writer
        if super_group:
            logdir = os.path.join(logdir, super_group)
        if super_group not in self.writers:
            self.writers[super_group] = tb.SummaryWriter(logdir)

        return self.writers[super_group]

    def before_epoch_train(self, model: torch.nn.Module,
                           train_loader: torch.utils.data.DataLoader,
                           epoch: int = 0, log_prefix=None, run=None,
                           device=None, **_):
        """Log the model graph and optionally some example training images."""
        # Only before first epoch
        if epoch and epoch > 0:
            return
        device = device or 'cpu'
        writer: tb.SummaryWriter = self._get_writer_for(log_prefix, run)
        imgs, targets = next(iter(train_loader))
        # Graph
        writer.add_graph(model.to(device), imgs.to(device))
        LOGGER.debug("%s, before_epoch_train: Added model graph.", self.__class__.__name__)
        # Images
        if self.log_sample_inputs and isinstance(imgs, torch.Tensor) and len(imgs.shape) == 4:
            writer.add_image('inputs/train',
                             tv.utils.make_grid(imgs, pad_value=1))
            LOGGER.debug("%s, before_epoch_train: Added input samples.", self.__class__.__name__)
        if self.log_sample_targets and isinstance(targets, torch.Tensor) and len(targets.shape) == 4:
            writer.add_image('targets/train',
                             tv.utils.make_grid(targets, pad_value=1))
            LOGGER.debug("%s, before_epoch_train: Added target samples.", self.__class__.__name__)

    def after_batch_train(self, kpi_train: pd.DataFrame,
                          batch: int, batches: int, epoch: int,
                          log_prefix: str = None, run: int = None,
                          **_):
        """Record the training KPIs in the tensorboard log directory."""
        writer = self._get_writer_for(log_prefix, run)

        scalars = dict(kpi_train.loc[batch])
        for scalar_name, scalar in scalars.items():
            group, name = scalar_name.split('_', maxsplit=1)
            if not np.isnan(scalar) and not isinstance(scalar, plt.Figure):
                writer.add_scalar(tag=f"{name}/{group}_batchwise",
                                  scalar_value=scalar,
                                  global_step=batch + epoch * batches)

    def after_epoch_train(self, kpi_train: pd.DataFrame, epoch: int,
                          aggregating_kpis: Sequence[str] = (),
                          log_prefix: str = None, run: int = None,
                          **_):
        """After each training epoch, close all writers to free threads."""
        writer = self._get_writer_for(log_prefix, run)

        # Aggregation: mean (normal), or last value (self-aggregating metrics)
        scalars = {scalar_name: (
            kpi_train[scalar_name].mean()
            if not any(name in scalar_name for name in aggregating_kpis)
            else kpi_train[scalar_name].iloc[-1])
            for scalar_name in kpi_train.columns
        }
        for scalar_name, scalar in scalars.items():
            group, name = scalar_name.split('_', maxsplit=1)
            if isinstance(scalar, plt.Figure):
                writer.add_figure(f"{name}/{group}", scalar, epoch)
                LOGGER.debug("%s, after_epoch_train: Added figure %s/%s for epoch %d.",
                             self.__class__.__name__, name, group, epoch)
            else:
                writer.add_scalar(tag=f"{name}/{group}",
                                  scalar_value=scalar,
                                  global_step=epoch)
                LOGGER.debug("%s, after_epoch_train: Added scalar %s/%s for epoch %d.",
                             self.__class__.__name__, name, group, epoch)
        self.flush_writers()

    def after_epoch_eval(self, kpi_val: pd.Series, epoch: int = None,
                         log_prefix: str = None, run: int = None,
                         **_):
        """Record the validation and test KPIs in the tensorboard log
        directory."""
        writer = self._get_writer_for(log_prefix, run)
        scalars = dict(kpi_val)
        for scalar_name, scalar in scalars.items():
            group, name = scalar_name.split('_', maxsplit=1)
            if isinstance(scalar, plt.Figure):
                writer.add_figure(f"{name}/{group}", scalar, epoch)
                LOGGER.debug("%s, after_epoch_eval: Added figure %s/%s for epoch %d.",
                             self.__class__.__name__, name, group, epoch)
            else:
                writer.add_scalar(tag=f"{name}/{group}",
                                  scalar_value=scalar,
                                  global_step=epoch)
                LOGGER.debug("%s, after_epoch_eval: Added scalar %s/%s for epoch %d.",
                             self.__class__.__name__, name, group, epoch)
        self.flush_writers()


class CsvLoggingCallback(Callback):
    """Extract the values stored in matplotlib figures and store them as CSV.
    Only figures with one axis and one line in the plot are supported.
    Figure data is saved into a CSV file under :py:attr:`log_dir` with the
    name given by :py:meth:`~Callback.to_descriptor` with one column for x-data,
    one for y-data.

    .. note::
        For the file path format see :py:meth:`file_path_for`
        and :py:meth:`file_paths_in`.
    """
    OTHER_KPI_NAME: str = "other"
    """The default KPI name for CSV files collecting non-image metric data."""
    DEFAULT_DESC: str = "_"
    """A default description to use if no other folder identifier can be determined."""

    def __init__(self, log_dir: str = ".", overwrite: Union[str, bool] = 'warn'):
        """Init."""
        super().__init__()
        self.log_dir: str = log_dir
        """The root directory under which to store figure data."""
        self.overwrite: Union[str, bool] = overwrite
        """Whether to overwrite existing files.
        The following values are treated special:

        - ``'warn'``: overwrite but log a warning if exists
        - ``'raise'``: do not overwrite and raise a ``FileExistsError`` if exists  
        """

    def file_path_for(self, kpi_name: str, run_type: str = None,
                      log_prefix: str = None, run: int = None, epoch: int = None, batch: int = None,
                      default_desc: str = None
                      ) -> str:
        """Given logging specs provide the file path these should be located at.

        :param default_desc: in case no descriptor can be determined use this value
            (see :py:meth:`~Callback.to_descriptor`); if not given, instead raise
        """
        desc: str = self.to_descriptor(log_prefix=log_prefix, run=run, epoch=epoch, batch=batch)
        if not desc:
            if not default_desc:
                raise ValueError(("file_path not given and could not determine file name info "
                                  "from {}").format(dict(log_prefix=log_prefix, run=run, epoch=epoch, batch=batch)))
            desc = default_desc
        dir_name: str = os.path.join(self.log_dir, desc)
        file_name: str = f"{kpi_name}.csv"
        if run_type is not None:
            file_name = f'{run_type}_{file_name}'
        file_path: str = os.path.join(dir_name, file_name)
        return file_path

    def _save_to_csv(self, data: Union[pd.DataFrame, pd.Series], *,
                     kpi_name: str = OTHER_KPI_NAME, run_type: str = None,
                     log_prefix: str = None, run: int = None, epoch: int = None, batch: int = None):
        """Save number value columns of pandas Frame or Series to CSV with unique file name."""
        data = data.loc[:, [*data.infer_objects().select_dtypes(np.number).columns,
                            *(['line'] if 'line' in data.columns else [])]]  # include line identifier col
        if len(data.columns) == 0:
            return
        file_path: str = self.file_path_for(kpi_name=kpi_name, log_prefix=log_prefix, run_type=run_type,
                                            run=run, epoch=epoch, batch=batch,
                                            default_desc=self.DEFAULT_DESC)
        if os.path.exists(file_path):
            if not self.overwrite:
                return
            if self.overwrite == 'warn':
                LOGGER.warning("Overwriting for CSV logging: %s", file_path)
            if self.overwrite == 'raise':
                raise FileExistsError(("Log CSV file {} already exists! Safely remove or "
                                       "set overwrite='warn'.").format(file_path))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        LOGGER.debug("%s: Saving DataFrame at %s.", self.__class__.__name__, file_path)
        data.reset_index().to_csv(file_path, index=False)
        LOGGER.debug("%s: Saved DataFrame at %s.", self.__class__.__name__, file_path)

    @staticmethod
    def _fig_to_pd(fig: plt.Figure) -> Optional[pd.DataFrame]:
        """Extract x and y values from ``fig`` and save to ``file_path``.

        :return: the data frame that is stored to ``file_path``;
            will be ``None`` if parsing is unsuccessful
        """
        # Retrieve data
        axes: List[plt.Axes] = fig.get_axes()
        if len(axes) > 1:
            return
        ax: plt.Axes = axes[0]
        lines: List[matplotlib.lines.Line2D] = ax.get_lines()
        line_datas: List[pd.DataFrame] = []
        for line in lines:
            if not hasattr(line, 'get_label') or not hasattr(line, 'get_data'):
                LOGGER.debug("Skipping non-line object in figure.")
                continue
            line_label: str = line.get_label()
            x_vals, y_vals = line.get_data()
            x_label, y_label = ax.get_xlabel() or 'x', ax.get_ylabel() or 'y'
            line_datas.append(pd.DataFrame({x_label: x_vals, y_label: y_vals,
                                            'line': [line_label] * len(x_vals)}))
        fig_data: pd.DataFrame = pd.concat(line_datas, ignore_index=True)
        return fig_data

    def _save_figs_as_csv(self, kpi_values: pd.Series, *,
                          log_prefix: str = None, run: int = None,
                          epoch: int = None, batch: int = None):
        """Filter out figures and save them."""
        figs: Dict[str, plt.Figure] = {k: kpi_values[k] for k in kpi_values.index
                                       if isinstance(kpi_values[k], plt.Figure)}
        if len(figs) == 0:
            return
        for kpi_name, fig in figs.items():
            fig_data = self._fig_to_pd(fig)
            LOGGER.debug("%s: Turned figure for KPI %s into DataFrame.", self.__class__.__name__, kpi_name)
            self._save_to_csv(fig_data, kpi_name=kpi_name, log_prefix=log_prefix,
                              run=run, epoch=epoch, batch=batch)

    def after_epoch_train(self, kpi_train: pd.DataFrame, epoch: int = None,
                          log_prefix: str = None, run: int = None, run_type: str = 'train',
                          **_):
        """Save figures in the final batch results."""
        LOGGER.debug("%s, after_epoch_train: Starting figure saving.", self.__class__.__name__)
        self._save_figs_as_csv(kpi_train.iloc[-1], epoch=epoch, log_prefix=log_prefix, run=run)
        LOGGER.debug("%s, after_epoch_train: Starting other scalar saving.", self.__class__.__name__)
        self._save_to_csv(kpi_train, epoch=epoch, log_prefix=log_prefix, run=run, run_type=run_type)

    def after_epoch_eval(self, kpi_val: pd.Series, epoch: int = None,
                         log_prefix: str = None, run: int = None, run_type: str = 'eval',
                         **_):
        """Save figures in the eval epoch results."""
        LOGGER.debug("%s, after_epoch_eval: Starting figure saving.", self.__class__.__name__)
        self._save_figs_as_csv(kpi_val, epoch=epoch, log_prefix=log_prefix, run=run)
        LOGGER.debug("%s, after_epoch_eval: Starting other scalar saving.", self.__class__.__name__)
        self._save_to_csv(pd.DataFrame(kpi_val).T, epoch=epoch, log_prefix=log_prefix, run=run, run_type=run_type)

    def from_csv(self, file_path: str = None,
                 kpi_name: str = OTHER_KPI_NAME, run_type: str = None,
                 log_prefix: str = None, run: int = None,
                 epoch: int = None, batch: int = None) -> pd.DataFrame:
        """Read a previously saved :py:class:`pandas.DataFrame`.
        Either give ``file_path`` or a combination of the other infos."""
        file_path = file_path or self.file_path_for(
            kpi_name=kpi_name, log_prefix=log_prefix, run_type=run_type,
            run=run, epoch=epoch, batch=batch)
        if not os.path.exists(file_path):
            raise FileNotFoundError("CSV file {} does not exist.".format(file_path))
        return pd.read_csv(file_path, index_col='index').infer_objects()

    @classmethod
    def file_paths_in(cls, log_dir: str, use_abs_paths: bool = True) -> List[Dict[str, str]]:
        """Given a ``log_dir`` return information on saved metrics.

        :param log_dir: the root logging directory under which metrics were saved
        :param use_abs_paths: whether to specify paths relative to ``log_dir`` or absolute
        :return: a DataFrame containing in each row meta information and
            the relative file path to a saved metric.
        """
        data = []
        for abs_desc, _, file_names in os.walk(log_dir):
            desc: str = os.path.relpath(abs_desc, log_dir)
            for file_name in [f for f in file_names if f.endswith('.csv')]:
                kpi_name = file_name.rsplit('.')[0]
                dir_path = os.path.abspath(abs_desc) if use_abs_paths else desc
                data.append({'kpi_name': kpi_name,
                             **cls.from_descriptor(desc),
                             'file_path': os.path.join(dir_path, file_name)})
        return data
