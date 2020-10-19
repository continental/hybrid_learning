"""Wrapper around pytorch optimizer to make them resettable."""
#  Copyright (c) 2020 Continental Automotive GmbH

# pylint: disable=no-name-in-module

import logging
from typing import Optional, Dict, Any, Callable, Union, Iterable

import torch
import torch.nn
import torch.utils
from torch.optim.optimizer import Optimizer

# pylint: enable=no-name-in-module

LOGGER = logging.getLogger(__name__)


class ResettableOptimizer:
    """Wrapper around torch optimizers to enable reset and automatic learning
    rate handling.
    Saves the optimizer/learning rate scheduler initialization arguments
    other than the parameters/optimizer.
    Replace

    .. code-block:: python
        :linenos:

        opt = OptimizerType(params, **opt_init_args)
        # optionally lr_scheduler:
        lr_scheduler = LRSchedulerType(opt, **lr_init_args)
        for epoch in epochs:
            for batch in dataset:
                opt.zero_grad()
                ...
                opt.step()
                lr_scheduler.step()  # if batch-wise update
            lr_scheduler.step() # if epoch-wise update

    with

    .. code-block:: python
        :linenos:

        opt_handle = ResetOptimizer(OptimizerType,
                                    LRSchedulerType, lr_init_args,
                                    # if batch-wise lr updates:
                                    batch_update_lr=True)
        ...
        opt = opt_handle(params)
        for epoch in epochs:
            for batch in dataset:
                opt.zero_grad()
                ...
                opt.step()
            opt.epoch_end()

    A cast as in :py:meth:`torch.optim.Optimizer.cast` is currently
    not supported.
    """

    def _check_after_init(self):
        """Check whether the optimizer is initialized, otherwise raise."""
        if self.optimizer is None:
            raise ValueError(
                "Wrapped optimizer is not yet initialized. Call init().")

    def __init__(self,
                 optim_type: Callable[..., Optimizer],
                 lr_scheduler_type: Optional[Callable] = None,
                 lr_kwargs: Optional[Dict[str, Any]] = None,
                 batch_update_lr: bool = False,
                 **optim_kwargs):
        """Init.

        :param optim_type: a callable that yields an optimizer when called with
            model parameters and the ``optim_kwargs``
        :param optim_kwargs: keyword arguments for creating the optimizer
        :param lr_scheduler_type: Optional; if given, a callable that yields
            a learning rate scheduler at which to register the optimizer
        :param lr_kwargs: arguments for the optional learning rate scheduler
        :param batch_update_lr: whether to step the learning rate scheduler
            after each batch, so on :py:meth:`step` call, or only on
            :py:meth:`epoch_end` call;
            set this e.g. for :py:class:`torch.optim.lr_scheduler.CyclicLR`
        """
        # actual optimizer
        self._optim_kwargs: Dict[str, Any] = optim_kwargs
        """The arguments (besides the ``parameters`` key) to
        ``_optim_type`` to get new optimizer.

        :meta public:
        """
        self._optim_type: Callable[..., Optimizer] = optim_type
        """Type/builder for optimizer. Used in :py:meth:`init` to get new
        optimizer.

        :meta public:
        """
        self.optimizer: Optional[Optimizer] = None
        """Reference to the current optimizer.
        ``None`` after :py:meth:`reset` and before :py:meth:`init`"""

        # learning rate scheduler
        self._lr_kwargs: Dict[str, Any] = lr_kwargs or {}
        """The arguments to ``_lr_scheduler_type`` besides the
        optimizer. Only used if ``_lr_scheduler_type`` is set.

        :meta public:
        """
        self._lr_scheduler_type: Optional[Callable[..., Any]] \
            = lr_scheduler_type
        """Optional learning rate scheduler builder.
        Used in :py:meth:`init` to get new scheduler.

        :meta public:
        """
        self.batch_update_lr: bool = batch_update_lr
        """Whether to call step on the learning rate scheduler after each batch
        or only after each epoch.
        Batch-wise updates are handled in :py:meth:`step`, epoch-wise ones in
        :py:meth:`epoch_end`,
        """
        self.lr_scheduler: Optional = None
        """Reference to the optional current learning rate scheduler.
        ``None`` if ``_lr_scheduler_type`` is ``None``,
        after :py:meth:`reset` and before :py:meth:`init`."""

    # pylint: disable=invalid-name
    @property
    def lr(self) -> float:
        """The used learning rate default of the optimizer
        (starting value for lr scheduler)"""
        reset_after: bool = False
        if self.optimizer is None:
            # do a dummy init to read out defaults
            # pylint: disable=no-member
            self.init([torch.ones(1)])
            # pylint: enable=no-member
            reset_after = True
        lr: float = self.optimizer.defaults['lr']
        if reset_after:
            self.reset()
        return lr

    @lr.setter
    def lr(self, lr: float):
        """Setter for the optimizer default learning rate.
        Re-initializes optimizer if it is set."""
        self._optim_kwargs['lr'] = lr
        if self.optimizer is not None:
            self.init(self.optimizer.param_groups)

    # pylint: enable=invalid-name

    @property
    def settings(self) -> Dict:
        """Return nice dict representation of init args and optimizer type."""
        return dict(optim_type=self._optim_type, **self._optim_kwargs,
                    lr_scheduler_type=self._lr_scheduler_type,
                    lr_kwargs=self._lr_kwargs,
                    batch_update_lr=self.batch_update_lr)

    def reset(self):
        """Reset optimizer (and lr scheduler); requires init before next use
        of optimizer."""
        self.lr_scheduler = None
        self.optimizer = None

    def init(self,
             params: Union[Iterable[torch.Tensor],
                           Iterable[Dict[str, torch.Tensor]]]):
        """Initialize optimizer and learning rate scheduler
        with given parameters."""
        self.reset()
        self.optimizer = self._optim_type(params, **self._optim_kwargs)
        if self._lr_scheduler_type is not None:
            self.lr_scheduler = self._lr_scheduler_type(self.optimizer,
                                                        **self._lr_kwargs)

    def __call__(self,
                 params: Union[Iterable[torch.Tensor],
                               Iterable[Dict[str, torch.Tensor]]]
                 ) -> 'ResettableOptimizer':
        """Return a fresh instance of the optimizer with the saved settings.
        Intends to wrap call to ``_optim_type``, such that an instance
        of a resettable optimizer can replace the type of its target optimizer.
        """
        self.init(params)
        return self

    def zero_grad(self):
        """Wrapper around :py:meth:`torch.optim.Optimizer.zero_grad`.
        Only call after :py:meth:`init` or :py:meth:`__call__`."""
        self._check_after_init()
        return self.optimizer.zero_grad()

    def add_param_group(self, param_group):
        """Wrapper around :py:meth:`torch.optim.Optimizer.add_param_group`.
        Only call after :py:meth:`init` or :py:meth:`__call__`."""
        self._check_after_init()
        return self.optimizer.add_param_group(param_group)

    def step(self, closure=None, epoch=None):
        """Step the optimizer and the learning rate scheduler if set."""
        self._check_after_init()
        self.optimizer.step(closure)
        if self._lr_scheduler_type is not None and self.batch_update_lr:
            self.lr_scheduler.step(epoch)

    def epoch_end(self):
        """Update of learning rate scheduler after an epoch.
        Only specific learning rate scheduler require updating after the
        epoch."""
        self._check_after_init()
        if self._lr_scheduler_type is not None and not self.batch_update_lr:
            self.lr_scheduler.step()

    def __repr__(self):
        repr_dict = dict(optimizer_type=self._optim_type.__name__,
                         **self._optim_kwargs)
        if self._lr_scheduler_type is not None:
            repr_dict.update(
                dict(lr_scheduler_type=self._lr_scheduler_type.__name__,
                     lr_kwargs=self._lr_kwargs,
                     batch_update_lr=self.batch_update_lr))
        repr_dict_str = ', '.join(
            ['{}={}'.format(k, v) for k, v in repr_dict.items()])
        return "ResettableOptimizer({})".format(repr_dict_str)

    def __str__(self):
        return repr(self)
