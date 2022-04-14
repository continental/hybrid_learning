"""Handle for early stopping."""

#  Copyright (c) 2022 Continental Automotive GmbH

import logging
from typing import Optional, Dict, Any

LOGGER = logging.getLogger(__name__)


class EarlyStoppingHandle:
    """Handle encapsulating early stopping checks.
    It measures the progress given the previous steps and determines whether
    one should proceed or stop early. This value is then stored in require_stop.

    *Usage:* Initialize the handle, then apply after each epoch
    :py:meth:`step` to the loss tensor. :py:meth:`step` will update the
    current best values and returns the new value of :py:attr:`require_stop`.
    """

    def __init__(self, min_delta: float = 0.001, patience: int = 1,
                 verbose: bool = False):
        """Init.

        :param min_delta: minimum decrease of the loss value such that
            it is considered progress.
            If the loss should be increasing, assign corresponding negative
            ``min_delta``
        :param patience: number of steps (epochs) with no progress after
            which to suggest to stop;
            values <=0 are mapped to 1 (i.e. no patience,
            stop if no progress since last step)
        :param verbose: logging verbosity level.
        """
        self.min_delta = min_delta
        """Minimum decrease (increase if <0) of loss value per step to be
        considered a progress."""
        self.patience = patience if patience > 0 else 1
        """Number of steps with no progress after which to suggest to stop."""
        self.verbose = verbose
        """Verbosity level (simple logging switch)."""

        # variables specific to a training phase
        self.require_stop = False
        """Whether the early stopping handle would suggest to stop
        or to proceed. Updated using the :py:meth:`step` method."""
        self._no_major_improvement_since: Optional[int] = None
        """Counter for counting the number of steps without progress."""
        self._curr_best_loss: Optional[int] = None
        """Storage for the current best loss value for comparison.
        Needed to derive whether any progress was made in the sense of a new
        best value."""

    def reset(self):
        """Reset the history of the handle to start at step 0."""
        self._curr_best_loss = None
        self._no_major_improvement_since = None
        self.require_stop = False

    def step(self, loss: float) -> bool:
        """Update stopping flag according to latest epoch results.
        If stopping is required, set :py:attr:`require_stop` to ``True``.

        :param loss: the loss of the last epoch unseen to the handle
        :return: the new value of :py:attr:`require_stop`
        """
        # Init _curr_best_loss at the first step
        if self._curr_best_loss is None:
            self._curr_best_loss = loss
            self.require_stop = False
            self._no_major_improvement_since = 0
            return self.require_stop

        # Was there any improvement according to self.min_delta?
        no_progress: bool = (self._curr_best_loss - loss > self.min_delta) \
            if self.min_delta < 0 else \
            (self._curr_best_loss - loss < self.min_delta)

        # Update epoch counter according to progress
        if self._no_major_improvement_since is not None and no_progress:
            self._no_major_improvement_since += 1  # call model not improved
        else:
            self._no_major_improvement_since = 0  # reset if improvement

        # Save (and log) current best test loss
        if (self.min_delta > 0 and loss < self._curr_best_loss) or \
                (self.min_delta < 0 and loss > self._curr_best_loss):
            self._curr_best_loss = loss

        if self.verbose:
            LOGGER.info("Current best loss %f\tno improvement since %d epochs",
                        self._curr_best_loss, self._no_major_improvement_since)

        # No progress for too long?
        if self._no_major_improvement_since >= self.patience:
            self.require_stop = True
            LOGGER.info(("No improvement greater %f from best result %f "
                         "since %d steps; suggesting to stop."),
                        self.min_delta, self._curr_best_loss,
                        self._no_major_improvement_since)

        return self.require_stop

    @property
    def settings(self) -> Dict[str, Any]:
        """A dict with the settings."""
        return dict(min_delta=self.min_delta, patience=self.patience,
                    verbose=self.verbose)

    def __repr__(self):
        """String representation with all properties."""
        return ("{cls}(min_delta={min_delta}, patience={patience}, "
                "verbose={verbose})"
                .format(cls=self.__class__.__name__, **self.settings))

    def __str__(self):
        return repr(self)
