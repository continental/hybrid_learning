#  Copyright (c) 2022 Continental Automotive GmbH
"""Testing routines for the generic model wrapper classes."""

# pylint: disable=no-self-use

from hybrid_learning.concepts.train_eval import EarlyStoppingHandle


class TestEarlyStopping:
    """Quick test of the early stopper class."""

    def test_patience(self):
        """Does the stopper stop if no progress is made at all for <patience>
        steps?"""
        early_stopper = EarlyStoppingHandle(patience=3)
        assert not early_stopper.step(0) and not early_stopper.require_stop  # 0
        assert not early_stopper.step(0) and not early_stopper.require_stop  # 1
        assert not early_stopper.step(0) and not early_stopper.require_stop  # 2
        assert early_stopper.step(0) and early_stopper.require_stop  # 3

    def test_neg_min_delta(self):
        """Test positive and negative min_delta."""
        early_stopper = EarlyStoppingHandle(min_delta=1, verbose=True)
        assert not early_stopper.step(0)
        assert not early_stopper.step(-1)
        assert not early_stopper.step(-2)
        assert early_stopper.step(-0.9)

        early_stopper = EarlyStoppingHandle(min_delta=-1, verbose=True)
        assert not early_stopper.step(0)
        assert not early_stopper.step(1)
        assert not early_stopper.step(2)
        assert early_stopper.step(0.9)

    def test_reset(self):
        """Does the reset really provide new start?"""
        early_stopper = EarlyStoppingHandle(patience=2)
        assert not early_stopper.step(0) and not early_stopper.require_stop  # 0
        assert not early_stopper.step(0) and not early_stopper.require_stop  # 1
        assert early_stopper.step(0) and early_stopper.require_stop  # 2

        early_stopper.reset()
        assert not early_stopper.require_stop
        assert not early_stopper.step(0) and not early_stopper.require_stop  # 0
        assert not early_stopper.step(0) and not early_stopper.require_stop  # 1
        assert early_stopper.step(0) and early_stopper.require_stop  # 2
