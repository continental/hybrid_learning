"""Testing functions for the KPI classes and functions."""

#  Copyright (c) 2022 Continental Automotive GmbH
from math import sqrt
from typing import List, Tuple, Sequence, Dict

import numpy as np
# pylint: disable=no-member
# pylint: disable=not-callable
# pylint: disable=no-self-use
import pytest
import torch
from matplotlib import pyplot as plt

from hybrid_learning.concepts.train_eval import kpis
from hybrid_learning.concepts.train_eval.kpis import AbstractIoULike, IoU, \
    BalancedPenaltyReducedFocalLoss
from hybrid_learning.datasets.transforms import BatchIoUEncode2D, Union


def test_get_area_axes():
    """test for get_area_axes method"""
    assert sorted(AbstractIoULike.get_area_axes(torch.ones((3,)))) == [0]
    assert sorted(AbstractIoULike.get_area_axes(torch.ones((3, 2)))) == [0, 1]
    assert (sorted(AbstractIoULike.get_area_axes(torch.ones((3, 2, 4))))
            == [1, 2])
    assert (sorted(AbstractIoULike.get_area_axes(torch.ones((3, 2, 4, 3))))
            == [2, 3])


def test_binarize():
    """test for binarize method"""
    vec: torch.Tensor = torch.ones((3,))
    assert AbstractIoULike.binarize(vec, 0.5).dtype == torch.bool
    assert AbstractIoULike.binarize(vec, 0.5).int().allclose(vec.int())
    assert AbstractIoULike.binarize(vec, 1).int().allclose(
        torch.zeros(vec.size()).int())
    assert AbstractIoULike.binarize(vec, 2).int().allclose(
        torch.zeros(vec.size()).int())


def test_smooth_division():
    """test whether smooth division actually prevents division by 0"""
    assert IoU().smooth_division(0, 0) == 1
    assert kpis.SetIoU().smooth_division(torch.zeros((1,)), torch.zeros((1,))
                                    ).allclose(torch.ones((1,)))
    assert BatchIoUEncode2D(np.array([[1]])).smooth_division(0, 0) == 1


def test_ious():
    """test fundamental properties common to both intersection over union for
    some values"""
    with torch.no_grad():
        for iou in (IoU(),):
            # Does __str__, __repr__ work?
            _, _ = str(iou), repr(iou)
            one: torch.Tensor = torch.tensor((1.,))
            zero: torch.Tensor = \
                torch.tensor((iou.smooth,)) / (4 * one + iou.smooth)
            # If both are equal, 1, if opposite, 0
            assert iou(torch.ones((2, 2)), torch.ones((2, 2))).allclose(one)
            assert iou(torch.ones((2, 2)), torch.zeros((2, 2))).allclose(zero)
            assert iou(torch.zeros((2, 2)), torch.ones((2, 2))).allclose(zero)
            assert iou(torch.zeros((2, 2)), torch.zeros((2, 2))).allclose(one)

            # IoU is symmetric
            batch1 = torch.randint(low=0, high=2, size=(3, 4, 5))
            batch2 = torch.randint(low=0, high=2, size=(3, 4, 5))
            iou_ab, iou_ba = iou(batch1, batch2), iou(batch2, batch1)
            assert iou_ab.allclose(iou_ba)

            # a more complicated pattern:
            up_left = torch.tensor(((1, 0), (1, 1)))
            low_right = torch.tensor(((0, 1), (1, 1)))
            assert iou(up_left, low_right).allclose(torch.tensor((0.5,)))


@pytest.mark.parametrize(
    "alpha,beta,factor_pos,inputs,targets,expected",
    [
        # fully correct
        (2, 4, 0.5, 1, 1, 0),
        (2, 4, 0.5, 0, 0, 0),
        (2, 4, 0.5, [[0]], [[0]], 0),
        (2, 4, 0.5, [[1]], [[1]], 0),
        # negatives slightly incorrect
        (2, 4, 0.5, [0.5, 1], [0, 1], -0.125 * np.log(0.5)),
        # negatives slightly incorrect with balancing towards negatives
        (2, 4, 0, [0.5, 1], [0, 1], -0.25 * np.log(0.5)),
        # negatives slightly incorrect with balancing towards positives
        (2, 4, 1, [0.5, 1], [0, 1], 0),
        # negatives slightly incorrect with penalty reduction
        (2, 4, .5, [0.5, 1], [0.25, 1], - .5 * .75 ** 4 * .5 ** 2 * np.log(.5)),
        (2, 4, 0, [0.5, 1], [0.25, 1], - .75 ** 4 * .5 ** 2 * np.log(.5)),
        (2, 3, 0, [0.5, 1], [0.25, 1], - .75 ** 3 * .5 ** 2 * np.log(.5)),
        (4, 3, 0, [0.5, 1], [0.25, 1], - .75 ** 3 * .5 ** 4 * np.log(.5)),
        # positives slightly incorrect
        (2, 4, 0.5, [0, 0.8], [0, 1], - .5 * .2 ** 2 * np.log(.8)),
        # both slightly incorrect
        (2, 4, 0.5, [0.3, 0.8], [0.25, 1], - .5 * (
                .75 ** 4 * .3 ** 2 * np.log(.7) + .2 ** 2 * np.log(.8))),
        # both slightly incorrect with balancing
        (2, 4, .6, [0.3, 0.8], [0.25, 1],
         - .6 * (.2 ** 2 * np.log(.8)) - .4 * (.75 ** 4 * .3 ** 2 * np.log(.7))
         ),
        # non-zero or -one number of positives
        (2, 4, .5, [0.3, 0.8, 0.8, 0.8], [0.25, 1, 1, 1],
         (1 / 6) * (- 3 * (.2 ** 2 * np.log(.8))
                    - (.75 ** 4 * .3 ** 2 * np.log(.7)))
         )
    ])
def test_focal_loss(alpha: float, beta: float, factor_pos: float,
                    inputs: Union[float, List], targets: Union[float, List],
                    expected: float):
    """Test the balanced, penalty-reduced focal loss from CenterNet."""
    focal_loss = BalancedPenaltyReducedFocalLoss(factor_pos, alpha, beta)
    inp = torch.tensor(inputs).float()
    targ = torch.tensor(targets).float()
    outp = focal_loss(inp, targ)
    exp = torch.tensor(expected).float()
    assert len(outp.size()) == 0
    assert outp.dtype == torch.float
    assert torch.allclose(outp, exp), f"Context: outp: {outp}, exp: {exp}"


@pytest.mark.parametrize('ddof,outputs,mean_res,var_res', [
    # Single values
    (1, (1,), 1, None),
    (0, (1,), 1, 0),
    (1, (-1.,), -1, None),
    (0, (-1.,), -1, 0),
    # Sequence
    (1, ([1, 2, 3, 4],), 10 / 4, 5 / 3),
    (0, ([1, 2, 3, 4],), 10 / 4, 5 / 4),
    (1, ([1, 3, 2, 4],), 10 / 4, 5 / 3),
    (0, ([1, 3, 2, 4],), 10 / 4, 5 / 4),
    # Other shapes
    (1, ([1, 2], [3, 4]), 10 / 4, 5 / 3),
    (0, ([1, 2], [3, 4]), 10 / 4, 5 / 4),
    (1, ([[1, 2], [3, 4]]), 10 / 4, 5 / 3),
    (0, ([[1, 2], [3, 4]]), 10 / 4, 5 / 4),
    # Negative values & other ddofs
    (4, ([1.5, -1.5, 2, -2],), 0, None),
    (3, ([1.5, -1.5, 2, -2],), 0, 12.5 / 1),
    (2, ([1.5, -1.5, 2, -2],), 0, 12.5 / 2),
    (1, ([1.5, -1.5, 2, -2],), 0, 12.5 / 3),
    (0, ([1.5, -1.5, 2, -2],), 0, 12.5 / 4),
])
def test_statistics(outputs: Tuple, mean_res: float, var_res: float, ddof: int):
    statistics_kpis = dict(mean=kpis.Mean(), variance=kpis.Variance(ddof=ddof), std_dev=kpis.StandardDev(ddof=ddof))
    statistics_vals = dict(mean=mean_res, variance=var_res, std_dev=sqrt(var_res) if var_res is not None else None)
    for name, kpi in statistics_kpis.items():
        assert kpi.value() is None, "Statistics KPI {} wrongly initialized".format(name)
        for outp in outputs:
            kpi.update(torch.tensor(outp))
        kpi_final_value = kpi.value()
        if statistics_vals[name] is None:
            assert kpi_final_value is None
        else:
            assert np.allclose(kpi_final_value, statistics_vals[name]), \
                ("Got wrong result for {}:\nvalue(): {}\n{}"
                 .format(name, kpi_final_value, {k: v for k, v in kpi.__dict__.items() if not k.startswith('_')}))


@pytest.mark.parametrize('outputs,max_res,min_res', [
    ((0,), 0, 0), ((1,), 1, 1), ((-1.,), -1, -1),
    (([1, 2, 3, 4],), 4, 1),
    (([0, 5, 2, 4],), 5, 0),
    (([1.5, -1.5, 3.5, -2],), 3.5, -2),
    (([1, 2], [3, 4]), 4, 1),
    (([[1, 2], [3, 4]]), 4, 1),
])
def test_min_max(outputs: Tuple, max_res: float, min_res: float):
    max_kpi = kpis.Maximum()
    min_kpi = kpis.Minimum()
    assert max_kpi.value() is None
    assert min_kpi.value() is None
    for outp in outputs:
        max_kpi.update(torch.tensor(outp))
        min_kpi.update(torch.tensor(outp))
    assert np.allclose(max_kpi.value(), max_res), \
        ("Got wrong max result:\nExpected: {}\nGot: {}".format(max_res, max_kpi.value()))
    assert np.allclose(min_kpi.value(), min_res), \
        ("Got wrong min result:\nExpected: {}\nGot: {}".format(min_res, min_kpi.value()))


@pytest.mark.parametrize('lower_bound,upper_bound,values,dist', [
    (0, 1, [1], [1]), (0, 1, [0], [1]), (0, 1, [0.5], [1]),
    (0, 1, [0, 0, 1], [2, 1]),
    (0, None, [0, 0, 1, 1.5, 1.2, 2, 2, 2, 2.3, 2.5, 3], [2, 3, 5, 1]),
])
def test_histogram(lower_bound: float, upper_bound: float, values: Sequence[float], dist: Sequence[int]):
    hist = kpis.Histogram(len(dist), lower_bound=lower_bound,
                          upper_bound=upper_bound or len(dist))
    for val in values:
        hist(torch.tensor(val), None)
    assert hist.count.int().numpy().tolist() == np.array(dist).tolist()
    fig = hist.value()
    assert isinstance(fig, plt.Figure)
    assert len(fig.get_axes()) == 1
    ax = fig.get_axes()[0]
    assert len(ax.patches) == len(dist)
    assert [patch.get_height() for patch in ax.patches] == dist


@pytest.mark.parametrize('ddof,n_bins,values,approx_mean,approx_var', [
    # Just one bin (=all values treated as 0.5): same as standard statistics
    (0, 1, [.5, .5], .5, 0),
    (0, 1, [.75, .25], .5, 0),
    (1, 1, (1,), .5, None),
    (0, 1, (1,), .5, 0),
    # Several bins but just one used
    (0, 4, [0, 0, 0.1, 0.2], .125, 0),
    # Sequence
    (1, 4, [.1, .8, .2, .9], .5, 4 * .375 ** 2 / 3),
    (0, 4, [.1, .8, .2, .9], .5, 4 * .375 ** 2 / 4),
    # Other shapes & ddofs
    (2, 10, [[.15, .25], [.35, .45]], .3, 2 * (.15 ** 2 + .05 ** 2) / 2),
    (1, 10, [[.15, .25], [.35, .45]], .3, 2 * (.15 ** 2 + .05 ** 2) / 3),
    (0, 10, [[.15, .25], [.35, .45]], .3, 2 * (.15 ** 2 + .05 ** 2) / 4),
    # Other
    (0, 2, [0, 1], .5, .125 / 2),
    (1, 2, [0, 1], .5, .125),
    (0, 2, [0, 0.25, 0.75, 1], .5, .25 / 4),
    (1, 2, [0, 1], .5, .125),
])
def test_approximate_std(ddof: int, n_bins: int, values: Sequence[float], approx_mean: float, approx_var: float):
    approx_kpis: Dict[str, kpis.ApproximateMean] = dict(
        mean=kpis.ApproximateMean(n_bins=n_bins),
        variance=kpis.ApproximateVariance(ddof=ddof, n_bins=n_bins),
        stddev=kpis.ApproximateStdDev(ddof=ddof, n_bins=n_bins))
    approx_res: Dict[str, float] = dict(
        mean=approx_mean, variance=approx_var,
        stddev=sqrt(approx_var) if approx_var is not None else None)
    for name, kpi in approx_kpis.items():
        assert kpi.value() is None, "Wrong init for {}".format(name)
        for val in values:
            kpi(torch.tensor(val), None)
        final_value = kpi.value()
        if approx_res[name] is None:
            assert final_value is None, "Expected None for {} but got {}".format(name, final_value)
        else:
            assert np.allclose(final_value.numpy(), approx_res[name]), \
                ("Got wrong result for {}:\nExpected: {}\nvalue(): {}\n{}"
                 .format(name, approx_res[name], final_value,
                         {k: v for k, v in kpi.__dict__.items() if not k.startswith('_')}))
