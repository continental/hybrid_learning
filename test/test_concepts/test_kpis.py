"""Testing functions for the KPI classes and functions."""

#  Copyright (c) 2020 Continental Automotive GmbH

import numpy as np
# pylint: disable=no-member
# pylint: disable=not-callable
# pylint: disable=no-self-use
import torch

from hybrid_learning.concepts.kpis import AbstractIoULike, IoU, SetIoU
from hybrid_learning.datasets.transforms import BatchIoUEncode2D


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
    assert AbstractIoULike.binarize(vec, 0.5).dtype == torch.int
    assert AbstractIoULike.binarize(vec, 0.5).allclose(vec.int())
    assert AbstractIoULike.binarize(vec, 1).allclose(
        torch.zeros(vec.size()).int())
    assert AbstractIoULike.binarize(vec, 2).allclose(
        torch.zeros(vec.size()).int())


def test_smooth_division():
    """test whether smooth division actually prevents division by 0"""
    assert IoU().smooth_division(0, 0) == 1
    assert SetIoU().smooth_division(torch.zeros((1,)), torch.zeros((1,))
                                    ).allclose(torch.ones((1,)))
    assert BatchIoUEncode2D(np.array([[1]])).smooth_division(0, 0) == 1


def test_ious():
    """test fundamental properties common to both intersection over union for
    some values"""
    with torch.no_grad():
        for iou in (IoU(), SetIoU()):
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
