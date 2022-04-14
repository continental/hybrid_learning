"""Tests for person size estimation from keypoint information."""

#  Copyright (c) 2022 Continental Automotive GmbH

# pylint: disable=no-self-use

from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import pytest

# noinspection PyProtectedMember
from hybrid_learning.datasets.custom.person_size_estimation import \
    _length_of_joint_part, _lengths_of_long_bones, lengths_to_body_size, \
    keypoints_to_lengths


class TestKeypointProcessing:
    """Tests for the keypoint processing functionalities."""

    def test_len_of_joint_part(self):
        """Test for the helper function _length_of_joint_part()."""
        # Test value tuples of the form
        # (slope1, intersect1), (slope2, intersect2), len1, result_len2
        test_vals: List[Tuple[Tuple[float, float], Tuple[float, float],
                              float, float]] = [
            ((2, 0), (1, 0), 1, 2),
            ((1, 0), (2, 0), 2, 1),
            ((2, 1), (1, 0), 1, 3),
            ((2, 1), (1, 1), 1, 2),
        ]

        for p1_params, p2_params, len_p1, expected in test_vals:
            factors = pd.DataFrame({'p1': p1_params, 'p2': p2_params},
                                   index=['slope', 'intersect'])
            lengths = {'p1': len_p1}
            res = _length_of_joint_part(*sorted(factors.columns),
                                        factors=factors, lengths=lengths)
            assert res == expected, \
                ("Context:\np1 {}\np2 {}\nlen(p1) {}\nexpected: {}\nresult: {}"
                 .format(p1_params, p2_params, len_p1, expected, res))

    def test_lengths_of_long_bones(self):
        """Test for the helper function _lengths_of_long_bones()."""
        kpts = {
            'left_wrist': np.array([0, 0]),
            'left_elbow': np.array([0, 1]),
            'left_shoulder': np.array([0, 1.75]),
            'right_wrist': np.array([0.5, 0]),
            'right_elbow': np.array([1, 0]),
        }

        res = _lengths_of_long_bones(kpts=kpts)
        assert res['lower_arm'] == 1
        assert res['upper_arm'] == 0.75
        assert res['lower_leg'] is None

    def test_lengths_to_body_sizes(self):
        """Test for helper function lengths_to_body_sizes()."""
        factors = {"p1": {'slope': 2, 'intersect': 1},
                   "p2": {'slope': 1, 'intersect': 2}}
        lengths = {"p1": 1}

        # Raise if intersect >= assumed_height
        with pytest.raises(ValueError):
            res = lengths_to_body_size(factors=factors, lengths=lengths,
                                       assumed_height=1)
        res = lengths_to_body_size(factors=factors, lengths=lengths,
                                   assumed_height=2)

        for key in factors:
            assert key in res
        assert res["p1"] == 4
        assert res["p2"] is None

    def test_keypoints_to_lengths(self):
        """Test helper function keypoints_to_lengths()."""
        kpts = {
            'left_wrist': np.array([0, 0]),
            'left_elbow': np.array([0, 1]),
            'left_shoulder': np.array([0, 1.75]),
            'right_wrist': np.array([0.5, 0]),
            'right_elbow': np.array([1, 0]),
        }

        res: Dict[str, float] = keypoints_to_lengths(kpts)

        for k in ('hip_to_shoulder', 'arm', 'leg', 'shoulder_width',
                  'wrist_to_wrist', 'body_height', 'head_width',
                  'head_height', 'upper_arm', 'lower_arm',
                  'upper_leg', 'lower_leg'):
            assert k in res

        assert res['arm'] == 1.75
        assert res['lower_arm'] == 1
        assert res['upper_arm'] == 0.75
        for k in res:
            if k not in ['arm', 'lower_arm', 'upper_arm']:
                assert res[k] is None, \
                    "length {} is not None but {}".format(k, res[k])
