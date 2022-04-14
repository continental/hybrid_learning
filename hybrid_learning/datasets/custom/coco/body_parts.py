"""Enum with definitions of standard body parts from MS COCO keypoint names.
The enum is :py:class:`BodyParts`.
It is supposed to ease the specification of body parts (one does not have
to provide a list with all involved keypoints every time).
"""

#  Copyright (c) 2022 Continental Automotive GmbH
import enum
from typing import Tuple


def _left_right(keypoint_id: str) -> Tuple[str, str]:
    """Yield tuple of names of left and right instances."""
    return 'left_{}'.format(keypoint_id), 'right_{}'.format(keypoint_id)


class BodyParts(tuple, enum.Enum):
    """Mapping of a visual concept description (body part) to COCO keypoint ID
    collections approximating it.
    A body part is considered to be a connected part of a person.
    E.g. a ``"face"`` can be approximated by
    ``("left_eye", "right_eye", "nose")``.
    An arm is either the connection between
    ``("left_wrist, left_elbow, left_shoulder)`` or between
    ``("right_wrist, right_elbow, right_shoulder)``.

    Available keypoints:

    ::

        "nose",
        "left_eye", "right_eye",
        "left_ear", "right_ear",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle"
    """
    # Original single keypoints:
    NOSE = ('nose',)

    LEFT_EYE = ('left_eye',)
    LEFT_EAR = ('left_ear',)
    LEFT_SHOULDER = ('left_shoulder',)
    LEFT_ELBOW = ('left_elbow',)
    LEFT_WRIST = ('left_wrist',)
    LEFT_HAND = LEFT_WRIST
    LEFT_HIP = ('left_hip',)
    LEFT_KNEE = ('left_knee',)
    LEFT_ANKLE = ('left_ankle',)
    LEFT_FOOT = LEFT_ANKLE

    RIGHT_EYE = ('right_eye',)
    RIGHT_EAR = ('right_ear',)
    RIGHT_SHOULDER = ('right_shoulder',)
    RIGHT_ELBOW = ('right_elbow',)
    RIGHT_WRIST = ('right_wrist',)
    RIGHT_HAND = RIGHT_WRIST
    RIGHT_HIP = ('right_hip',)
    RIGHT_KNEE = ('right_knee',)
    RIGHT_ANKLE = ('right_ankle',)
    RIGHT_FOOT = RIGHT_ANKLE

    # (Non-connected) Keypoints pairs:
    # EAR = _left_right('ear')
    # ELBOW = _left_right('elbow')
    # WRIST = _left_right('wrist')
    # HAND = WRIST
    # KNEE = _left_right('knee')
    # ANKLE = _left_right('ankle')
    # FOOT = ANKLE

    # Connected keypoint pairs:
    # EYE = _left_right('eye')  # This is connected!
    SHOULDER = _left_right('shoulder')  # This is connected!
    HIP = _left_right('hip')  # This is connected!

    # Combinations:
    # Head
    FACE = (*LEFT_EYE, *RIGHT_EYE, *NOSE)
    HEAD = (*FACE, *LEFT_EAR, *RIGHT_EAR)

    # Arm
    # UPPER_ARM = (*SHOULDER, *ELBOW)  # This is connected!
    LEFT_UPPER_ARM = (*LEFT_SHOULDER, *LEFT_ELBOW)
    RIGHT_UPPER_ARM = (*RIGHT_SHOULDER, *RIGHT_ELBOW)

    # LOWER_ARM = (*ELBOW, *HAND)
    LEFT_LOWER_ARM = (*LEFT_ELBOW, *LEFT_HAND)
    RIGHT_LOWER_ARM = (*RIGHT_ELBOW, *RIGHT_HAND)

    # ARM = (*UPPER_ARM, *LOWER_ARM)  # This is connected!
    LEFT_ARM = (*LEFT_UPPER_ARM, *LEFT_LOWER_ARM)
    RIGHT_ARM = (*RIGHT_UPPER_ARM, *RIGHT_LOWER_ARM)

    # Leg
    # UPPER_LEG = (*HIP, *KNEE)  # This is connected!
    LEFT_UPPER_LEG = (*LEFT_HIP, *LEFT_KNEE)
    RIGHT_UPPER_LEG = (*RIGHT_HIP, *RIGHT_KNEE)

    # LOWER_LEG = (*KNEE, *FOOT)
    LEFT_LOWER_LEG = (*LEFT_KNEE, *LEFT_FOOT)
    RIGHT_LOWER_LEG = (*RIGHT_KNEE, *RIGHT_FOOT)

    # LEG = (*UPPER_LEG, *LOWER_LEG)  # This is connected!
    LEFT_LEG = (*LEFT_UPPER_LEG, *LEFT_LOWER_LEG)
    RIGHT_LEG = (*RIGHT_UPPER_LEG, *RIGHT_LOWER_LEG)

    # Torso
    TORSO = (*SHOULDER, *HIP)
