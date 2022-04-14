"""Util functions to estimate the size of a person in an image from
keypoint information.
The core function is :py:func:`keypoints_to_lengths`.
"""

#  Copyright (c) 2022 Continental Automotive GmbH
from typing import Mapping, Union, Optional, Dict

import numpy as np
import pandas as pd

FACTORS: Mapping[str, Mapping[str, float]] = pd.DataFrame({
    'bbox_width': (1, 0), 'bbox_height': (1, 0),  # fallback
    'wrist_to_wrist': (1.1, 0),  # schooling material
    'hip_to_shoulder': (2.4, 0),  # estimate
    'body_height': (1.1, 0),  # estimate (to add missing forehead)
    'head_height': (7, 0),  # arts
    'upper_leg': (2.77, 0.4050),  # paper
    'lower_leg': ((3.13 + 3.02) / 2, (0.5011 + 0.5011) / 2),  # paper
    'leg': ((1.49 + 1.48) / 2, (0.4353 + 0.4300) / 2),  # paper
    'upper_arm': (3.72, 0.4486),  # paper
    'lower_arm': (4.46, 0.5694),  # paper
}, index=['slope', 'intersect'])
r"""A collection of parameters for linear relations each between an anatomic
size and the full body size (both in meters).
The formulas are of the form
:math:`h(l) = l\cdot s + c` with
slope :math:`s` and intersect :math:`c`,
:math:`l` the value of an anatomic size, and
:math:`h(l)` the full body size estimated from the value of :math:`l`
(all values in meters).
The format is a mapping of the form
``{anatomic_size_id: {'slope': float, 'intersect': float}``.

Sources for the estimates:

- ``bbox_width``, ``bbox_height``:
  This is considered the fallback
  (a person is at least as large as its 2D bounding box width or height)
- ``wrist_to_wrist``:
  This is an estimate of how to calculate the arm span (including hands)
  from the wrist-to-wrist span.
  `Schooling material <https://www.scientificamerican.com/article/human-body-ratios/>`_
  suggests that a person at any age is about as tall as its arm span.
- ``hip_to_shoulder``: this is an estimate
- ``body_height``:
  This value is assumed to cut off forehead and feet.
  The formula estimates how to re-add these parts.
- ``head_height``:
  In `arts <https://www.makingcomics.com/2014/01/19/standard-proportions-human-body/>`_
  a typical estimate of the full body size is about 7 to 8 times the head
  height. To not drastically overestimate the size of children, the lower
  bound is taken.
- ``upper_leg``, ``lower_leg``, ``leg``, ``upper_arm``, ``lower_arm``:
  These values are means over genders and populations taken from
  the up-to-date measurements in the paper [Ruff2012]_

See also `https://en.wikipedia.org/wiki/Estimation_of_stature`_.

.. Ruff, Christopher B., Brigitte M. Holt, Markku Niskanen, Vladimir Sladék,
    Margit Berner, Evan Garofalo, Heather M. Garvin, et al. 2012.
    “Stature and Body Mass Estimation from Skeletal Remains in the
    European Holocene.”
    American Journal of Physical Anthropology 148 (4): 601–17.
    https://doi.org/10.1002/ajpa.22087.

"""
# pylint: enable=line-too-long


def _length_of_links(*joints: str, kpts: Mapping[str, np.ndarray],
                     includes: Mapping[str, Union[bool, int]] = None
                     ) -> Optional[float]:
    """Calculate the maximum length of the path described by the list of joints.
    The joints to consider and in which order is given by ``joints``,
    the actual keypoint x-y-coordinates of each joint must be stored in
    ``kpts``.

    In case any of the joint names contains a ``{}``, two lengths are
    calculated: One where all occurrences of ``{}`` are replaced by ``left``
    and one where they are replaced by ``right``;
    the maximum of both lengths is returned.

    Before length evaluation, it is checked whether the coordinates of all
    required joints are available from ``kpts`` and are marked as known
    in ``includes``. If not, ``None`` is returned and the ``kpts`` are ignored.

    *Usage example*

    To estimate the length of an arm from skeleton information,
    call:

    .. code-block:: python
        _length_of_links('{}_elbow', '{}_wrist', '{}_shoulder',
                      kpts=kpts, includes=includes)

    Under the assumption that both arms are the same length and arms appear
    shorter due to projection to 2D, we want the maximum length of left and
    right arm from skeleton information.
    Assume the wrist of the right arm is not in the image anymore, so its
    keypoint information is unknown. The the path length of the left arm is
    calculated and returned.

    :meta: public

    :param joints: the identifier names of the joints to consider,
        their order describing the path to measure;
        names may contain a ``{}``, which is then replaced by ``left``
        respectively ``right`` during evaluation (see above)
    :param kpts: a mapping of joint identifier to its x-y-coordinate
    :param includes: a mapping of joint identifier to a truth value stating
        whether the joint coordinate is known;
        may be e.g. a mapping of identifier to MS COCO visibility value;
        joints not mentioned are assumed to be known
    :return: the maximum length of a path described by the list of ``joints``
        and their ``kpts`` if all joints and their keypoints are known,
        else ``None``

    :meta public:
    """
    # Instantiate "left" and "right" versions if necessary:
    joints_list = [joints]
    if any(["{}" in joint for joint in joints]):
        joints_list = [[joint.format(lr) if "{}" in joint else joint
                        for joint in joints]
                       for lr in ("left", "right")]

    # Extend includes information to all mentioned joints
    # (joint is included if its keypoint is known and it is not explicitly
    # marked as excluded)
    includes = includes if includes is not None else {}
    includes = {jnt: (jnt in kpts and (jnt not in includes or includes[jnt]))
                for joints in joints_list for jnt in joints}

    # Can any path length be calculated?
    if not any([all([includes[jnt] for jnt in jnts]) for jnts in joints_list]):
        return None

    tot_lens = []
    for joints in joints_list:
        if all([includes[name] for name in joints]):
            links = list(zip(joints, joints[1:]))
            tot_lens.append(sum([np.linalg.norm(kpts[link[1]] - kpts[link[0]])
                                 for link in links]))
    return max([0, *tot_lens])


def _length_of_joint_part(given_part: str, missing_part: str,
                          factors: Mapping[str, Mapping[str, float]],
                          lengths: Mapping[str, float]
                          ) -> Optional[float]:
    r"""Get the length of a joint part from the length of another joint part.
    The assumption is that

    - formulas are given for the total body height deduced from parts
      ``given_part`` :math:`p_1` and
      ``missing_part`` :math:`p_2`
      (e.g. lower and upper leg), and
    - the length of ``given_part`` is known.

    The formulas are assumed to be of the form

    .. math::

        s_1 \cdot len(p_1) + c_1
        = \text{body\_height} =
        s_2 \cdot len(p_2) + c_2 \\

    for slope constants :math:`s_i \in R` (``factors[i].slope``) and
    intersect constants :math:`c_i \in R` (``factors[i].intersect``).

    So the formula is:

    .. math:: len(p_2) = \frac{s_1}{s_2} \cdot len(p_1) + (c_1 - c_2)

    :meta: public

    :param given_part: identifier name of the part for which the length is
        given
    :param missing_part: identifier name of the part for which the length is
        missing
    :param factors: a mapping holding the formula parameters in the form
        ``{part_id: {'slope': float, 'intersect': float}}``
    :param lengths: a mapping holding the given lengths in the form
        ``{part_id: length}``
    :return: the length of the ``missing_part`` if the length of ``given_part``
        is known, else ``None``
    :raises: if the length of ``given_part`` is known but one of the
        formula parameter information for ``given_part`` or ``missing_part``
        is missing from ``factors``
    """
    if given_part not in lengths or not lengths[given_part]:
        return None

    return (((factors[given_part]['slope'] / factors[missing_part]['slope'])
             * lengths[given_part])
            + (factors[given_part]['intersect']
               - factors[missing_part]['intersect']))


def _lengths_of_long_bones(kpts: Mapping[str, np.ndarray],
                           includes: Mapping[str, Union[bool, int]] = None
                           ) -> Dict[str, float]:
    """Collect length estimates for the long bones of a skeleton.
    The length of each bone is estimated using :py:func:`_length_of_links`.
    So the length of bones for which joint information is missing from
    ``kpts`` or marked as unknown in ``includes`` is set to ``None``.

    Used keypoint names (prefixed by ``left_`` and ``right_``):
    ankle, knee, hip, shoulder, elbow, wrist

    Calculated long bones:
    upper_leg, lower_leg, upper_arm, lower_arm

    :meta: public

    :param kpts: a mapping of joint identifier to its x-y-coordinate
    :param includes: a mapping of joint identifier to a bool stating whether
        keypoint information about this joint is known or the joint should be
        excluded; see :py:func:`_length_of_links`
    :return: a :py:class:`pandas.Series` with the long bone names as
        index names and their estimated length as values (resp. ``None`` if
        the length could not be estimated)

    :meta public:
    """
    lens: Dict[str, float] = {}
    # SINGLE LONG BONES and BACK
    for bone, joints in (('upper_leg', ('{}_knee', '{}_hip')),
                         ('lower_leg', ('{}_knee', '{}_ankle')),
                         ('upper_arm', ('{}_elbow', '{}_shoulder')),
                         ('lower_arm', ('{}_elbow', '{}_wrist')),
                         ):
        lens[bone] = _length_of_links(*joints, kpts=kpts, includes=includes)
    return lens


def lengths_to_body_size(lengths: Mapping[str, float],
                         factors=FACTORS,
                         assumed_height: float = 1.7):
    r"""Estimate the body size in pixels given anatomical ``lengths`` in pixels,
    the ``assumed_height`` and formula ``factors`` in meters.

    Assume a person of height :math:`h = \text{assumed\_height}` in meters
    is scaled by a factor :math:`f` to a height of :math:`h'` pixels.
    The following is given:

    - From ``factors``:
      The formula :math:`h(l) = l \cdot s + c` to calculate the total body size
      in meters of a person from an anatomical size :math:`l` in meters
      with given slope :math:`s` and intersect :math:`c`.
    - From ``lengths``:
      The value :math:`l' = f \cdot l` of the anatomical size :math:`l` scaled
      by :math:`f`.
    - The value of :math:`h(l)` (in meters) is assumed to be ``assumed_height``.

    The goal is to find the scaled size :math:`f\cdot h(l)`.
    This is given by the formula

    .. math::
        f \cdot h(l)
        = f\cdot (l \cdot s + c)
        = l'\cdot s + f \cdot c \\
        \Rightarrow f = \frac{l'}{h(l) - c} \\
        \Rightarrow f \cdot h(l) = l' \cdot \frac{h(l)}{h(l) - c}


    :param lengths: mapping of anatomic length identifiers to floats
    :param factors: mapping of anatomic length identifier to parameters for
        linear formula to calculate the body size in meters
    :param assumed_height: the assumed height of the person in meters
    :return: a mapping of anatomic length to body height estimated from its
        value and formula;
        if an anatomic length is not given (missing from lengths or ``None``),
        its total size return value is ``None``
    """
    lengths = {key: lengths[key] if key in lengths else None for key in factors}

    # region Value check: avoid division by 0 and negative results
    for key in factors:
        if lengths[key] is not None and \
                factors[key]['intersect'] >= assumed_height:
            raise ValueError(
                ("assumed_height {} smaller than the intersect {} for anatomic "
                 "length of {}").format(assumed_height,
                                        factors[key]['intersect'], key))
    # endregion

    scaling_factors = {
        key: ((assumed_height / (assumed_height - factors[key]['intersect']))
              * factors[key]['slope'])
        if lengths[key] else None
        for key in factors}

    return {key: (lengths[key] * scaling_factors[key] if lengths[key] else None)
            for key in factors}


def keypoints_to_lengths(kpts: Mapping[str, np.ndarray],
                         includes: Mapping[str, Union[bool, int]] = None,
                         factors=FACTORS) -> Dict[str, float]:
    # pylint: disable=line-too-long
    r"""Estimate different skeletal lengths from given 2D joint coordinates.
    Due to 2D projection, the maximum estimate is returned for any body part.
    Joints may be marked as excluded from calculation by mapping their bool
    value to ``False`` or 0 in ``includes`` (e.g. use to exclude keypoints
    which are not in an image).

    To infer the length of body parts for which information is missing,
    the given ``factors`` and further relations listed below are used.
    It is assumed that ``factors`` represents parameters to linearly estimate
    the same size (not necessarily the body size) from different body part
    sizes. Further used approximate relations from arts best-practice:

    - head width
      (see https://www.makingcomics.com/2014/01/19/standard-proportions-human-body/):

      - head width ~ 5 eye widths
      - the space between 2 eyes ~ 1 eye width

    - head depth (including nose):

      - head depth ~ :math:`2\times` ear-to-eye
        (see https://www.artistsnetwork.com/art-mediums/drawing-human-head/)
      - 4/7 head depth ~ ear-to-nose
        (see https://www.artyfactory.com/portraits/pencil-portraits/proportions-of-a-head.html)

    - head height
      (see https://www.artyfactory.com/portraits/pencil-portraits/proportions-of-a-head.html):

      - head height ~ :math:`\frac{3}{2}` head width
        (however, height width is often overestimated due to the quality of ear
        keypoints, therefore a factor of 1.1 is taken instead)
      - head height ~ :math:`\frac{8}{7}` head depth (including nose)

    *Estimated lengths* (index of returned :py:class:`pandas.Series`):
    long bones (see :py:func:`_lengths_of_long_bones`),
    hip_to_shoulder, arm, leg, shoulder_width, wrist_to_wrist, body_height,
    head_width, head_height

    *Assumed given joint names:*
    left_ear, right_ear, left_eye, right_eye, nose, and
    those needed by :py:func:`_lengths_of_long_bones`

    :param kpts: mapping of joint identifier to x-y-coordinate
    :param includes: mapping of joint identifier to bool stating whether to
        assume the joint information as known and include the joint in
        calculations; see :py:func:`_length_of_links`
    :param factors: parameter information of the same form as
        :py:data:`FACTORS`;
        it is not required that the linear formulas represented there calculate
        the body height, but they must all calculate the same size estimate
    :return: a :py:class:`pandas.Series` with the length identifier as
        index names and the float lengths as values
    """

    # pylint: enable=line-too-long

    def len_of(*jnts: str):
        """Estimate the length of a path described by the given joints from the
        known keypoint and visibility information."""
        return _length_of_links(*jnts, kpts=kpts, includes=includes) or 0

    def len_by_for(given_part: str, missing_part: str):
        """Estimate the unknown length of a path from the length of another
        path and their formulas for estimating the total body height."""
        return _length_of_joint_part(given_part, missing_part,
                                     factors=factors, lengths=lens) or 0

    # PARTS DIRECTLY DERIVED FROM SKELETON
    # ------------------------------------

    # LONG BONES
    lens: Dict[str, float] = \
        _lengths_of_long_bones(kpts=kpts, includes=includes)
    # BACK
    lens['hip_to_shoulder'] = len_of('{}_hip', '{}_shoulder') or None
    # ARM & LEG
    lens['arm'] = max([lens['lower_arm'] + lens['upper_arm']
                       if (lens['lower_arm'] and lens['upper_arm']) else 0,
                       len_by_for('upper_arm', 'lower_arm'),
                       len_by_for('lower_arm', 'upper_arm')]) or None
    lens['leg'] = max([lens['upper_leg'] + lens['lower_leg']
                       if (lens['upper_leg'] and lens['lower_leg']) else 0,
                       len_by_for('upper_leg', 'lower_leg'),
                       len_by_for('lower_leg', 'upper_leg')]) or None
    # SHOULDER WIDTH & ARM WIDTH
    lens['shoulder_width'] = len_of('left_shoulder', 'right_shoulder') or None
    lens['wrist_to_wrist'] = 2 * lens['arm'] + lens['shoulder_width'] \
        if (lens['arm'] and lens['shoulder_width']) else None
    # BODY HEIGHT
    lens['body_height'] = max([
        len_of('{}_ankle', '{}_knee', '{}_hip', '{}_shoulder', '{}_ear'),
        len_of('{}_ankle', '{}_knee', '{}_hip', '{}_shoulder', '{}_eye'),
        len_of('{}_ankle', '{}_knee', '{}_hip', '{}_shoulder', 'nose')
    ]) or None

    # PARTS DERIVED FROM OTHER SKELETON PARTS
    # ---------------------------------------
    # arts: head portrait
    # - head width ~ 5 eyes; space between 2 eyes ~ 1 eye
    lens['head_width'] = max([len_of('left_ear', 'right_ear'),
                              len_of('left_ear', 'right_eye'),
                              len_of('right_ear', 'left_ear'),
                              2.5 * len_of('left_eye', 'right_eye')]) or None
    # arts: head profile (head depth including nose)
    # - head depth ~ 2*ear-to-eye
    # - 4/7 head depth ~ ear-to-nose
    lens['head_depth'] = max([2 * len_of('{}_ear', '{}_eye'),
                              1.75 * len_of('{}_ear', 'nose')]) or None
    # arts:
    # - head height ~ 3/2 head width
    #   (however, width is often overestimated due to the quality of ear kpts)
    # - head height ~ 8/7 head depth (including nose)
    lens['head_height'] = max([1.1 * (lens['head_width'] or 0),
                               8 / 7 * (lens['head_depth'] or 0)]) or None
    return lens
