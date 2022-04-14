"""Util functions to process, transform, and evaluate MS COCO keypoint
information.

The main functionality is:

- annotation transformation:
  :py:func:`pad_and_scale_annotation`,
  :py:func:`annotations_to_mask`
- estimation of the size (in pixels) of a person in an image from an annotation:
  :py:func:`annotation_to_sizes`, :py:func:`annotation_to_tot_height`
- some functions that may serve as filtering conditions:
  :py:func:`person_has_rel_size`,
  :py:func:`all_keypts_visible`,
  :py:func:`any_keypts_visible`
"""
#  Copyright (c) 2022 Continental Automotive GmbH
import math
from typing import Tuple, List, Sequence, Iterable, Dict, Any, Mapping, \
    Optional, Union

import PIL.Image
import PIL.ImageDraw
import numpy as np
import pandas as pd
import pycocotools.mask
# format: {ID: (slope, intercept)};
from pycocotools.coco import COCO

from ..person_size_estimation import FACTORS, \
    lengths_to_body_size, keypoints_to_lengths
from ...base import add_gaussian_peak
from ...transforms.image_transforms import padding_for_ratio

COCO_STD_KEYPOINT_NAMES: Tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle"
)
"""The standard list of keypoint names for MS COCO.
These are needed to translate a raw list of keypoint information of the
form ``[x1,y1,v1, x2,y2,v2, ...]`` into a dictionary of
``{keypoint_name: keypoint_info}``.
"""


# ---------------------------------
# COCO specific keypoint handling
# ---------------------------------


def _ann_to_kpts_info(annotation: Dict[str, Any]
                      ) -> Tuple[Tuple[int], Tuple[int], Tuple[int]]:
    """Extract the keypoint information from the annotations.

    :param annotation: a COCO style annotation
    :return: a tuple of 3 lists:

        - x-coordinate: pixels from upper left corner
        - y-coordinate: pixels from upper left corner
        - visibility: 0=not marked, 1=marked but occluded, 2=visible

        The index in the list is the same as that of the keypoint name in the
        ``keypoints`` list specified in the COCO annotations.
    """
    kpt_xs, kpt_ys, visibilities = \
        tuple(zip(*np.array_split(
            # kpts in format [x1,y1,v1,x2,..]:
            np.array(annotation['keypoints']),
            # number of kpts:
            indices_or_sections=len(annotation['keypoints']) // 3)))
    return kpt_xs, kpt_ys, visibilities


def annotation_to_sizes(
        annotation: Dict[str, Any],
        all_keypoint_names: Sequence[str] = COCO_STD_KEYPOINT_NAMES,
        assumed_height=1.7,
        factors=FACTORS
) -> Mapping[str, Mapping[str, float]]:
    # pylint: disable=line-too-long
    """Estimate the body size of a person in an image in
    pixels from skeletal keypoints and linear formulas given by ``factors``.

    The ``factors`` holds parameters of linear functions that each calculate
    an estimate of the physical body height of a person in meters from
    a given anatomical size in meters (see
    :py:data:`hybrid_learning.datasets.custom.person_size_estimation.FACTORS`).
    Assuming a real height in meters of ``assumed_height``, missing sizes
    are inferred using these linear relations.

    :param annotation: MS COCO style annotation dict for a single instance
        holding skeletal keypoint information
    :param all_keypoint_names: list of names for each keypoint occurring in the
        annotation in order
    :param assumed_height: the assumed total height of the person in meters
    :param factors: see parameters for linear relations between anatomic
        sizes and total body height in meters as in
        :py:data:`hybrid_learning.datasets.custom.person_size_estimation.FACTORS`
    :return: a :py:class:`pandas.DataFrame` representing a mapping of anatomic
        size identifiers (columns) each to a mapping with keys (index)

        - ``'len'``: the value of the anatomic size in pixels
        - ``'tot_height'``: the value of the total body size of the person
          in pixels assuming a real body height of ``assumed_height`` in meters
          and estimated from the one anatomic size using relation from
          ``factors``
    """
    # pylint: enable=line-too-long
    # Collect named keypoint and visibility information:
    kpt_xs, kpt_ys, visibilities = _ann_to_kpts_info(annotation)
    named_kpts: Dict[str, np.ndarray] = {
        all_keypoint_names[i]: np.array([kpt_xs[i], kpt_ys[i]])
        for i in range(len(all_keypoint_names))
    }
    named_vis: Dict[str, int] = dict(zip(all_keypoint_names, visibilities))

    # Collect anatomic lengths in pixels:
    lengths: Mapping[str, float] = \
        keypoints_to_lengths(named_kpts, named_vis, factors=factors)
    lengths['bbox_width'], lengths['bbox_height'] = annotation['bbox'][2:]

    # Collect body height information:
    return pd.DataFrame({
        'len': lengths,
        'tot_height': lengths_to_body_size(lengths, factors=factors,
                                           assumed_height=assumed_height)
    }).transpose().infer_objects()


def annotation_to_tot_height(
        annotation: Dict[str, Any],
        all_keypoint_names: Sequence[str] = COCO_STD_KEYPOINT_NAMES,
        assumed_height=1.7,
        factors=FACTORS,
        use_bbox_info: bool = False,
) -> Optional[float]:
    """Wrapper around :py:func:`annotation_to_sizes` that returns the cleaned
    maximum total height.

    :param annotation: the annotation to estimate the total person height of
    :param all_keypoint_names: see :py:func:`annotation_to_sizes`
    :param assumed_height: see :py:func:`annotation_to_sizes`
    :param factors: see :py:func:`annotation_to_sizes`
    :param use_bbox_info: whether to use the bounding box width and height
        as fallback estimations of the person height;
        only useful for very small sizes, as otherwise close-ups of body parts
        with too few annotated keypoints may result in wrong classification
    :return: the maximum estimated total height of the person described by the
        annotation instance;
        if size cannot be estimated, ``None`` is returned
    """
    sizes = annotation_to_sizes(annotation=annotation,
                                all_keypoint_names=all_keypoint_names,
                                assumed_height=assumed_height,
                                factors=factors)
    tot_heights = [sizes[by_part]['tot_height'] for by_part in sizes.keys()
                   if use_bbox_info or 'bbox' not in by_part]
    if all([pd.isna(t) for t in tot_heights]):
        return None
    person_height = max([t if not pd.isna(t) else 0 for t in tot_heights])
    return person_height


def person_has_rel_size(
        img_meta: Dict[str, Any], ann: Dict[str, Any],
        min_rel_height: float = None, max_rel_height: float = None,
        img_target_size: Tuple[int, int] = None,
        all_keypoint_names: Sequence[str] = COCO_STD_KEYPOINT_NAMES,
        assumed_height=1.7,
        factors=FACTORS,
        use_bbox_info: bool = False,
) -> Optional[bool]:
    """Check whether the relative size of the person described by annotation
    ``ann`` wrt image size is in specified range.
    If ``img_target_size`` is given, it is assumed the image is padded before
    heights are compared (i.e. either height or width are increased until
    the same ratio is achieved as for ``img_target_size``).
    The size of the person is the maximum total height estimated using
    :py:func:`annotation_to_sizes`.


    :param img_meta: the image meta data containing the image size info;
        the dict must at least contain
        ``{"width": img_width, "height": img_height``}
        describing the dimensions to which the annotation currently applies
    :param ann: the annotation with the keypoint data for the person
    :param min_rel_height: the lower bound for the relative height
    :param max_rel_height: the upper bound for the relative height
    :param img_target_size: target image size ``(height, width)`` in pixels;
        if this is given, it is assumed that the image is padded to the ratio
        specified by the given size before applying the height ratio check
        between image and person height
    :param all_keypoint_names: see :py:func:`annotation_to_sizes`
    :param assumed_height: see :py:func:`annotation_to_sizes`
    :param factors: see :py:func:`annotation_to_sizes`
    :param use_bbox_info: whether to use the bounding box information;
        see :py:func:`annotation_to_tot_height` for details
    :return: whether the relative height of the person to the image height
        is within the given bounds (possibly scaling the annotation if
        required);
        ``None`` if the person height could not be determined
    """
    # The assumed height in pixels:
    img_size: Tuple[int, int] = (img_meta['height'], img_meta['width'])
    # Resize annotation if requested:
    if img_target_size is not None:
        ann: Dict[str, Any] = pad_and_scale_annotation(
            ann, from_size=img_size, to_size=img_target_size)
        img_size = img_target_size

    person_height = annotation_to_tot_height(
        annotation=ann,
        all_keypoint_names=all_keypoint_names,
        assumed_height=assumed_height,
        factors=factors,
        use_bbox_info=use_bbox_info
    )
    if person_height is None:
        return None

    rel_height = person_height / img_size[0]
    if (min_rel_height is None or min_rel_height <= rel_height) and \
            (max_rel_height is None or rel_height <= max_rel_height):
        return True
    return False


def annotations_to_mask(*, annotation_wh: Tuple[int, int],
                        annotations: List[dict],
                        keypoint_idxs: Union[Sequence[int],
                                             Iterable[Iterable[int]]],
                        skeleton: Sequence[Tuple[int, int]],
                        pt_radius: float = 10 / 400,
                        link_width: float = None,
                        ) -> PIL.Image.Image:
    """Create a mask of the linked keypoints from the ``annotations`` list.
    Keypoints are specified by their ``keypoint_idxs``.
    It is assumed that the original image considered in ``annotations``
    has ``annotation_wh``, which is the used as the size of the created mask.

    :param annotation_wh: :py:class:`PIL.Image.Image` size of the
        original image assumed by the annotation and output size of the mask;
        format: ``(width, height)`` in pixels
    :param annotations: annotations from which to create mask
    :param keypoint_idxs: indices of the starting positions of keypoints to
        process; the keypoints are saved in a list of the form
        ``[x1,y1,v1, x2,y2,v2, ...]``, and to process keypoints 1 and 2 one
        needs ``keypoint_idxs=[0,3]``;
        ``keypoint_idxs`` should be given as a list of "parts", where each
        part is a list of keypoint indices which should be connected by
        a link line in the mask;
        in case just a list of ``int`` values is given, these are assumed to
        represent only one part
    :param skeleton: list of links as tuples ``[kpt1_idx, kpt2_idx]``;
        this is the skeleton list from the COCO annotations, only each
        entry reduced by 1
    :param pt_radius: radius of a point relative to the height of the annotated
        person (as returned by :py:func:`annotation_to_tot_height`);
        if the height cannot be estimated, it is assumed to be the image height
    :param link_width: the width the line of a link should have
        relative to the person height (see ``pt_radius``);
        defaults to 2x the ``pt_radius``
    :return: mask (same size as original image)
    """
    link_width: float = link_width if link_width is not None else 2 * pt_radius

    # start with empty mask
    mask = PIL.Image.new('1', annotation_wh)
    draw = PIL.ImageDraw.Draw(mask)

    keypoint_idxs: Iterable[Iterable[int]] = [keypoint_idxs] \
        if isinstance(keypoint_idxs[0], int) else keypoint_idxs

    for annotation in annotations:
        # Get point radius and link width wrt. person height:
        person_height = annotation_to_tot_height(annotation) or annotation_wh[0]
        abs_pt_radius: int = int(math.ceil(pt_radius * person_height))
        abs_link_width: int = int(math.ceil(link_width * person_height))

        # obtain all keypoints:
        # The keypoints annotations have format [x1,y1,v1, x2,y2,v2, ...];
        # turn this into numpy arrays
        # xs=[x1,x2,..], ys=[y1,y2,..], vs=[v1,v2,..]:
        kpt_xs, kpt_ys, visibilities = _ann_to_kpts_info(annotation)
        # sanity check: are all keypoints within image?
        for kpt_x, kpt_y in zip(kpt_xs, kpt_ys):
            if not (0 <= kpt_x <= annotation_wh[0]
                    and 0 <= kpt_y <= annotation_wh[1]):
                raise ValueError(
                    ("Keypoint coordinates ({}, {}) out of bounds of"
                     " image size ({}, {})"
                     ).format(kpt_x, kpt_y, *annotation_wh))

        # add one part (keypoints & links) after another:
        for part in keypoint_idxs:
            includes: List[bool] = [
                visibilities[idx] == 2 and idx * 3 in part
                for idx in range(len(visibilities))]

            _add_kpts_and_links(draw,
                                list(zip(kpt_xs, kpt_ys)), includes, skeleton,
                                abs_pt_radius, abs_link_width)
    return mask


# noinspection PyIncorrectDocstring
def annotations_to_heatmap(annotation_wh: Tuple[int, int],
                           annotations: List[dict],
                           keypoint_idxs: Union[Sequence[int],
                                                Iterable[Iterable[int]]],
                           skeleton: Sequence[Tuple[int, int]],
                           pt_radius: float = 10 / 400,
                           link_width: float = None,
                           ) -> PIL.Image.Image:
    # pylint: disable=unused-argument
    """Create a heatmap marking the centroids of the linked keypoints from the
    ``annotations`` list.
    Keypoints are specified by their ``keypoint_idxs``.
    For details on the parameters, see :py:func:`annotations_to_mask`.

    :param pt_radius: this is used as standard deviation and as the
        point radius to calculate connected components
    :return: heatmap (1-channel grayscale image in mode ``'L'``) of
        size ``annotation_wh``
    """
    # start with empty mask (numpy representation of a PIL.Image.Image)
    mask_np = np.zeros((annotation_wh[1], annotation_wh[0]))

    keypoint_idxs: Iterable[Iterable[int]] = [keypoint_idxs] \
        if isinstance(keypoint_idxs[0], int) else keypoint_idxs
    if any(len(list(kpts)) > 1 for kpts in keypoint_idxs):
        raise NotImplementedError("Received a body part consisting of more "
                                  "than one keypoint. Centroids of connected"
                                  "keypoints are not supported yet!"
                                  "keypoint_idxs: {}".format(keypoint_idxs))

    for annotation in annotations:
        # Get point radius and link width wrt. person height:
        person_height = annotation_to_tot_height(annotation) or annotation_wh[0]
        abs_pt_radius: int = int(math.ceil(pt_radius * person_height))

        # obtain all keypoints:
        # The keypoints annotations have format [x1,y1,v1, x2,y2,v2, ...];
        # turn this into numpy arrays
        # xs=[x1,x2,..], ys=[y1,y2,..], vs=[v1,v2,..]:
        kpt_xs, kpt_ys, visibilities = _ann_to_kpts_info(annotation)

        # add one part (keypoints & links) after another:
        for part in keypoint_idxs:
            # TODO: collect centroids of connected components
            # For now: Just treat each keypoint separately
            # Allocate keypoints centroids in the middle of the pixels: (px[0]+0.5, px[1]+0.5)
            centroids: List[Tuple[int, int]] = [
                (kpt_xs[idx] + 0.5, kpt_ys[idx] + 0.5) for idx in range(len(visibilities))
                if visibilities[idx] == 2 and idx * 3 in part
            ]

            for centroid in centroids:
                mask_np = add_gaussian_peak(mask_np, centroid,
                                            binary_radius=abs_pt_radius)

    mask = PIL.Image.fromarray(mask_np * 255).convert('L')
    return mask


def _add_kpts_and_links(
        draw: PIL.ImageDraw.Draw,
        kpt_coords: Sequence[Tuple[int, int]], includes: Sequence[bool],
        skeleton: Sequence[Tuple[int, int]],
        abs_pt_radius: int = 4,
        abs_link_width: int = None):
    """Given a draw object, add points for the given keypoint coordinates
    and lines for the specified links.

    :param draw: the draw object
    :param kpt_coords: a sequence of keypoint coordinates within the given
        draw object as ``(x, y)``
    :param includes: visibility information for the given keypoints;
        ``kpt_coords`` at index ``i`` are added and considered for
        link drawing in case ``includes[i]`` is (``True``)
    :param skeleton: the list of links between the keypoints for which to
        draw lines; a link between keypoint at index ``idx1`` and the one
        at index ``idx2`` in the ``kpt_coord`` list should be given as
        the tuple ``(idx1, idx2)``
    :param abs_pt_radius: the absolute radius of keypoint points to draw
        in pixels
    :param abs_link_width: the absolute width of link lines to draw in pixels;
         defaults to 2x ``abs_pt_radius``
    """
    abs_link_width = abs_link_width or 2 * abs_pt_radius

    # draw skeleton
    for link in skeleton:
        # if visible and of interest, draw link
        if all([includes[link[i]] for i in (0, 1)]):
            draw.line(xy=[kpt_coords[link[0]],
                          kpt_coords[link[1]]],
                      width=abs_link_width,
                      fill=1)
    # draw keypoints
    for kpt_x, kpt_y in [kpt_coords[i] for i in range(len(kpt_coords))
                         if includes[i]]:
        draw.ellipse(
            (kpt_x - abs_pt_radius, kpt_y - abs_pt_radius,
             kpt_x + abs_pt_radius, kpt_y + abs_pt_radius),
            fill=1)


def to_keypoint_idxs(keypoint_names: Iterable[str],
                     coco: COCO = None,
                     all_keypoint_names=COCO_STD_KEYPOINT_NAMES) -> List[int]:
    """Return the start indices for the given keypoints within an
    annotations ``'keypoint'`` field.

    Keypoints

        ``{keypt_name1: (x1,y1,visibility1), keypt_name2: (...), ...}``

    are in COCO annotated as a list of

        ``[x1, y1, visibility1,   x2, y2, ...]``

    This method returns the index of the x-coordinates within the
    annotation list.
    To obtain the mapping from keypoints to keypoint indices, either directly
    specify the list of keypoint names or provide the coco instance to retrieve
    it from.

    :param keypoint_names: list of keypoint names to collect the start
        indices for
    :param all_keypoint_names: mapping of keypoint names to keypoint indices
        (3x their index in the list)
    :param coco: coco handler from which to retrieve mapping of keypoint
        names to keypoint indices (overrides all_keypoint_names)
    :return: list with the start indices for the given keypoints
    """
    if isinstance(keypoint_names, str):
        raise ValueError("keypoint_names is str not iterable of strings: {}"
                         .format(keypoint_names))
    if coco:
        all_keypoint_names: list = [cat for cat in coco.cats.values()
                                    if cat['name'] == 'person'][0]['keypoints']
    keypoint_idxs = []
    for kpt_name in keypoint_names:
        if kpt_name not in all_keypoint_names:
            raise KeyError(("keypoint name {} not in available keypoint "
                            "names {}").format(kpt_name,
                                               all_keypoint_names))
        keypoint_idxs.append(3 * all_keypoint_names.index(kpt_name))
    return keypoint_idxs


def all_keypts_visible(ann: Dict[str, Any],
                       keypoint_idxs: Optional[Sequence[int]]) -> bool:
    """Whether all given keypoints are marked as visible in the given
    annotation.

    :param ann: coco annotation to check
    :param keypoint_idxs: keypoint constraints:
        starting indices of keypoints in the ``keypoint`` list of the
        annotation that must be marked as visible
        (cf. :py:meth:`to_keypoint_idxs`);
        if ``None`` or empty, ``True`` is returned for the annotation
        to be regarded valid
    :return: whether all specified keypoints are marked as visible in the
        annotation
    """
    # Keypoint constraints empty?
    if keypoint_idxs is None or len(keypoint_idxs) == 0:
        return True

    # Any keypoint annotated?
    if not ann['num_keypoints'] >= 1:
        return False

    # Needed keypoints visible?
    keypoint_infos: List[List[int]] = [ann['keypoints'][idx:idx + 3]
                                       for idx in keypoint_idxs]
    # for each info: info[2] is visibility info, a value of 2 means visible
    if not all([k[2] == 2 for k in keypoint_infos]):
        return False
    return True


def any_keypts_visible(ann: Dict[str, Any],
                       keypoint_idxs: Optional[Sequence[int]] = None) -> bool:
    """Whether any of the given keypoints are marked as visible in the given
    annotation.

    :param ann: coco annotation to check
    :param keypoint_idxs: keypoint constraints:
        starting indices of keypoints in the ``keypoint`` list of the
        annotation that must be marked as visible
        (cf. :py:meth:`to_keypoint_idxs`);
        if ``None`` or empty, ``True`` is returned for the annotation
        to be regarded valid
    :return: whether all specified keypoints are marked as visible in the
        annotation
    """
    # Keypoint constraints empty?
    if keypoint_idxs is None or len(keypoint_idxs) == 0:
        return True

    # Any keypoint annotated?
    if not ann['num_keypoints'] >= 1:
        return False

    # Needed keypoints visible?
    for idx in keypoint_idxs:
        if ann['keypoints'][idx + 3] == 2:
            return True
    return False


def _padding_and_scale_for(from_size: Tuple[int, int],
                           to_size: Tuple[int, int]
                           ) -> Tuple[Tuple[int, int, int, int], float]:
    """Calculate padding and scale to get from ``from_size`` to ``to_size``
    via first padding then scaling a 2D image.

    :param from_size: previous size as ``(height, width)`` in px
    :param to_size: new target size as ``(height, width)`` in px
    :return: padding values as tuple
        ``(pad_left, pad_right, pad_top, pad_bottom)``
        and float scale value
    """
    from_h = from_size[0]
    from_w = from_size[1]
    to_h = to_size[0]
    to_w = to_size[1]
    if from_h / from_w > to_h / to_w:  # pad width
        scale = to_h / from_h
    else:  # pad height
        scale = to_w / from_w

    padding = padding_for_ratio(from_size=from_size,
                                to_ratio=to_size[1] / to_size[0])
    return padding, scale


def pad_and_scale_annotation(
        ann: Dict[str, Any], from_size: Tuple[int, int],
        scale: Union[float, Tuple[float, float]] = 1.,
        padding: Union[Tuple[int, int], Tuple[int, int, int, int]] = (0, 0),
        to_size: Tuple[int, int] = None,
        inplace: bool = False,
        skip: Union[str, Sequence[str]] = None
):
    """Given a MS COCO style annotation, apply first padding and/or then
    scaling to its coordinates.
    The image transformation operation that is imitated on the
    annotation is

    1. addition of 0-padding at top, bottom, left, and right.
    2. a rescaling by the given ``scale=(scale_height, scale_width)``
       factors.

    The following coordinate information is adjusted:

    - bounding box (float values)
    - keypoints (float values)
    - segmentation mask (polygon or Run-Length Encoded)

    If ``inplace`` is ``True``, operate on ``ann`` and return it in the end,
    else operate on a copy which is finally returned.
    If ``to_size`` is given, ``padding`` and ``scale`` are determined to match
    the :py:class:`~hybrid_learning.datasets.transforms.image_transforms.PadAndResize`
    transformation behavior.

    :param ann: the annotation to scale and pad
    :param from_size: the original size of the image the annotation is for
        as ``(height, width)`` in pixels;
        can be obtained from the image meta annotation
    :param to_size: the target size as ``(height, width)`` to achieve by first
        scaling then padding;
        overrides ``scale`` and ``padding``
    :param scale: the scaling factors as ``(scale_height, scale_width)``;
        if one float value is given, a uniform scaling is applied
    :param padding: the padding to apply after rescaling in pixels as
        ``(padding_left_right, padding_top_bottom)`` or
        ``(pad_left, pad_right, pad_top, pad_bottom)``
    :param inplace: whether to write changes back to ``ann``
    :param skip: list of strings (or comma separated string concatenation)
        of any of ``keypoints``, ``segmentation``, ``bbox``;
        any mentioned information will be excluded from scaling & padding
        and is left unchanged (may increase speed in case only one type of
        rescale is needed)
    :return: annotation with coordinate information scaled and padded;
        a new instance in case ``inplace`` is false, else the changed ``ann``
    """
    skip = skip or []
    if to_size is not None:
        padding, scale = _padding_and_scale_for(from_size=from_size,
                                                to_size=to_size)
    if len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])
    if isinstance(scale, float):
        scale: Tuple[float, float] = (scale, scale)
    scale_height, scale_width = scale
    pad_left, pad_right, pad_top, pad_bottom = padding
    to_size = to_size or (int(from_size[0] * scale_height),
                          int(from_size[1] * scale_width))
    new_ann = dict(ann)

    # region KEYPOINTS
    if "keypoints" not in skip:
        kpts_x, kpts_y, vis = _ann_to_kpts_info(ann)
        kpts_y = list((np.array(kpts_y) + pad_top) * scale_height)
        kpts_x = list((np.array(kpts_x) + pad_left) * scale_width)
        new_ann['keypoints'] = [i for kpts_info in zip(kpts_x, kpts_y, vis)
                                for i in kpts_info]
    # endregion

    # region SEGMENTATION
    if "segmentation" not in skip:
        if new_ann['iscrowd']:
            # RLE (run-length encoding) storage format:
            # For details on format have a look at the pycocotools.mask source.
            # The format can be parsed to and from PIL.Image.Image using the
            # pycocotools utils.
            seg = ann['segmentation']
            rle = seg['count'] if not isinstance(seg['counts'], list) else \
                pycocotools.mask.frPyObjects(seg, from_size[0], from_size[1])
            new_mask_np = pycocotools.mask.decode(rle)
            # pad
            new_mask_np = np.pad(
                new_mask_np,
                pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
                constant_values=0)
            # rescale
            new_mask_np = np.asfortranarray(
                PIL.Image.fromarray(new_mask_np, mode='L').resize(
                    size=(to_size[1], to_size[0]),
                    resample=PIL.Image.NEAREST))
            rle = pycocotools.mask.encode(new_mask_np)
            new_ann['segmentation'] = rle
        else:
            # Polygon:
            # The coordinates of connected corner points of the polygon is
            # saved as list of the form [x1,y1, x2,y2, ...].
            # A segmentation may consist of one or more of such lists.
            new_segs: List[float] = []
            for seg in ann['segmentation']:
                orig_xy = np.array(seg).reshape(-1, 2)
                new_xy = ((orig_xy + np.array([pad_left, pad_top]))
                          * np.array([scale_width, scale_height]))
                new_segs.append(new_xy.reshape(-1).tolist())
            new_ann['segmentation'] = new_segs
        # endregion

    # region BBOX
    if "bbox" not in skip:
        bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']
        new_ann['bbox'] = [(bbox_x + pad_left) * scale_width,
                           (bbox_y + pad_top) * scale_height,
                           bbox_w * scale_width,
                           bbox_h * scale_height]
    # endregion

    if inplace:
        for key, val in new_ann.items():
            ann[key] = val
        return ann
    return new_ann
