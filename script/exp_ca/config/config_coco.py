#  Copyright (c) 2022 Continental Automotive GmbH
"""Sacred default configuration to conduct concept analysis on a subset
of MS COCO with concept maskings.
To run as script, override the following configurations:

- ``model_key``: the key of the model builder to use (must be registered before using config_ca.register_model_builder)
- ``layer_infos``: a list with the layer IDs of the layers to analyse

This can e.g. be done by importing the experiment handle :py:data:`ex` from
this module and appending configuration/named configuration.
Then call ``ex.run_commandline()``.

To add a file storage observer, add to as commandline argument
``-F BASEDIR`` or ``--file-storage=BASEDIR``.
"""

import os
import sys
from typing import Tuple, Dict, Any, Optional, Sequence, Union

# Assuming this is called from project root
import torch

# pylint: disable=no-name-in-module,import-error
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
# pylint: disable=unused-variable,unused-argument,unused-import
# pylint: disable=too-many-arguments

from hybrid_learning.datasets import caching
from hybrid_learning.datasets.custom import coco
from hybrid_learning.datasets.custom.coco.keypoints_processing import \
    person_has_rel_size

from .config_ca import ex, DataGetter, analysis


# noinspection PyUnusedLocal
@ex.config
def coco_config():
    # pylint: disable=unused-variable
    """The default configuration for coco specifics."""
    # Information (the subparts and the rel_size) of coco BodyParts to consider;
    # 'subparts' is a list or single string entry of a part name, and defaults
    # to the entry key in upper case;
    # 'rel_size' must have the format of 'default_rel_size' and defaults to it
    part_infos: dict = dict(
        eye=dict(subparts=['LEFT_EYE', 'RIGHT_EYE']),
        arm=dict(subparts=['LEFT_ARM', 'RIGHT_ARM']),  # segmentation
        # arm=dict(subparts=['LEFT_ELBOW', 'RIGHT_ELBOW']),  # detection
        leg=dict(subparts=['LEFT_LEG', 'RIGHT_LEG']),  # segmentation
        # leg=dict(subparts=['LEFT_KNEE', 'RIGHT_KNEE']),  # detection
        wrist=dict(subparts=['LEFT_WRIST', 'RIGHT_WRIST']),
        ankle=dict(subparts=['LEFT_ANKLE', 'RIGHT_ANKLE']),
    )
    # Which parts from part_infos to analyse (by default: all):
    part_keys: list = list(part_infos.keys())
    # Path to the images directory of the coco dataset to use.
    dataset_root: str = os.path.join("dataset", "coco_test")
    img_size: list = [400, 400]
    act_data_setts = dict(
        # Path for caching of all datasets (set to "" to disable caching)
        cache_root=os.path.join(
            "cache", "coco", "cache_{}x{}".format(*img_size))
    )
    img_mask_cache_root: str = act_data_setts["cache_root"]
    # Path to the directory under which to put the COCO concept mask folders
    # with generated masks; if unset, the class default is used
    masks_root_root: str = None
    # The default radius of keypoint markers relative to the person size;
    # Best choose smaller for detection
    default_pt_radius: float = 10 / 400
    # Licenses to include in data (IDs are MS COCO license IDs; None=all)
    license_ids: list = None
    # The callable to use for data retrieval
    get_data: COCODataGetter = COCODataGetter()


class COCODataGetter(DataGetter):
    """Get coco data train test tuple."""
    DATASET_CLASS = coco.ConceptDataset
    """The (coco) dataset class to use."""

    # noinspection PyUnusedLocal
    @classmethod
    @ex.capture
    def __call__(cls,
                 subparts: Sequence[str],
                 img_mask_cache_root: str,
                 pt_radius: float,
                 dataset_root: str,
                 masks_root_root: str = None,
                 img_size: Optional[Tuple[int, int]] = None,
                 mask_size: Optional[Tuple[int, int]] = None,
                 person_size: Tuple[float, float] = None,
                 license_ids: Sequence[int] = None,
                 device: Union[str, torch.device] = None,
                 show_progress_bars: bool = True,
                 **_other_parts_specs
                 ) -> Tuple[coco.ConceptDataset, coco.ConceptDataset]:
        """Get concept data subsetted according to settings."""
        parts: list = [coco.BodyParts[part_name]
                       for part_name in subparts]

        # Annotation file and dataset root
        dataset_root_templ: str = os.path.join(dataset_root, "images", "{}")
        ann_fp_templ: str = os.path.join(
            dataset_root, "annotations",
            "person_keypoints_" + cls.DATASET_CLASS.settings_to_str(
                dataset_root=dataset_root_templ, body_parts=parts,
                img_size=img_size, mask_size=mask_size,
                person_rel_size_range=person_size
            ))

        datasets: Dict[str, cls.DATASET_CLASS] = {}
        for split, img_base in (("TEST", "val2017"), ("TRAIN", "train2017")):

            # Subsetting settings
            ann_fp = ann_fp_templ.format(img_base)
            if os.path.isfile(ann_fp):  # load from file without subsetting
                condition, annotations_fp = None, ann_fp
                subsetting_body_parts = None
            else:  # enable subsetting by person size
                annotations_fp = None
                subsetting_body_parts = parts
                condition = None if person_size is None else \
                    lambda i, a: person_has_rel_size(
                        i, a, min_rel_height=person_size[0],
                        max_rel_height=person_size[1], img_target_size=img_size)

            # Actual data acquisition
            if not show_progress_bars:
                ex.logger.info("Loading and subsetting dataset split %s", split)
            data: cls.DATASET_CLASS = cls.DATASET_CLASS(
                dataset_root=dataset_root_templ.format(img_base),
                masks_root_root=masks_root_root,
                annotations_fp=annotations_fp,
                pt_radius=pt_radius, split=split, body_parts=parts,
                img_size=img_size, mask_size=mask_size, device=device,
            ).subset(body_parts=subsetting_body_parts, license_ids=license_ids,
                     condition=condition, show_progress_bar=show_progress_bars)

            # Annotations caching
            if person_size is not None and len(data) > 0 \
                    and not os.path.isfile(ann_fp):
                data.to_raw_anns(description=(
                    "Subset of MS COCO dataset with persons of relative size in"
                    " the range [{}, {}] assuming images are padded & scaled "
                    "to HxW={}x{}").format(*person_size, *img_size),
                                 save_as=ann_fp)

            # Cache settings
            if img_mask_cache_root:
                img_cache = caching.JPGCache(
                    cache_root=os.path.join(img_mask_cache_root, "images",
                                            img_base))
                mask_cache = caching.PTCache(
                    cache_root=analysis.analysis_handle.default_mask_cache_root(
                        wrapped_data=data, cache_root=img_mask_cache_root
                    )
                )
                data.transforms_cache = caching.CacheTuple(img_cache,
                                                           mask_cache)

            datasets[split] = data

        return datasets["TRAIN"], datasets["TEST"]


# noinspection PyUnusedLocal
@ex.config_hook
def complete_part_infos(config: Dict[str, Any],
                        command_name, logger) -> Dict[str, Any]:
    # pylint: disable=unused-argument
    """Complete part_infos:

    - the relative size defaults to default_rel_size
    - the sub-parts defaults to the part name in upper case
    - the sub-parts should be a list of str (if str, encapsule in list)
    """
    new_infos: dict = {}
    for prt, infos in config['part_infos'].items():
        new_infos[prt] = dict(infos)
        new_infos[prt].setdefault('rel_size', config['default_rel_size'])
        new_infos[prt].setdefault('pt_radius', config['default_pt_radius'])
        new_infos[prt].setdefault('subparts', [prt.upper()])
        if isinstance(new_infos[prt]['subparts'], str):
            new_infos[prt]['subparts'] = [new_infos[prt]['subparts']]
        new_infos[prt].setdefault('label', "-".join(new_infos[prt]['subparts']))
    return dict(part_infos=new_infos)


if __name__ == "__main__":
    ex.run_commandline()
