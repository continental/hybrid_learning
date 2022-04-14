#  Copyright (c) 2022 Continental Automotive GmbH
"""Sacred default configuration to conduct concept analysis on a subset
of BRODEN with concept maskings.
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
from typing import Tuple, Dict, Any, Optional, List

# pylint: disable=no-name-in-module,import-error
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
# pylint: disable=unused-variable,unused-argument,unused-import

sys.path.insert(0, "")
from hybrid_learning.datasets import caching, transforms as trafos
from hybrid_learning.datasets.custom import broden, coco

from .config_ca import ex, DataGetter


# noinspection PyUnusedLocal
@ex.config
def broden_config():
    # pylint: disable=unused-variable
    """The default configuration for broden specifics."""
    # Information (the subparts and the rel_size) of coco BodyParts to consider;
    # 'subparts' is a list or single string entry of a part name, and defaults
    # to the entry key in upper case;
    # 'rel_size' must have the format of 'default_rel_size' and defaults to it
    part_infos: dict = dict(
        hand=dict(broden_label="person&&hand"),
        arm=dict(broden_label="person&&arm||hand"),
        foot=dict(broden_label="person&&foot"),
        leg=dict(broden_label="person&&leg||foot"),
        eye=dict(broden_label="person&&eye"),
        nose=dict(broden_label="person&&nose"),
        face=dict(broden_label="person&&head||eye||nose||mouth"),
        head=dict(broden_label="person&&head||eye||nose||mouth||ear||hair"),
        neck=dict(broden_label="person&&neck"),
        torso=dict(broden_label="person&&torso"),
    )
    # Which parts from part_infos to analyse (by default: all):
    part_keys: list = list(part_infos.keys())
    # Path to the images directory of the coco dataset to use.
    dataset_root: str = os.path.join("dataset", "broden1_224")
    # The callable to use for data retrieval
    img_size: list = [224, 224]
    act_data_setts = dict(
        # Path for caching of all datasets (set to "" to disable caching)
        cache_root=os.path.join(
            "cache", "broden", "cache_{}x{}".format(*img_size))
    )
    img_mask_cache_root: str = act_data_setts["cache_root"]
    get_data: BrodenDataGetter = BrodenDataGetter()


# noinspection PyUnusedLocal
@ex.config_hook
def complete_part_infos(config: Dict[str, Any],
                        command_name, logger) -> Dict[str, Any]:
    # pylint: disable=unused-argument
    """The default configuration part 2.
    Complete part_infos:

    - the relative size defaults to default_rel_size
    - the sub-parts merge operation defaults to the part name
    """
    new_infos: dict = {}
    for prt, infos in config['part_infos'].items():
        new_infos[prt] = dict(infos)
        new_infos[prt].setdefault('rel_size', config['default_rel_size'])
        new_infos[prt].setdefault('broden_label', prt.lower())

        label_for_fp: str = str(new_infos[prt]['broden_label'])
        for merge_op in (trafos.AND, trafos.OR, trafos.NOT):
            label_for_fp = label_for_fp.replace(merge_op.SYMB,
                                                merge_op.__name__)
        new_infos[prt].setdefault('label', label_for_fp)
    return dict(part_infos=new_infos)


class BrodenDataGetter(DataGetter):
    """Get coco data train test tuple."""

    # noinspection PyUnusedLocal
    @staticmethod
    @ex.capture
    def __call__(broden_label: str,
                 label: str,
                 img_mask_cache_root: str,
                 dataset_root: str = None,
                 img_size: Optional[Tuple[int, int]] = None,
                 mask_size: Optional[Tuple[int, int]] = None,
                 device: Optional[Tuple[int, int]] = None,
                 **other_parts_specs
                 ) -> Tuple[broden.BrodenHandle, broden.BrodenHandle]:
        """Get concept data subsetted according to settings."""
        # Transforms settings
        transforms = \
            coco.ConceptDataset.get_default_transforms(
                img_size, mask_size, device=device) + \
            trafos.OnTarget(trafos.ToFixedDims(3))

        # Annotation file and dataset root
        ann_fp_templ: str = os.path.join(
            dataset_root,
            "index_{}_" + ("{label}.csv".format(label=label)))

        datasets: Dict[str, broden.BrodenHandle] = {}
        for split, broden_split in (("TEST", 'val'), ("TRAIN", 'train')):
            # Subsetting settings
            ann_fp = ann_fp_templ.format(broden_split)
            if os.path.isfile(
                    ann_fp):  # load from file without special subsetting
                annotations_fp, prune = ann_fp, False
            else:  # enable subsetting by person size
                annotations_fp, prune = None, True

            # Actual data acquisition
            data: broden.BrodenHandle = broden.BrodenHandle.custom_label(
                dataset_root=dataset_root,
                annotations_fp=annotations_fp,
                split=split, broden_split=broden_split,
                label=broden_label,
                transforms=transforms,
                prune_empty=prune, prune_na=prune, device=device
            )

            # Annotations caching
            if ann_fp is not None and len(data) > 0 \
                    and not os.path.isfile(ann_fp):
                data.save_annotations_table(ann_fp)

            # Cache settings
            if img_mask_cache_root:
                img_cache = caching.JPGCache(cache_root=os.path.join(
                    img_mask_cache_root, "images", broden_split))
                mask_cache = caching.PTCache(cache_root=os.path.join(
                    img_mask_cache_root, "masks", broden_split, label))
                data.transforms_cache = caching.CacheTuple(img_cache,
                                                           mask_cache)

            datasets[split] = data

        return datasets["TRAIN"], datasets["TEST"]


if __name__ == "__main__":
    ex.run_commandline()
