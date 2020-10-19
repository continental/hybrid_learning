#!/usr/bin/env python3.6
"""Script to create a small MS COCO test set with proper image attributions.
The attributions are collected from the flickr image homepages via
page scraping.
The script must be called from the project root location.
The starting dataset is by default assumed to lie under
``datasets/coco`, the generated dataset is put into ``datasets/coco_test``.
Make sure to remove the folders ``datasets/coco_test/images`` and
``datasets/coco_test/annotations`` before execution.
"""
#  Copyright (c) 2020 Continental Automotive GmbH

import os
import sys

# pylint: disable=wrong-import-position
PROJECT_ROOT = os.path.join("..")
sys.path.insert(0, PROJECT_ROOT)

from urllib.request import urlopen
import urllib.error
from bs4 import BeautifulSoup

import numpy
import torch
from hybrid_learning.datasets.custom import coco

# ensure reproducibility:
numpy.random.seed(0)
torch.manual_seed(0)

ATTRIBUTION_TEMPLATE = "- {file_name} ([online source]({file_url})):" \
                       " by [{author_name}]({author_url})" \
                       " licensed under [{license_name}]({license_url})"


def attribution_text(data: coco.COCODataset, i: int,
                     template=ATTRIBUTION_TEMPLATE):
    """Fill and return the attribution template for image at index ``i``
    in ``data``."""
    attribution = data.image_attribution(i)
    file_name = os.path.basename(attribution['file_path'])
    print("Processing", attribution['flickr_page'])

    # author info
    try:
        soup = BeautifulSoup(urlopen(attribution['flickr_page']), 'html.parser')
        author_box = soup.find('a', attrs={'class': 'owner-name'})
        assert author_box is not None
    except (urllib.error.URLError, AssertionError):
        print("Warning: Skipping {}".format(file_name))
        return None

    return template.format(
        file_name=file_name,
        file_url=attribution['source'],
        author_name=author_box.text.strip(),
        author_url="https://flickr.com" + author_box.attrs['href'],
        license_name=attribution['license'],
        license_url=attribution['license_url']
    )


def copy_data_with_attributions(num_samples: int, license_ids,
                                dataset_root: str, new_dataset_root: str,
                                body_parts=coco.BodyParts.FACE):
    """Under ``new_dataset_root`` create a subset of ``num_samples`` of a
    coco dataset lying under ``dataset_root``."""

    print("Collecting dataset ...")
    data = coco.KeypointsDataset(dataset_root=dataset_root, split=split.upper())
    data.img_ids = sorted(data.img_ids)  # Ensure reproducibility
    data.subset(body_parts=body_parts, license_ids=license_ids,
                num=2 * num_samples)  # some buffer for next subsetting step

    print("Subsetting data by availability of attributions ...")
    attr_texts, i = {}, 0
    while len(attr_texts) < num_samples and i < len(data):
        attr_text = attribution_text(data, i)
        if attr_text is not None:
            attr_texts[data.img_ids[i]] = attr_text
        i += 1
    data.img_ids = list(attr_texts.keys())
    print("Length for '{}'".format(split), len(data))

    print("Copying data ...")
    data.copy_to(dataset_root=new_dataset_root,
                 description='Excerpt from original COCO 2017 dataset',
                 overwrite=False)

    print("Storing image attributions ...")
    with open(os.path.join(new_dataset_root, "ATTRIBUTIONS.md"), 'w',
              encoding='utf8') as attr_file:
        attr_file.write(
            "Image Attributions\n==================\n\n" +
            "Attributions and license information for images in this folder:"
            "\n\n" + ('\n'.join(attr_texts.values()))
        )


if __name__ == "__main__":
    YEAR = 2017
    NUM_SAMPLES_PER_SPLIT = {'train': 15, 'val': 3}
    ROOT = os.path.join(PROJECT_ROOT, "dataset", "coco")

    for split in ("train", "val"):
        DATASET_ROOT = os.path.join(ROOT, "images", "{}{}".format(split, YEAR))
        copy_data_with_attributions(
            num_samples=NUM_SAMPLES_PER_SPLIT[split],
            license_ids=[4],  # only CC BY 2.0,
            dataset_root=DATASET_ROOT,
            new_dataset_root=DATASET_ROOT.replace("coco", "coco_test")
        )
