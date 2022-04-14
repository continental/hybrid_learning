"""Common fixtures for testing concept models and analysis."""
#  Copyright (c) 2022 Continental Automotive GmbH

# Workaround to make pylint cope correctly with pytest fixtures:
# pylint: disable=redefined-outer-name
import os
from typing import Dict, Any, Tuple

import pytest
import torch
import torchvision as tv

from hybrid_learning.concepts.concepts import SegmentationConcept2D
from hybrid_learning.concepts.models import ConceptDetectionModel2D
from hybrid_learning.datasets import DataTriple, Sequence
from hybrid_learning.datasets.custom import coco


@pytest.fixture(scope="module")
def main_model() -> torch.nn.Module:
    """Load a main model to work with."""
    return tv.models.alexnet(pretrained=False)


@pytest.fixture(scope="module")
def sample_layer() -> Dict[str, Any]:
    """Sample layer index to use for testing just on one layer."""
    return dict(
        layer_id="features.5",
        out_channels=192,
        kernel_size=(2, 2),
        out_size=(13, 13),  # output size (height, width)
    )


@pytest.fixture(scope="module")
def input_size() -> Tuple[int, int, int]:
    """The input size for the used main model."""
    return 3, 224, 224


@pytest.fixture
def concept(input_size: Tuple[int, int, int]) -> SegmentationConcept2D:
    """Provide a simple concept with tiny dataset for the concept model.
     (use the same data for training, validation, and testing)."""
    concept_name = 'FACE'
    coco_concepts: Sequence[coco.BodyParts] = [coco.BodyParts[concept_name]]

    max_num_imgs = 10
    root = os.path.join("dataset", "coco_test")
    concept_data = coco.ConceptDataset(
        dataset_root=os.path.join(root, "images", "train2017"),
        body_parts=coco_concepts,
        transforms=coco.ConceptDataset.get_default_transforms(input_size[1:])
    ).subset(
        body_parts=coco_concepts,
        num=max_num_imgs
    )
    assert 0 < len(concept_data) <= max_num_imgs

    return SegmentationConcept2D(
        name=concept_name,
        data=DataTriple(train=concept_data, val=concept_data,
                        test=concept_data),
        rel_size=0.16)


@pytest.fixture
def concept_model(concept: SegmentationConcept2D, main_model: torch.nn.Module,
                  sample_layer: Dict):
    """Return a standard concept model for given concept for experiments."""
    concept_model = ConceptDetectionModel2D(
        concept=concept, model=main_model,
        layer_id=sample_layer["layer_id"],
        in_channels=sample_layer["out_channels"])
    return concept_model
