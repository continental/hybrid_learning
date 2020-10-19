"""Common fixtures for testing concept models and analysis."""
#  Copyright (c) 2020 Continental Automotive GmbH

# Workaround to make pylint cope correctly with pytest fixtures:
# pylint: disable=redefined-outer-name
import os

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
    return tv.models.detection.maskrcnn_resnet50_fpn(pretrained=True)


@pytest.fixture(scope="module")
def concept() -> SegmentationConcept2D:
    """Provide a simple concept with tiny dataset for the concept model.
     (use the same data for training, validation, and testing)."""
    concept_name = 'FACE'
    coco_concepts: Sequence[coco.BodyParts] = [coco.BodyParts[concept_name]]

    max_num_imgs = 10
    root = os.path.join("dataset", "coco_test")
    concept_data = coco.ConceptDataset(
        dataset_root=os.path.join(root, "images", "train2017"),
        body_parts=coco_concepts
    ).subset(
        body_parts=coco_concepts,
        num=max_num_imgs
    )
    assert 0 < len(concept_data) <= max_num_imgs

    return SegmentationConcept2D(
        name=concept_name,
        data=DataTriple(train=concept_data, val=concept_data,
                        test=concept_data),
        rel_size=0.1)


@pytest.fixture(scope="module")
def train_concept():
    """Provide simple concept with small dataset for concept model training."""
    concept_name = 'FACE'
    coco_concepts: Sequence[coco.BodyParts] = [coco.BodyParts[concept_name]]
    max_num_imgs = 50
    root = os.path.join("dataset", "coco_test")

    concept_data_train: coco.ConceptDataset = coco.ConceptDataset(
        dataset_root=os.path.join(root, "images", "train2017"),
        body_parts=coco_concepts
    ).subset(body_parts=coco_concepts,
             num=max_num_imgs)

    concept_data_test: coco.ConceptDataset = coco.ConceptDataset(
        annotations_fp=os.path.join(
            root, "annotations", "person_keypoints_val2017.json"),
        dataset_root=os.path.join(root, "images", "val2017"),
        body_parts=coco_concepts
    ).subset(body_parts=coco_concepts,
             num=max(1, max_num_imgs // 3))
    assert len(concept_data_test) > 0

    return SegmentationConcept2D(name=concept_name,
                                 data=DataTriple(train_val=concept_data_train,
                                                 test=concept_data_test),
                                 rel_size=0.1)


@pytest.fixture
def concept_model(concept: SegmentationConcept2D, main_model: torch.nn.Module):
    """Return a standard concept model for given concept for experiments."""
    layer_idx = 'backbone.body.layer3'
    in_channels = 1024  # specify for faster initialization
    concept_model = ConceptDetectionModel2D(concept=concept, model=main_model,
                                            layer_id=layer_idx,
                                            in_channels=in_channels)
    return concept_model
