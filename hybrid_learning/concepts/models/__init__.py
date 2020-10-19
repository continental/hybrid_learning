"""Models and handles to obtain concept embeddings."""

#  Copyright (c) 2020 Continental Automotive GmbH

from .base_handles import EarlyStoppingHandle, ResettableOptimizer, \
    TrainEvalHandle
from .concept_detection import ConceptDetectionModel2D, \
    ConceptDetection2DTrainTestHandle
from .concept_segmentation import ConceptSegmentationModel2D, \
    ConceptSegmentation2DTrainTestHandle
