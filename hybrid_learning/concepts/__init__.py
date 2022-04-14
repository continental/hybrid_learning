#  Copyright (c) 2022 Continental Automotive GmbH
"""
Models and handles for concept embedding analysis and model extension.
This includes handles for

- Defining concept models
- Training and evaluating concept models
- Conducting a series of concept embedding analysis experiments
"""

from . import analysis
from . import models
from . import train_eval
from .models import ConceptEmbedding, \
    ConceptDetectionModel2D, ConceptDetection2DTrainTestHandle, \
    ConceptClassificationModel2D, ConceptClassification2DTrainTestHandle, \
    ConceptSegmentationModel2D, ConceptSegmentation2DTrainTestHandle, \
    model_extension
from .train_eval import EarlyStoppingHandle, ResettableOptimizer, TrainEvalHandle, kpis
from .concepts import Concept, SegmentationConcept2D
from .analysis import ConceptAnalysis, AnalysisResult
