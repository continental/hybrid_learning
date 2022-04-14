#  Copyright (c) 2022 Continental Automotive GmbH
"""Concept model architectures and train eval handles for concept embedding analysis.
The models defined here are assumed to accept activation map
outputs of a main DNN as inputs. They can be attached to the
main DNN using wrappers from the ``model_extension`` module.

The main model considered is
:py:class:`~hybrid_learning.concepts.models.concept_models.concept_models.concept_detection.ConceptDetectionModel2D`,
which is a single convolution (with options for better calibration).
The following derivatives are pre-defined:

- :py:class:`~hybrid_learning.concepts.models.concept_models.concept_detection.ConceptDetectionModel2D`:
  The base concept model. Can be used for detection of concepts in an activation map region.
- :py:class:`~hybrid_learning.concepts.models.concept_models.concept_segmentation.ConceptSegmentationModel2D`:
  The base model with fixed kernel size of 1x1 for concept segmentation.
- :py:class:`~hybrid_learning.concepts.models.concept_models.concept_classification.ConceptClassificationModel2D`
  The base model but with a single output in the interval [0,1]
  (realized by setting the kernel size to the input size and turning padding off).
"""

from .concept_detection import ConceptDetectionModel2D, ConceptDetection2DTrainTestHandle
from .concept_segmentation import ConceptSegmentationModel2D, ConceptSegmentation2DTrainTestHandle
from .concept_classification import ConceptClassificationModel2D, ConceptClassification2DTrainTestHandle
