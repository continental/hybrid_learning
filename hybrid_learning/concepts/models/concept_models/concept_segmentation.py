#  Copyright (c) 2022 Continental Automotive GmbH
"""Model for concept segmentation, and corresponding training and evaluation
handle.
See :py:class:`ConceptSegmentationModel2D` and
:py:class:`ConceptSegmentation2DTrainTestHandle` for details.
"""

import torch.nn

from .concept_detection import ConceptDetectionModel2D, ConceptDetection2DTrainTestHandle
from ...concepts import SegmentationConcept2D


class ConceptSegmentationModel2D(ConceptDetectionModel2D):
    # pylint: disable=line-too-long
    r"""A concept model that segments the concept in an image.
    This is solved by a localization model with window respectively kernel size
    :math:`1\times 1`.

    For a usage example compare
    :py:class:`~hybrid_learning.concepts.models.concept_models.concept_detection.ConceptDetection2DTrainTestHandle`.
    """

    # pylint: enable=line-too-long

    def __init__(self, concept: SegmentationConcept2D, model: torch.nn.Module,
                 layer_id: str, in_channels: int = None):
        # pylint: disable=line-too-long
        r"""Init.

        Wrapper around init of a
        :py:class:`~hybrid_learning.concepts.models.concept_models.concept_detection.ConceptDetectionModel2D`
        with fixed
        :py:attr:`~hybrid_learning.concepts.models.concept_models.concept_detection.ConceptDetectionModel2D.kernel_size`
        of :math:`1\times1`.
        For details on the arguments see the init function of the super class
        :py:class:`~hybrid_learning.concepts.models.concept_models.concept_detection.ConceptDetectionModel2D`.
        """
        # pylint: enable=line-too-long
        super().__init__(
            concept=concept, model=model, layer_id=layer_id,
            kernel_size=(1, 1), in_channels=in_channels)


class ConceptSegmentation2DTrainTestHandle(ConceptDetection2DTrainTestHandle):
    """Train and test handle for concept segmentation providing loss and
    metric defaults.
    See :py:class:`ConceptSegmentationModel2D` for details on the model.
    """
