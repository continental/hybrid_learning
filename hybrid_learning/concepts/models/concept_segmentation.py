"""Model for concept segmentation, and corresponding training and evaluation
handle."""

#  Copyright (c) 2020 Continental Automotive GmbH

from typing import Callable, Dict

import torch.nn

from hybrid_learning.datasets import DatasetSplit, BaseDataset
from hybrid_learning.datasets.transforms import TupleTransforms
from .base_handles import EarlyStoppingHandle, ResettableOptimizer
from .concept_detection import ConceptDetectionModel2D, \
    ConceptDetection2DTrainTestHandle
from ..concepts import SegmentationConcept2D


class ConceptSegmentationModel2D(ConceptDetectionModel2D):
    # pylint: disable=line-too-long
    r"""A concept model that segments the concept in an image.
    This is solved by a localization model with window respectively kernel size
    :math:`1\times 1`.

    For a usage example compare
    :py:class:`~hybrid_learning.concepts.models.concept_detection.ConceptDetection2DTrainTestHandle`.
    """

    # pylint: enable=line-too-long

    def __init__(self, concept: SegmentationConcept2D, model: torch.nn.Module,
                 layer_id: str, in_channels: int = None):
        # pylint: disable=line-too-long
        r"""Init.

        Wrapper around init of a
        :py:class:`~hybrid_learning.concepts.models.concept_detection.ConceptDetectionModel2D`
        with fixed
        :py:attr:`~hybrid_learning.concepts.models.concept_detection.ConceptDetectionModel2D.kernel_size`
        of :math:`1\times1`.
        For details on the arguments see the init function of the super class
        :py:class:`~hybrid_learning.concepts.models.concept_detection.ConceptDetectionModel2D`.
        """
        # pylint: enable=line-too-long
        super(ConceptSegmentationModel2D, self).__init__(
            concept=concept, model=model, layer_id=layer_id,
            kernel_size=(1, 1), in_channels=in_channels)


class ConceptSegmentation2DTrainTestHandle(ConceptDetection2DTrainTestHandle):
    """Train and test handle for concept segmentation providing loss and
    metric defaults.
    See :py:class:`ConceptSegmentationModel2D` for details on the model.
    """

    def __init__(self,
                 concept_model: ConceptDetectionModel2D,
                 device: torch.device = torch.device("cpu"),
                 batch_size: int = 8,
                 max_epochs: int = 5,
                 loss_fn: Callable[[torch.Tensor, torch.Tensor],
                                   torch.Tensor] = None,
                 metric_fns: Dict[str, Callable[[torch.Tensor, torch.Tensor],
                                                torch.Tensor]] = None,
                 early_stopping_handle: EarlyStoppingHandle = None,
                 optim_handle: ResettableOptimizer = None,
                 model_output_transform: TupleTransforms = None,
                 act_map_filepath_fns: Dict[DatasetSplit,
                                            Callable[[int, BaseDataset],
                                                     str]] = None
                 ):  # pylint: disable=too-many-arguments
        # pylint: disable=line-too-long
        """Init.

        For details on the init parameters see the init of the super class
        :py:class:`~hybrid_learning.concepts.models.concept_detection.ConceptDetection2DTrainTestHandle`.
        """
        # pylint: enable=line-too-long
        super(ConceptSegmentation2DTrainTestHandle, self).__init__(
            concept_model=concept_model,
            device=device,
            batch_size=batch_size,
            max_epochs=max_epochs,
            loss_fn=loss_fn,
            early_stopping_handle=early_stopping_handle,
            metric_fns=metric_fns,
            optim_handle=optim_handle,
            act_map_filepath_fns=act_map_filepath_fns,
            model_output_transform=model_output_transform
        )
