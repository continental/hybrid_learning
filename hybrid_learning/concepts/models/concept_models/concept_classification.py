#  Copyright (c) 2022 Continental Automotive GmbH
"""Model for concept classification, and its training and evaluation handle.
See :py:class:`ConceptClassificationModel2D` and
:py:class:`ConceptClassification2DTrainTestHandle` for details.
"""


from typing import Tuple, Sequence, Optional, List

import torch.nn

from hybrid_learning.datasets.transforms import TupleTransforms
from .concept_detection import ConceptDetectionModel2D, ConceptDetection2DTrainTestHandle
from ..model_extension import ModelStump
from ...concepts import Concept
from ....datasets import DataTriple


class ConceptClassificationModel2D(ConceptDetectionModel2D):
    # pylint: disable=line-too-long
    r"""A concept model that classifies whether an image-level concept is
    recognized in an activation map.
    This is solved by a localization model with window respectively kernel size
    the same as the input size.

    A forward run with a batch of size
    ``(batch, in_channels, h, w)`` returns a batch of confidence values of
    the size ``(batch, 1)``.
    For a usage example compare
    :py:class:`~hybrid_learning.concepts.models.concept_models.concept_detection.ConceptDetection2DTrainTestHandle`.

    .. warning::
        An error is raised if a tensor with ``(height, width)`` different from
        :py:attr:`~hybrid_learning.concepts.models.concept_models.concept_detection.ConceptDetectionModel2D.kernel_size`
        is provided to :py:meth:`forward`.
    """

    # pylint: enable=line-too-long

    def __init__(self, concept: Optional[Concept] = None,
                 model: Optional[torch.nn.Module] = None,
                 layer_id: Optional[str] = None,
                 in_channels: int = None,
                 act_map_size: Tuple[int, int] = None,
                 **other_settings,
                 ):
        # pylint: disable=line-too-long
        r"""Init.

        Wrapper around init of a
        :py:class:`~hybrid_learning.concepts.models.concept_models.concept_detection.ConceptDetectionModel2D`
        with fixed
        :py:attr:`~hybrid_learning.concepts.models.concept_models.concept_detection.ConceptDetectionModel2D.kernel_size`
        of :math:`1\times1` and disabled padding.
        For details on the arguments see the init function of the super class
        :py:class:`~hybrid_learning.concepts.models.concept_models.concept_detection.ConceptDetectionModel2D`.
        """
        # pylint: enable=line-too-long
        if in_channels is None or act_map_size is None:
            for _missing in (argn for argn, arg in
                             (("model", model), ("layer_id", layer_id),
                              ("concept", concept)) if arg is None):
                raise ValueError("{} not given so cannot infer in_channels "
                                 "and layer_out_size".format(_missing))
            auto_layer_out_size, _ = super()._layer_out_info(
                concept, ModelStump(model, layer_id))
            in_channels = auto_layer_out_size[0] \
                if in_channels is None else in_channels
            act_map_size: Sequence[int] = auto_layer_out_size[-2:] if \
                act_map_size is None else act_map_size

        # Set the concept separately, since it need not be a segmentation
        # concept!
        other_settings.update(kernel_size=act_map_size)
        super().__init__(model=model, layer_id=layer_id,
                         in_channels=in_channels,
                         **{**other_settings, **dict(apply_padding=False)})
        self._concept: Concept = concept

    def forward(self, inp: torch.Tensor) -> List[torch.Tensor]:
        """Wrapper around super forward method."""
        if inp.size()[-2] != self.kernel_size[0] or \
                inp.size()[-1] != self.kernel_size[1]:
            raise ValueError(("Got tensor of size {} but expected size "
                              "(batch, {}, {}, {})").format(
                inp.size(), self.in_channels, *self.kernel_size
            ))
        outp = super().forward(inp)
        return [out.squeeze(-1).squeeze(-1).squeeze(-1) for out in outp]


class ConceptClassification2DTrainTestHandle(ConceptDetection2DTrainTestHandle):
    """Train and test handle for concept classification providing loss and
    metric defaults.
    See :py:class:`ConceptClassificationModel2D` for details on the model.
    """

    def __init__(self,
                 concept_model: ConceptClassificationModel2D,
                 data: DataTriple, *,
                 model_output_transform: TupleTransforms = False,
                 metric_input_transform: TupleTransforms = False,
                 **kwargs
                 ):  # pylint: disable=too-many-arguments
        # pylint: disable=line-too-long
        """Init.

        For details on the init parameters see the init of the super class
        :py:class:`~hybrid_learning.concepts.models.concept_models.concept_detection.ConceptDetection2DTrainTestHandle`.
        """
        # pylint: enable=line-too-long

        # Ensure that the super model does not set the model_output_transform
        super().__init__(
            concept_model=concept_model, data=data,
            model_output_transform=model_output_transform,
            metric_input_transform=metric_input_transform,
            **kwargs
        )
