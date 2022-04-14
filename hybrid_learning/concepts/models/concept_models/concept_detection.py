"""Model for concept detection, and its training and evaluation handle.
See :py:class:`ConceptDetectionModel2D` and
:py:class:`ConceptDetection2DTrainTestHandle` for details.
"""
#  Copyright (c) 2022 Continental Automotive GmbH

import logging
import warnings
from typing import Optional, Tuple, Dict, Any, Union, List, Sequence

import torch
import torch.nn
import torch.nn.functional

from ..model_extension import ModelStump, output_size
from ...train_eval import predict_laplace, TrainEvalHandle
from ...concepts import SegmentationConcept2D, Concept
from ..embeddings import ConceptEmbedding
from ....datasets import DataTriple
from ....datasets.transforms import SameSize, same_padding, TupleTransforms, Lambda, OnInput

# pylint: disable=too-many-arguments
# pylint: disable=not-callable

LOGGER = logging.getLogger(__name__)


class ConceptDetectionModel2D(torch.nn.Module):
    """Pytorch model implementation of a concept embedding for 2D conv layers.
    The model itself simply is an ensemble (see :py:attr:`ensemble_count`) of
    convolutional layers with (optional) sigmoid activation
    (see :py:attr:`apply_sigmoid`).
    The goal of this model is to tell in each ensemble member from the
    activation map of a :py:attr:`main_model`, which spatial regions of the
    activation map belong to a given concept and which not.
    These regions are windows of the concept model :py:attr:`kernel_size`.

    Additional features compared to a normal Conv2D layer:

    - Convenience:: During init :py:attr`in_channels` and :py:attr:`kernel_size`
      can be automatically determined from a given main model and concept data.
      Also, if :py:attr:`apply_padding` is set to ``True``, a zero padding is
      automatically determined such that the output size of the convolution is
      the same as the input size (assuming constantly sized inputs).
    - Flexible architecture:: With the :py:attr:`use_bias`, the bias can be
      disabled during init (assumed to be constantly 0).
    - Storage of meta information:: If given during init,
      meta information like references to the :py:attr:`main_model` and the
      :py:attr:`concept` are kept for reproducibility.
    - Storage:: An ensemble can be turned into a generic save format that also
      captures meta and architecture specification
      (see :py:meth:`to_embedding`).

    The model forward works as follows:

    :Input: Activation map output of a 2D convolutional layer.
    :Output:
        List of heatmaps (one for each ensemble member) showing which
        centers of boxes of :py:attr:`kernel_size` belong to :py:attr:`concept`.
        The heatmap values are the sigmoid of a convolution operation
        if :py:attr:`apply_sigmoid` is ``True``.
    """

    @property
    def concept(self) -> Optional[SegmentationConcept2D]:
        """The concept for which this model was configured."""
        return self._concept

    @property
    def concept_name(self) -> Optional[str]:
        """The name of the associated concept if known.
        Defaults to the name of :py:attr:`concept` if given."""
        if self._concept_name is not None:
            return self._concept_name
        if self.concept is not None:
            return self.concept.name
        return None

    @concept_name.setter
    def concept_name(self, c_name: Optional[str]):
        """Setter for concept name."""
        self._concept_name: Optional[str] = c_name

    @property
    def main_model_stump(self) -> ModelStump:
        """Stump of the main model for which this instance was configured.
        The concept model is assumed to accept as input the output of this
        model stump (i.e. the corresponding layer of the :py:attr:`main_model`).

        Implementation detail::
        The actual attribute is wrapped into a tuple to hide the
        parameters, since these shall not be updated; see
        https://discuss.pytorch.org/t/how-to-exclude-parameters-from-model/6151
        """
        return self._main_model_stump[0]

    @property
    def main_model(self) -> torch.nn.Module:
        """Shortcut to access the main model.
         It is wrapped by :py:attr:`main_model_stump`.
        """
        return self.main_model_stump.wrapped_model \
            if self.main_model_stump is not None else None

    @property
    def layer_id(self) -> str:
        """Layer to extract concept from.
        Shortcut to access the information from :py:attr:`main_model_stump`.
        """
        return self.main_model_stump.stump_head

    @property
    def kernel_size(self) -> Tuple[int, ...]:
        """Size of the convolution kernel.
        This is the assumed concept size in activation map pixels."""
        return self.concept_layer_0.kernel_size

    @property
    def in_channels(self) -> int:
        """Number of input channels.
        This is the number of output channels of layer to investigate."""
        return self.concept_layer_0.in_channels

    @property
    def apply_sigmoid(self) -> bool:
        """Whether a sigmoid is applied to the output of the forward function
        before returning it."""
        return self.activation is not None

    @property
    def apply_padding(self) -> bool:
        """Whether a zero-padding is applied to the input of the forward
        function.
        The padding should ensure that the input equals the output size."""
        return self.padding is not None

    @property
    def settings(self) -> Dict[str, Any]:
        """The current model settings as dictionary."""
        return dict(
            concept=self.concept,
            model=self.main_model,
            layer_id=self.layer_id,
            kernel_size=self.kernel_size,
            in_channels=self.in_channels,
            concept_name=self.concept_name,
            apply_sigmoid=self.apply_sigmoid,
            apply_padding=self.apply_padding,
            ensemble_count=self.ensemble_count,
            use_laplace=self.use_laplace,
            use_bias=self.use_bias
        )

    def __init__(self,
                 concept: Optional[SegmentationConcept2D] = None,
                 model: Optional[torch.nn.Module] = None,
                 layer_id: Optional[str] = None,
                 kernel_size: Tuple[int, int] = None, in_channels: int = None,
                 concept_name: Optional[str] = None,
                 apply_sigmoid: bool = True,
                 apply_padding: bool = True,
                 ensemble_count: int = 1,
                 use_laplace: bool = False,
                 use_bias: bool = True):
        # pylint: disable=line-too-long
        """Init.

        :param model: model the concept should be embedded in;
            used to create (and later accessible in)
            :py:attr:`main_model_stump`;
            used for :py:attr:`kernel_size` and :py:attr:`in_channels`
            auto-inference
        :param layer_id: the layer index in
            :py:meth:`~torch.nn.Module.state_dict`, the output of which is to
            be fed to the the concept model; used to create (and later
            accessible) in :py:attr:`main_model_stump`;
            used for :py:attr:`kernel_size` and :py:attr:`in_channels`
            auto-inference
        :param concept: Concept to train for; must be a segmentation concept
            featuring ground truth masks; used for :py:attr:`kernel_size` and
            :py:attr:`in_channels` auto-inference
        :param in_channels: Number of filters of the
            :py:class:`~torch.nn.Conv2d`-Layer to analyse;
            the value is automatically determined if ``in_channels`` or
            ``kernel_size`` is ``None``;
            an automatically generated value overwrites a given value with a
            warning
        :param kernel_size: Size in activation map pixels of a window for
            which to assess whether it is part of the ``concept`` or not;
            by default it is determined by the relative sizes in the concept's
            :py:attr:`~hybrid_learning.concepts.concepts.SegmentationConcept2D.rel_size`
            and the layer output size;
            if ``concept.rel_size`` is not set, :py:attr:`kernel_size` is set to
            ``(1, 1)`` with a warning
        :param concept_name: The concept name identifier to use for
            :py:attr:`concept_name`; defaults to the name
            given in :py:attr:`concept`
        :param apply_sigmoid: see :py:attr:`apply_sigmoid`
        :param apply_padding: see :py:attr:`apply_padding`
        :param ensemble_count: number of deep ensemble models,
            see :py:attr:`ensemble_count`
        :param use_laplace: if true, the covariance of the prediction are
            approximated using laplace
        :param use_bias: see :py:attr:`use_bias`
        """
        # pylint: enable=line-too-long
        super().__init__()
        self.use_laplace: bool = use_laplace
        """Whether training handles should use Laplace approximation."""
        self.ensemble_count: int = ensemble_count  # TODO: make read-only
        """Number of deep ensemble models.
        This is also the first dimension of the forward output.
        Each ensemble member simply is a separate convolutional layer,
        and all members are run in parallel."""
        self.use_bias: bool = use_bias  # TODO: make read-only
        """Whether the convolution should have and learn a bias, or
        the bias should be constantly 0."""

        # region: Meta data
        if concept is not None:
            concept: SegmentationConcept2D = SegmentationConcept2D.new(concept)
        self._concept: Optional[SegmentationConcept2D] = concept
        """Internal storage of the concept to localize.
        See :py:attr:`concept`."""
        self._concept_name: Optional[str] = concept_name
        """Default value for :py:attr:`concept_name` property
        if :py:attr:`concept` is ``None``."""
        self._main_model_stump: Tuple[ModelStump] = (None,) \
            if model is None or layer_id is None else \
            (ModelStump(model, layer_id),)
        """Stump of the main model in the head of which to localize the
        concept embedding. Assumed to be used to generate the activation maps
        needed for concept analysis training.
        Must be wrapped into a tuple to hide the parameters from being added to
        the :py:meth:`torch.nn.Module.state_dict`, since these are not to be
        updated."""
        # endregion

        # Automatically determine kernel_size and in_channels if one isn't given
        # (this may be time consuming as it requires one run through the model);
        # automatic determination is not possible if concept.rel_size is None,
        # in this case set the kernel_size to (1,1)
        with torch.set_grad_enabled(False):
            in_channels, kernel_size = self._determine_channels_and_kernel(
                kernel_size, in_channels, self.main_model_stump, self.concept)

        # region: Layers
        assert len(kernel_size) == 2, \
            "kernel size not of len 2: {}".format(kernel_size)
        # Beware: The padding for ZeroPad2d has crude specification:
        # 1. width pad, 2. height pad
        self.padding: Optional[torch.nn.ZeroPad2d] = \
            None if not apply_padding else torch.nn.ZeroPad2d(
                padding=same_padding((kernel_size[1], kernel_size[0])))
        """The padding to apply before the convolution.
        Defaults to a padding such that the output size equals the input size
        if ``apply_padding`` is set to ``True`` during init. If set to ``None``,
        no padding is applied."""

        for i in range(ensemble_count):
            # TODO: use ModuleList
            concept_layer = torch.nn.Conv2d(in_channels=in_channels,
                                            kernel_size=kernel_size,
                                            out_channels=1,
                                            bias=self.use_bias)
            if self.use_laplace:
                unfolded_input_tensor = torch.nn.functional.unfold(
                    torch.zeros(1, in_channels,
                                concept_layer.kernel_size[0],
                                concept_layer.kernel_size[1]),
                    kernel_size=concept_layer.kernel_size,
                    padding=concept_layer.padding,
                    stride=concept_layer.stride)
                concept_layer.register_buffer('hessian', torch.zeros(
                    unfolded_input_tensor.shape[1] + (
                        1 if concept_layer.bias else 0),
                    unfolded_input_tensor.shape[1] + (
                        1 if concept_layer.bias else 0)), True)
                concept_layer.register_buffer('var0', torch.zeros(1), True)
            setattr(self, f'concept_layer_{i}', concept_layer)

        self.activation: Optional[torch.nn.Sigmoid] = \
            torch.nn.Sigmoid() if apply_sigmoid else None
        """The  activation layer to obtain heatmaps in ``[0,1]``.
        Defaults to a sigmoid if ``apply_sigmoid`` is set to ``True`` during
        init. If set to ``None``, no activation is applied."""

    @classmethod
    def _determine_channels_and_kernel(
            cls, kernel_size: Optional[Tuple[int, int]],
            in_channels: Optional[int],
            main_model_stump: Optional[ModelStump],
            concept: Optional[SegmentationConcept2D]
    ) -> Tuple[int, Tuple[int, int]]:
        """Determine the ``in_channel`` and ``kernel_size`` information from
        the available inputs.
        If ``in_channels`` is not given, the ``main_model_stump`` and one
        sample of the ``concept`` (data) are used to determine it using.
        If ``kernel_size`` is not given, also the ``rel_size`` of the
        ``concept`` is used if given, else ``kernel_size`` is set to ``(1, 1)``.
        Internally, :py:meth:`_layer_out_info` is used.

        For the parameters see ``__init__``.
        """
        if in_channels is not None and kernel_size is not None:
            return in_channels, kernel_size

        # region: Value checks
        _missing = "in_channels" if in_channels is None else "kernel_size"
        if concept is None:
            raise ValueError("concept not given, so cannot auto-infer "
                             "sizes, but {} isn't given.".format(_missing))
        if main_model_stump is None:
            raise ValueError("model or layer_id not given, so cannot auto-infer"
                             " sizes, but {} isn't given.".format(_missing))
        # endregion

        auto_layer_out_size, auto_kernel_size = \
            cls._layer_out_info(concept, main_model_stump)

        # in_channels is by default the number of output filters of the layer:
        auto_in_channels: int = auto_layer_out_size[0]

        # region: Warnings for deviations
        if in_channels is not None and in_channels != auto_in_channels:
            LOGGER.warning(
                "The number of in_channels specified for %s was %d, but the"
                " automatically determined value was %d",
                cls.__name__, in_channels, auto_in_channels)
        if kernel_size is not None and kernel_size != auto_kernel_size:
            LOGGER.warning(
                "The kernel_size specified for %s was %d, but the"
                " automatically determined value was %d",
                cls.__name__, kernel_size, auto_kernel_size)
        # endregion

        in_channels = auto_in_channels if in_channels is None else in_channels
        kernel_size = auto_kernel_size if kernel_size is None else kernel_size
        return in_channels, kernel_size

    @staticmethod
    def _layer_out_info(concept: Concept,
                        main_model_stump: torch.nn.Module
                        ) -> Tuple[torch.Size, Tuple[int, int]]:
        # pylint: disable=line-too-long
        """Extract channel and kernel size information from model output.
        This is done by collecting the layer output size from one forward run
        of the model.
        It is then assumed that the layer output is a tensor of shape
        ``(output channels/filters, height, width, ...)``, where
        ``height, width, ...`` is activation map shape information that
        should have the same number of dimensions as the
        :py:attr:`~hybrid_learning.concepts.concepts.SegmentationConcept2D.rel_size` of
        the ``concept``, if this is set.

        :param main_model_stump: the model of which to analyse the output;
            output must be a single tensor with size of shape
            ``(output channels/filters, width, height, ...)``
        :param concept: the concept from which to draw the dummy sample size
            and the concept size
        :return: tuple of
            :in_channels: number of output channels of the layer) and of
            :kernel_size:
                the size in activation map pixels the kernel must have to
                provide (up to rounding) the same aspect ratio as specified
                in the :py:attr:`~hybrid_learning.concepts.concepts.SegmentationConcept2D.rel_size`
                of the ``concept`` if this is set;
                if ``concept.rel_size`` is not set, ``kernel_size`` is ``(1,1)``
        :return: tuple of ``(in_channels, kernel_size)``
        """
        # pylint: enable=line-too-long
        inp, _ = concept.train_data[0]
        # layer output size without batch dimension:
        layer_out_size = output_size(main_model_stump, input_size=inp.size())

        # Some size checks:
        # assuming layer_out_size = batch + (filters, width, height)
        if len(layer_out_size) != 3:
            raise AttributeError(
                ("The size of layer {} output was not of shape "
                 "(filters, width, height), but was {}"
                 ).format(main_model_stump.stump_head, layer_out_size))

        # assuming layer_out_size[1:] gives same image dimensions as
        # concept.rel_size
        rel_size = concept.rel_size \
            if isinstance(concept, SegmentationConcept2D) else None
        if rel_size is not None and len(rel_size) != len(layer_out_size) - 1:
            raise AttributeError(
                ("The concept size has {} image dimensions, the layer "
                 "output has {}; concept size: {}, layer out size: {}"
                 ).format(len(concept.rel_size), len(layer_out_size) - 1,
                          concept.rel_size, layer_out_size))

        # kernel_size is by default the percentage of the layer output size
        # given by concept size;
        # if concept.rel_size is not set, it is (1,1)
        if rel_size is not None:
            if not all([0 <= s <= 1 for s in concept.rel_size]):
                raise ValueError(("Some concept size entries are invalid (not "
                                  "in [0,1]): {}").format(concept.rel_size))
            auto_kernel_size = tuple([max(1, int(round(r * o)))
                                      for r, o in zip(concept.rel_size,
                                                      layer_out_size[1:])])
        else:
            auto_kernel_size = (1, 1)

        return layer_out_size, auto_kernel_size

    def reset_parameters(self) -> None:
        """Randomly (re)initialize weight and bias."""

        @torch.no_grad()
        def reset_params(module: torch.nn.Module) -> None:
            """Try to call a parameter reset method within module."""
            if hasattr(module, 'reset_parameters'):
                # noinspection PyCallingNonCallable
                module.reset_parameters()

        for child in self.children():
            reset_params(child)

    def to_embedding(self) -> List[ConceptEmbedding]:
        r"""Return the plain representation of the ensemble as list of
        :py:class:`~hybrid_learning.concepts.models.embeddings.ConceptEmbedding`.
        I.e.

        :as parameters: weight and bias of the concept layers, and
        :as meta info:
            the :py:attr:`concept` and :py:attr:`main_model`
            with :py:attr:`layer_id`.

        .. note::
            This must be a deep copy to avoid overwriting in a consecutive
            training session.

        The resulting embedding describes the decision hyperplane of the
        concept model. Its normal vector :math:`n` is the concept layer weight.
        The orthogonal support vector given by :math:`b\cdot n` for a scalar
        factor :math:`b` must fulfill

        .. math::
            \forall v: (v - b\cdot n) \circ n
            = d(v)
            = (v \circ \text{weight}) + \text{bias}

        i.e.

        .. math::
            n = \text{weight} \quad\text{and}\quad
            b = - \frac{\text{bias}} {|\text{weight}|^2}.

        Here, :math:`d(v)` is the signed distance measure of a vector
        from the hyperplane, i.e.

        .. math::
            d(v)
            \begin{cases}
                > 0  & \text{iff vector yields a positive prediction,}\\
                \equiv 0 & \text{iff vector on decision boundary hyperplane,}\\
                < 0  & \text{iff vector yields a negative prediction.}
            \end{cases}
        """
        embeddings_list = []
        for i in range(self.ensemble_count):
            concept_layer = getattr(self, f'concept_layer_{i}')
            state = {key: value.detach().cpu().numpy() for key, value in
                     concept_layer.state_dict().items()}
            # Preparation for rename model -> main_model
            main_model = self.settings.get('model', None)
            common_setts = dict(**self.settings, state_dict=state,
                                main_model=main_model,
                                model_stump=self.main_model_stump)
            embeddings_list.append(ConceptEmbedding(
                **common_setts,
                normal_vec_name='weight',
                bias_name='bias' if concept_layer.bias else None))
        return embeddings_list

    @staticmethod
    def from_embedding(embeddings_list: Union[ConceptEmbedding,
                                              Sequence[ConceptEmbedding]],
                       legacy_warnings: bool = True,
                       **kwargs) -> 'ConceptDetectionModel2D':
        # pylint: disable=line-too-long
        r"""Initialize a concept localization model from an embedding.
        The weight and bias are obtained as follows:

        :weight: The weight is the normal vector of the embedding
        :bias:
            Given the ``embedding``'s
            :py:attr:`~hybrid_learning.concepts.models.embeddings.ConceptEmbedding.support_factor`
            as :math:`b`, the bias calculates as
            (compare :py:meth:`to_embedding`):

            .. math:: \text{bias} = - b \cdot (|\text{weight}|^2)

        :param embeddings_list: the embeddings to use
        :param legacy_warnings: whether to give warnings about legacy, non-captured
            embedding attributes
        :param kwargs: any keyword arguments to the concept model
            (overwrite the values obtained from embedding)
        :return: a concept localization model initialized with the embedding
            information
        """
        # pylint: enable=line-too-long

        # region Value check
        # Add option to hand over single concept model for backward compat.
        if isinstance(embeddings_list, ConceptEmbedding):
            embeddings_list: List[ConceptEmbedding] = [embeddings_list]
        ensemble_count: int = kwargs.get('ensemble_count', len(embeddings_list))
        if len(embeddings_list) != ensemble_count:
            LOGGER.warning(
                f'Data mismatch: Ensemble count = {ensemble_count} '
                f'Embeddings = {len(embeddings_list)} ... '
                f'fixing count (assuming singular best embedding)')
            ensemble_count = len(embeddings_list)
        kwargs['ensemble_count'] = ensemble_count
        # endregion

        # Model init
        kwargs = {'model': kwargs.get('main_model', None),
                  'in_channels': embeddings_list[0].normal_vec.shape[1],
                  'kernel_size': embeddings_list[0].kernel_size,
                  **embeddings_list[0].meta_info,
                  **kwargs}
        # Treat legacy naming:
        if 'main_model' in kwargs:
            del kwargs['main_model']
        if 'model_stump' in kwargs:
            del kwargs['model_stump']
        if 'use_vi' in kwargs:
            del kwargs['use_vi']
        c_model = ConceptDetectionModel2D(**kwargs)
        state_dict = {}
        for i, embedding in enumerate(embeddings_list):
            # Apply the scaling factor
            scaled_embedding = embedding.scale()
            emb_state_dict: Dict[str, torch.Tensor] = {
                f'concept_layer_{i}.{key}': torch.tensor(value)
                for key, value in scaled_embedding.state_dict.items()}
            # Consistency check
            if legacy_warnings:
                for key in (k for k in emb_state_dict.keys()
                            if k not in c_model.state_dict().keys()):
                    logging.getLogger().warning("The key %s from state_dict of embedding %s"
                                                " is not in the concept model state_dict with keys %s",
                                                key, i, c_model.state_dict().keys())
            # Update & proper reshaping
            state_dict.update({key: value.view(c_model.state_dict()[key].size())
                               for key, value in emb_state_dict.items() if key in c_model.state_dict().keys()})

        c_model.load_state_dict(state_dict)

        return c_model

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Torch model forward evaluation method."""
        # region Size checks
        # Suppress tracer warning
        # https://github.com/pytorch/pytorch/issues/19364#issuecomment-484225256
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore',
                                    category=torch.jit.TracerWarning,
                                    module=r'.*')
            if len(inp.size()) < 4 or inp.size()[-3] - self.in_channels != 0:
                raise ValueError(("Got tensor of size {} but expected size "
                                  "(batch, in_channels, height, width) with "
                                  "in_channels={}"
                                  ).format(inp.size(), self.in_channels))
        # endregion

        # Laplace prediction in eval mode
        if not self.training and self.use_laplace and self.concept_layer_0.var0 > 0:
            outps: torch.Tensor = predict_laplace(self, inp)
            if self.apply_sigmoid:
                outps = self.activation(outps)
            return outps

        # Normal prediction
        outp: torch.Tensor = inp
        if self.apply_padding:
            outp: torch.Tensor = self.padding(outp)

        outp_list: List[torch.Tensor] = []
        for i in range(self.ensemble_count):
            out_val = getattr(self, f'concept_layer_{i}')(outp)
            if self.apply_sigmoid:
                out_val = self.activation(out_val)
            outp_list.append(out_val)

        return torch.stack(outp_list)


class ConceptDetection2DTrainTestHandle(TrainEvalHandle):
    r"""Train test handle for concept localization models.
    Applies sensible defaults to ``model_output_transform``.
    """

    DEFAULT_MASK_INTERPOLATION: str = "bilinear"
    """Interpolation method used in default transforms for resizing masks
    to activation map size. Argument may be one of the modes accepted by
    :py:func:`torch.nn.functional.interpolate`.
    """

    def __init__(self,
                 concept_model: ConceptDetectionModel2D,
                 data: DataTriple, *,
                 model_output_transform: TupleTransforms = None,
                 metric_input_transform: TupleTransforms = None,
                 **kwargs
                 ):  # pylint: disable=too-many-arguments
        # pylint: disable=line-too-long
        """Init.

        For further parameter descriptions see ``__init__()`` of
        :py:class:`~hybrid_learning.concepts.train_eval.base_handles.train_test_handle.TrainEvalHandle`.

        :param concept_model: the concept localization model to work on with
            concept.
        :param data: data for the concept model, i.e. Sequence of tuples
            ``(activation, mask)``
        """
        # pylint: enable=line-too-long
        # Default transforms: resize model to image output
        # (take care of clamping after bilinear upscaling!
        if model_output_transform is None:
            model_output_transform = (
                    SameSize(resize_target=False,
                             interpolation=self.DEFAULT_MASK_INTERPOLATION)
                    + OnInput(Lambda(lambda t: t.clamp(0, 1)))
            )

        super().__init__(
            model=concept_model, data=data,
            model_output_transform=model_output_transform,
            metric_input_transform=metric_input_transform,
            **kwargs
        )
        # Trick to teach IDE the type of self.model:
        self.model: ConceptDetectionModel2D = self.model
