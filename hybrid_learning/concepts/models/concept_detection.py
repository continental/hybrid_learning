"""Model for concept detection, and its training and evaluation handle."""
#  Copyright (c) 2020 Continental Automotive GmbH

import hashlib
import logging
from typing import Optional, Tuple, Dict, Any, Sequence, Callable

import numpy as np
import torch
import torch.nn

from .base_handles import EarlyStoppingHandle, ResettableOptimizer, \
    TrainEvalHandle
from .model_extension import ModelStump, output_size
from ..concepts import SegmentationConcept2D, ConceptTypes
from ..embeddings import ConceptEmbedding
from ...datasets import DatasetSplit, BaseDataset, ActivationDatasetWrapper, \
    DataTriple
from ...datasets.transforms import SameSize, same_padding, TupleTransforms, \
    Compose, ToDevice, OnBothSides

LOGGER = logging.getLogger(__name__)


class ConceptDetectionModel2D(torch.nn.Module):
    """Pytorch model implementation of a concept embedding for 2D conv layers.
    The model itself simply is a convolutional layer with sigmoid activation.
    The goal of this model is to tell from an activation map, which spatial
    "parts" of the activation map belong to a given concept and which not.
    These parts are windows of the concept model :py:attr:`kernel_size`.

    The model features training and evaluation functionality for concept
    analysis, i.e. for training this concept module on the activation map output
    of the given model and layer without changing the main model.

    When the model forward works as follows:

    :Input: Activation map output of a 2D convolutional layer.
    :Output:
        Heatmap showing which centers of boxes of :py:attr:`kernel_size` belong
        to :py:attr:`concept`.
        The heatmap values are the sigmoid of a convolution operation.
    """

    @property
    def concept(self) -> Optional[SegmentationConcept2D]:
        """The concept (data) for which this model is/should be trained."""
        return self._concept

    @property
    def concept_name(self) -> str:
        """The name of the associated concept if known."""
        return self._concept_name if self.concept is None else self.concept.name

    @property
    def main_model_stump(self) -> ModelStump:
        """Stump of the main model in the head of which to localize the
        concept embedding.
        Used to generate the activation maps needed for concept analysis
        training. The actual attribute is wrapped into a tuple to hide the
        parameters, since these shall not be updated; see
        https://discuss.pytorch.org/t/how-to-exclude-parameters-from-model/6151
        """
        return self._main_model_stump[0]

    @main_model_stump.setter
    def main_model_stump(self, main_model_stump: ModelStump) -> None:
        """Setter of :py:attr:`main_model_stump`."""
        self._main_model_stump = (main_model_stump,)

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
        return self.concept_layer.kernel_size

    @property
    def in_channels(self) -> int:
        """Number of input channels.
        This is the number of output channels of layer to investigate."""
        return self.concept_layer.in_channels

    @property
    def settings(self) -> Dict[str, Any]:
        """The current model settings as dictionary."""
        return dict(
            concept=self.concept,
            model=self.main_model,
            layer_id=self.layer_id,
            kernel_size=self.kernel_size,
            in_channels=self.in_channels
        )

    def __init__(self,
                 concept: Optional[SegmentationConcept2D],
                 model: Optional[torch.nn.Module], layer_id: Optional[str],
                 kernel_size: Tuple[int, int] = None, in_channels: int = None,
                 concept_name: str = None):
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
        :param concept_name: The default value for the :py:attr:`concept_name`
            property if :py:attr:`concept` is ``None``; serves as ID for
            the concept model
        """
        # pylint: enable=line-too-long
        # Parameter post-processing:
        if concept is not None:
            concept: SegmentationConcept2D = SegmentationConcept2D.new(concept)

        super(ConceptDetectionModel2D, self).__init__()

        self._main_model_stump: Tuple[ModelStump] = \
            (ModelStump(model, layer_id),) \
                if model is not None else (None,)
        """Stump of the main model in the head of which to localize the
        concept embedding. Used to generate the activation maps needed for
        concept analysis training.
        Must be wrapped into a tuple to hide the parameters from being added to
        the :py:meth:`torch.nn.Module.state_dict`, since these are not to be
        updated."""
        self._concept: Optional[SegmentationConcept2D] = concept
        """Internal storage of the concept to localize."""
        self._concept_name: Optional[str] = \
            self._concept.name if self._concept is not None else concept_name
        """Default value for :py:attr:`concept_name` property
        if :py:attr:`concept` is ``None``."""

        # automatically determine kernel_size and in_channels if one isn't given
        # (this may be time consuming as it requires one run through the model);
        # automatic determination is not possible if concept.rel_size is None,
        # in this case set the kernel_size to (1,1)
        if in_channels is None or kernel_size is None:
            if concept is None:
                raise ValueError("Concept not given, so cannot auto-infer "
                                 "sizes, but in_channels or kernel_size not "
                                 "given.")
            auto_in_channels, auto_kernel_size = \
                self._layer_out_info(concept, self.main_model_stump)
            if in_channels is not None and in_channels != auto_in_channels:
                LOGGER.warning(
                    "The number of in_channels specified for %s was %d, but the"
                    " automatically determined value was %d; taking auto one",
                    self.__class__.__name__, in_channels, auto_in_channels)
            in_channels = auto_in_channels
            kernel_size = kernel_size \
                if kernel_size is not None else auto_kernel_size

        # Layers
        assert len(kernel_size) == 2, \
            "kernel size not of len 2: {}".format(kernel_size)
        # Beware: The padding for ZeroPad2d has crude specification:
        # 1. width pad, 2. height pad
        self.padding = torch.nn.ZeroPad2d(
            padding=same_padding((kernel_size[1], kernel_size[0])))
        self.concept_layer = torch.nn.Conv2d(in_channels=in_channels,
                                             kernel_size=kernel_size,
                                             out_channels=1)
        """The Conv layer which is trained to detect windows
        in which concept is located.
        The number of input channels is automatically determined if not given
        as ``in_channels`` in the ``__init__`` call.
        (automatic determination requires one forward of the main model)."""
        self.activation = torch.nn.Sigmoid()
        """The sigmoid activation layer to obtain heatmaps in ``[0,1]``."""

    @staticmethod
    def _layer_out_info(concept: SegmentationConcept2D,
                        main_model_stump: torch.nn.Module
                        ) -> Tuple[int, Tuple[int]]:
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
        if concept.rel_size is not None and len(concept.rel_size) != len(
                layer_out_size) - 1:
            raise AttributeError(
                ("The concept size has {} image dimensions, the layer "
                 "output has {}; concept size: {}, layer out size: {}"
                 ).format(len(concept.rel_size), len(layer_out_size) - 1,
                          concept.rel_size, layer_out_size))

        # in_channels is by default the number of output filters of the layer:
        auto_in_channels: int = layer_out_size[0]

        # kernel_size is by default the percentage of the layer output size
        # given by concept size;
        # if concept.rel_size is not set, it is (1,1)
        if concept.rel_size is not None:
            if not all([0 <= s <= 1 for s in concept.rel_size]):
                raise ValueError(("Some concept size entries are invalid (not "
                                  "in [0,1]): {}").format(concept.rel_size))
            auto_kernel_size = tuple([max(1, int(round(r * o)))
                                      for r, o in zip(concept.rel_size,
                                                      layer_out_size[1:])])
        else:
            LOGGER.warning("concept.rel_size is not set; setting "
                           "auto_kernel_size to (1, 1)")
            auto_kernel_size = (1, 1)

        return auto_in_channels, auto_kernel_size

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

    def to_embedding(self) -> ConceptEmbedding:
        r"""Return the plain representation as
        :py:class:`~hybrid_learning.concepts.embeddings.ConceptEmbedding`.
        I.e.

        :as parameters: weight and bias of :py:attr:`concept_layer`, and
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
        normal_vec: np.ndarray = np.copy(
            self.concept_layer.weight.detach().cpu().numpy())
        normal_vec_length = np.linalg.norm(normal_vec)
        bias: np.ndarray = np.copy(
            self.concept_layer.bias.detach().cpu().numpy())
        support_factor: np.ndarray = - bias / (normal_vec_length ** 2)
        return ConceptEmbedding(normal_vec=normal_vec,
                                support_factor=support_factor,
                                concept=self.concept,
                                model_stump=self.main_model_stump)

    @staticmethod
    def from_embedding(embedding: ConceptEmbedding, *,
                       main_model: torch.nn.Module = None,
                       concept: SegmentationConcept2D = None
                       ) -> 'ConceptDetectionModel2D':
        # pylint: disable=line-too-long
        r"""Initialize a concept localization model from an embedding.
        The weight and bias are obtained as follows:

        :weight: The weight is the normal vector of the embedding
        :bias:
            Given the ``embedding``'s
            :py:attr:`~hybrid_learning.concepts.embeddings.ConceptEmbedding.support_factor`
            as :math:`b`, the bias calculates as
            (compare :py:meth:`to_embedding`):

            .. math:: \text{bias} = - b \cdot (|\text{weight}|^2)

        :param embedding: the embedding to use
        :param main_model: ``main_model`` to use for init of the new
            :py:class:`ConceptDetectionModel2D`; defaults to the ``embedding``'s
            :py:attr:`~hybrid_learning.concepts.embeddings.ConceptEmbedding.main_model`
        :param concept: ``concept`` to use for init of the new
            :py:class:`ConceptDetectionModel2D`
            must be valid input to the
            :py:meth:`~hybrid_learning.concepts.concepts.Concept.new` method
            of the
            :py:meth:`~hybrid_learning.concepts.concepts.SegmentationConcept2D`
            class; defaults to the ``embedding``'s
            :py:attr:`~hybrid_learning.concepts.embeddings.ConceptEmbedding.concept`
        :return: a concept localization model initialized with the embedding
            information
        """
        # pylint: enable=line-too-long
        # Value checks and defaults
        concept: Optional[SegmentationConcept2D] = concept or embedding.concept
        if concept is not None and concept.type != ConceptTypes.SEGMENTATION:
            raise ValueError(
                "Expected embedded concept to be of type segmentation, "
                "but was {}".format(embedding.concept.type))
        main_model: torch.nn.Module = main_model or embedding.main_model

        # Apply the scaling factor
        scaled_embedding = embedding.scale()
        # State dict collection (make sure the parameters are proper copies)
        weight_np = scaled_embedding.normal_vec
        bias_np = -(scaled_embedding.support_factor *
                    (np.linalg.norm(scaled_embedding.normal_vec) ** 2))
        # pylint: disable=not-callable
        state_dict = {'concept_layer.weight': torch.tensor(weight_np),
                      'concept_layer.bias': torch.tensor(bias_np)}
        # pylint: enable=not-callable

        # Model init
        c_model = ConceptDetectionModel2D(
            concept=concept, model=main_model,
            layer_id=scaled_embedding.layer_id,
            kernel_size=tuple(weight_np.shape[-2:]),
            in_channels=scaled_embedding.normal_vec.shape[1],
            concept_name=scaled_embedding.concept_name)
        c_model.load_state_dict(state_dict)

        return c_model

    def forward(self, *inp: Sequence[torch.Tensor]):
        """Torch model forward evaluation method."""
        assert len(inp) == 1, \
            "Only accepting one input tensor, but {} given".format(len(inp))
        outp = self.padding(*inp)
        outp = self.concept_layer(outp)
        outp = self.activation(outp)
        return outp


class ConceptDetection2DTrainTestHandle(TrainEvalHandle):
    r"""Train test handle for concept localization models.
    Takes the concept data of the concept model's concept, and automatically
    converts it appropriately for the concept model (see
    :py:meth:`data_from_concept`).
    """

    DEFAULT_MASK_INTERPOLATION: str = "bilinear"
    """Interpolation method used in default transforms for resizing masks
    to activation map size. Argument may be one of the modes accepted by
    :py:func:`torch.nn.functional.interpolate`.
    """

    def __init__(self,
                 concept_model: ConceptDetectionModel2D,
                 device: torch.device = torch.device("cpu"),
                 batch_size: int = 8,
                 max_epochs: int = 5,
                 loss_fn: Callable[
                     [torch.Tensor, torch.Tensor], torch.Tensor] = None,
                 metric_fns: Dict[str, Callable[
                     [torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 early_stopping_handle: EarlyStoppingHandle = None,
                 optim_handle: ResettableOptimizer = None,
                 act_map_filepath_fns: Dict[
                     DatasetSplit, Callable[[int, BaseDataset], str]] = None,
                 transforms: Callable[[torch.Tensor, torch.Tensor],
                                      Tuple[torch.Tensor, torch.Tensor]] = None,
                 model_output_transform: TupleTransforms = None
                 ):  # pylint: disable=too-many-arguments
        # pylint: disable=line-too-long
        """Init.

        For further parameter descriptions see ``__init__()`` of
        :py:class:`~hybrid_learning.concepts.models.base_handles.train_test_handle.TrainEvalHandle`.

        :param concept_model: the concept localization model to work on with
            concept.
        :param act_map_filepath_fns: dictionary of ``{split: func}`` where
            func is the functions for returning the path to an activation map
            file given index ``i`` and the base dataset behind ``split``.
            For details (e.g. on the default) see
            :py:meth:`hybrid_learning.datasets.activations_handle.ActivationDatasetWrapper.act_map_filepath_fn`.
        :param transforms: transformation the activation dataset wrapper should
            apply to the tuples of activation map and target mask tensors;
            must at least ensure that the output target mask tensor has the
            same shape as the output activation map tensor
        """
        # pylint: enable=line-too-long
        # Default transforms:
        if model_output_transform is None and transforms is None:
            model_output_transform = SameSize(
                resize_target=False,  # resize the model output to image size!
                interpolation=self.DEFAULT_MASK_INTERPOLATION)
        if transforms is not None:
            transforms_wt_device: Compose = Compose([
                OnBothSides(ToDevice(device)), transforms])
        else:
            transforms_wt_device: OnBothSides = \
                OnBothSides(ToDevice(device))

        # obtain train and test data
        data: DataTriple = self.data_from_concept(
            concept_model, transforms_wt_device, act_map_filepath_fns)
        super(ConceptDetection2DTrainTestHandle, self).__init__(
            model=concept_model,
            data=data,
            device=device,
            batch_size=batch_size,
            max_epochs=max_epochs,
            loss_fn=loss_fn,
            early_stopping_handle=early_stopping_handle,
            metric_fns=metric_fns,
            optimizer=optim_handle,
            model_output_transform=model_output_transform
        )
        concept_model.main_model_stump.to(self.device)
        # Trick to teach IDE the type of self.model:
        self.model: ConceptDetectionModel2D = self.model

    @staticmethod
    def _model_hash(model: torch.nn.Module, truncate: Optional[int] = 8):
        """Return a hex md5-hash of the main model topology for comparison
        purposes.
        Truncate to the first truncate letters if ``truncate`` is given."""
        hex_md5 = hashlib.md5(repr(model).encode()).hexdigest()
        if truncate is not None:
            if truncate <= 0:
                raise ValueError("truncate value must be > 0, but was {}"
                                 .format(truncate))
            hex_md5 = hex_md5[0:truncate]
        return hex_md5

    @classmethod
    def data_from_concept(
            cls,
            concept_model: ConceptDetectionModel2D,
            transforms: Optional[Callable[[torch.Tensor, torch.Tensor],
                                          Tuple[torch.Tensor, torch.Tensor]]],
            act_map_filepath_fns: Dict[DatasetSplit,
                                       Callable[[int, BaseDataset], str]] = None
    ) -> DataTriple:
        # pylint: disable=line-too-long
        """Data handles with activation maps for and ground truth from
        :py:attr:`~ConceptDetectionModel2D.concept`.
        The data from the concept model's concept is wrapped by an
        :py:class:`~hybrid_learning.datasets.activations_handle.ActivationDatasetWrapper`.
        Its input and ground truth are:

        :input: the required activation maps of the main model
        :ground truth:
          the segmentation masks scaled to the activation map size
          (currently scaling is done on ``__getitem__``-call of
          :py:class:`~hybrid_learning.datasets.activations_handle.ActivationDatasetWrapper`)

        :raises: :py:exc:`ValueError` if the data dimensions do not fit the
            :py:attr:`~ConceptDetectionModel2D.in_channels` of the concept
            model's :py:attr:`~ConceptDetectionModel2D.concept_layer`.
        :return: tuple of train data, test data, validation data, all with
            activation maps as outputs
        """
        # pylint: enable=line-too-long
        main_model_stump: ModelStump = concept_model.main_model_stump
        common_args = dict(
            act_map_gen=main_model_stump,
            transforms=transforms,
            model_description=(main_model_stump.wrapped_model.__class__.__name__
                               + cls._model_hash(main_model_stump)))

        splits: Dict[DatasetSplit, ActivationDatasetWrapper] = {}
        for split in (DatasetSplit.TRAIN, DatasetSplit.TEST, DatasetSplit.VAL):
            data = concept_model.concept.data[split]

            # Get the dataset_root
            # TODO: better handling of subsets and concatenations of datasets
            if hasattr(data, "dataset_root"):  # is a BaseDataset like
                dataset_root = data.dataset_root
            elif hasattr(data, "dataset"):  # is a wrapper
                if not hasattr(data.dataset, "dataset_root"):
                    raise AttributeError(
                        ("Either data split {} (type {}) or its attribute "
                         "dataset (type {}) must provide a dataset_root "
                         "attribute.").format(split.value, type(data),
                                              type(data.dataset)))
                dataset_root = data.dataset.dataset_root
            else:
                raise AttributeError(
                    ("Data split {} (type {}) must provide a dataset_root "
                     "attribute or an attribute dataset that does so."
                     ).format(split.value, type(data)))

            # Get the activation map filepath function
            act_map_filepath_fn = None if not act_map_filepath_fns else \
                act_map_filepath_fns[split]

            # Create wrapper
            splits[split] = ActivationDatasetWrapper(
                dataset=data, split=split,
                dataset_root=dataset_root,
                act_map_filepath_fn=act_map_filepath_fn,
                **common_args)

        # Validation: size checks
        in_channels: int = concept_model.concept_layer.in_channels
        for split, data in splits.items():
            act_map, _ = data[0]
            if act_map.size()[0] != in_channels:
                raise ValueError(
                    ("in_channels ({}) of concept layer does not match number "
                     "of filters in activation map of {} data sample 0 which "
                     "has size {}").format(in_channels, split.value,
                                           act_map.size()))

        return DataTriple.from_dict(splits)
