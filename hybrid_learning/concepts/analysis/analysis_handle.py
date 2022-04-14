"""Handle for conducting a concept embedding analysis.

For details on the workflow of a concept analysis see
:py:meth:`ConceptAnalysis.analysis`.
In short:

:Input: All of

    - The *concept* (defined via concept data)
    - The *main model*
    - The *layers* to analyse and compare

:Output: All of

    - The *layer* hosting the best embedding,
    - The *best embedding*,
    - The *quality metric values* for the best embedding
"""
#  Copyright (c) 2022 Continental Automotive GmbH

import enum
import hashlib
import logging
import os
from typing import Tuple, Dict, Any, Sequence, Callable, List, Optional, \
    Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

from .results import AnalysisResult, BestEmbeddingResult
from . import visualization as vis
from ..train_eval.kpis import aggregating_kpis, batch_kpis as losses
from ..concepts import ConceptTypes, Concept, SegmentationConcept2D
# For type hints:
from ..models import ConceptDetectionModel2D, ConceptDetection2DTrainTestHandle, ConceptEmbedding
from ..models.model_extension import ModelStump
from ...datasets import data_visualization as datavis, \
    ActivationDatasetWrapper, BaseDataset, DataTriple, DatasetSplit, caching

LOGGER = logging.getLogger(__name__)


def data_for_concept_model(
        concept_model: ConceptDetectionModel2D = None,
        main_model_stump: Optional[torch.nn.Module] = None,
        concept: Optional[Concept] = None,
        in_channels: Optional[int] = None,
        transforms: Optional[
            Callable[[torch.Tensor, torch.Tensor],
                     Tuple[torch.Tensor, torch.Tensor]]] = None,
        cache_builder: Optional[Callable[[BaseDataset,
                                          ConceptDetectionModel2D],
                                         caching.Cache]] = None,
        cache_root: str = None, cache_in_memory: bool = False,
        device: Union[str, torch.device] = None
) -> DataTriple:
    # pylint: disable=line-too-long
    """Data handles with activation maps for and ground truth from
    :py:attr:`~hybrid_learning.concepts.models.concept_models.concept_detection.ConceptDetectionModel2D.concept`.
    The data from the concept model's concept is wrapped by an
    :py:class:`~hybrid_learning.datasets.activations_handle.ActivationDatasetWrapper`.
    Its input and ground truth are:

    :input: the required activation maps of the main model
    :ground truth:
      the segmentation masks scaled to the activation map size
      (currently scaling is done on ``__getitem__``-call of
      :py:class:`~hybrid_learning.datasets.activations_handle.ActivationDatasetWrapper`)

    :param concept_model: the concept model (with concept and main model)
        to generate the wrapped dataset from; if not set,
        ``main_model_stump``, ``concept``, and ``in_channels`` are used
    :param main_model_stump: the model stump that generates the activations
    :param concept: the concept the data of which is to be wrapped
    :param in_channels: (optional for validation purposes) the input
        channels for the concept model
    :param transforms: the transformations to add to each wrapper instance
    :param cache_builder: a builder that accepts the dataset to be wrapped
        and the concept model for which to wrap it, and returns a cache to
        be registered to the dataset wrapper; should have no side effects;
        defaults to a cache tuple of each a cache cascade for the activations
        and the masks
    :param cache_root: in case ``cache_root`` is given instead of
        ``cache_builder``, a default cache builder is defined using
        ``cache_root`` and ``default_cache_roots``.
    :param cache_in_memory: apply in-memory caches as default cache;
        if cache_root is also set, use as default a cache cascades of in-memory
        then file cache
    :param device: the device to move all dataset items to after loading
    :raises: :py:exc:`ValueError` if the data dimensions do not fit the
        :py:attr:`~hybrid_learning.concepts.models.concept_models.concept_detection.ConceptDetectionModel2D.in_channels`
        of the concept model's concept layers
    :return: tuple of train data, test data, validation data, all with
        activation maps as outputs
    """
    # pylint: enable=line-too-long
    # region default values and value checks
    if concept_model is not None:
        main_model_stump: ModelStump = concept_model.main_model_stump
        concept: Concept = concept_model.concept
        # validation
        in_channels: int = concept_model.in_channels
    elif main_model_stump is None or concept is None:
        raise ValueError(
            ("concept_model not given but concept or main_model_stump "
             "None;\nconcept: {}\nmain_model_stump: {}").format(
                concept, main_model_stump))

    if cache_builder is None and cache_root is not None:
        def default_cache_builder(dat, **kwargs) -> caching.Cache:
            """Return cache to register given dataset and concept model."""
            caches: List[Tuple[caching.Cache, caching.Cache]] = []
            if cache_in_memory:
                caches.append((caching.TensorDictCache(),
                               caching.TensorDictCache()))
            if cache_root is not None:
                act_root, mask_root = default_cache_roots(
                    dat, cache_root=cache_root, **kwargs)
                if transforms is not None:
                    mask_root += "_for_act_maps"
                caches.append((
                    caching.PTCache(cache_root=act_root, dtype=torch.bfloat16),
                    caching.PTCache(cache_root=mask_root, dtype=torch.bfloat16))
                )
            return caching.CacheTuple(*[sum(ca, None) for ca in zip(*caches)])

        cache_builder = default_cache_builder
    # endregion

    common_args = dict(
        act_map_gen=main_model_stump,
        transforms=transforms,
        device=device
    )

    splits: Dict[DatasetSplit, ActivationDatasetWrapper] = {}
    for split in (DatasetSplit.TRAIN, DatasetSplit.TEST, DatasetSplit.VAL):
        data = concept.data[split]

        # Create wrapper
        splits[split]: ActivationDatasetWrapper = \
            ActivationDatasetWrapper(dataset=data, split=split,
                                     **common_args)
        if cache_builder is not None:
            cache = cache_builder(data, main_model_stump=main_model_stump,
                                  concept=concept)
            splits[split].transforms_cache = cache
            # Independent act map and mask caches:
            if isinstance(cache, caching.CacheTuple) and transforms is None:
                splits[split].act_maps_cache = cache.caches[0]

    # Validation: size checks
    if in_channels is not None:
        for split, data in splits.items():
            act_map, _ = data[0]
            if act_map.size()[0] != in_channels:
                raise ValueError(
                    ("in_channels value of {} for concept layer does not match "
                     "number of filters in activation map of {} data sample"
                     " 0 which has size {}").format(
                        in_channels, split.value, act_map.size()))

    return DataTriple.from_dict(splits)


def model_hash(model: torch.nn.Module, truncate: Optional[int] = 8):
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


def default_cache_roots(wrapped_data: BaseDataset,
                        concept_model: ConceptDetectionModel2D = None,
                        main_model_stump: ModelStump = None,
                        concept: Concept = None,
                        cache_root: str = None) -> Tuple[str, str]:
    r"""Given a dataset to be wrapped and a concept model, provide
    standard cache roots paths.
    Either give a ``concept_model`` or its defining parameters, namely
    ``main_model_stump`` and the ``concept``.
    See :py:func:`default_mask_cache_root` and
    :py:func:`default_act_cache_root` for details.
    """
    # region default values and value checks
    cache_root = cache_root or os.path.join("dataset", "cache")
    concept = concept or getattr(concept_model, "concept", None)
    if main_model_stump is None and concept_model is None:
        raise ValueError("concept_model not given but main_model_stump None")
    main_model_stump = main_model_stump or getattr(concept_model,
                                                   "main_model_stump", None)
    # endregion
    act_root = default_act_cache_root(
        wrapped_data, cache_root, main_model_stump)
    masks_root = default_mask_cache_root(
        wrapped_data, cache_root, getattr(concept, "name", None))
    return act_root, masks_root


def default_act_cache_root(wrapped_data: Union[BaseDataset, Subset],
                           cache_root: str,
                           main_model_stump: ModelStump) -> str:
    """Determine the default activation cache root as
    ``<cache_root>/<img_base>_<mainmodelclassname>-<layerid>``.
    ``<img_base>`` is the base name of the ``wrapped_data.dataset_root``."""
    if not isinstance(wrapped_data, (Subset, BaseDataset)):
        raise TypeError(("Cannot handle datasets other than Subset or "
                         "BaseDataset, but was of type {} (dataset: {})"
                         ).format(type(wrapped_data), wrapped_data))
    if isinstance(wrapped_data, BaseDataset):
        img_base: str = os.path.basename(wrapped_data.dataset_root)
    else:
        # noinspection PyUnresolvedReferences
        img_base: str = os.path.basename(wrapped_data.dataset.dataset_root)
    # region Collect base names
    act_basename = "{img_base}_{model_hash}-{module_id}".format(
        img_base=img_base,
        model_hash=(main_model_stump.wrapped_model.__class__.__name__
                    + model_hash(main_model_stump)),
        module_id=main_model_stump.stump_head
    )
    act_root: str = os.path.join(cache_root, "activations", act_basename)
    return act_root


def default_mask_cache_root(wrapped_data: Union[BaseDataset, Subset],
                            cache_root: str = os.path.join("dataset", "cache"),
                            concept_name: str = None) -> str:
    """Determine the default activation data mask cache root.
    It is ``<cache_root>/<base>`` with ``<base>`` being the base name of the
    ``wrapped_data.masks_root`` if this is available,
    else ``<img_base>_<concept_name>`` with ``<img_base>`` the base name of
    the ``wrapped_data.dataset_root``.
    """
    # Get mask_root property or set default:
    if hasattr(wrapped_data, "masks_root"):
        masks_path = wrapped_data.masks_root
    elif hasattr(wrapped_data, "dataset") and \
            hasattr(wrapped_data.dataset, "masks_root"):
        masks_path = wrapped_data.dataset.masks_root
    elif concept_name is None:
        raise ValueError(("Cannot determine mask_root of dataset {} of type {} "
                          "and no default concept or concept_model is given")
                         .format(wrapped_data, type(wrapped_data)))
    else:
        if isinstance(wrapped_data, BaseDataset):
            img_base = os.path.basename(wrapped_data.dataset_root)
        else:
            # noinspection PyUnresolvedReferences
            img_base = os.path.join(wrapped_data.dataset.dataset_root)
        masks_path = os.path.join(
            "masks", "_".join([spec for spec in (img_base, concept_name)
                               if spec is not None]))
    masks_basename = os.path.basename(masks_path)
    masks_root_root = os.path.basename(os.path.dirname(masks_path))
    masks_root: str = os.path.join(cache_root, masks_root_root, masks_basename)
    return masks_root


class EmbeddingReduction(enum.Enum):
    """Aggregator callables to get the mean from a list of embeddings."""

    MEAN_NORMALIZED_DIST = (ConceptEmbedding.mean,)
    """Embedding with distance function the mean of those of the
    normed representations"""
    MEAN_DIST = (ConceptEmbedding.mean_by_distance,)
    """Embedding with distance the mean of the distance functions"""
    MEAN_ANGLE = (ConceptEmbedding.mean_by_angle,)
    """Embedding with distance function the mean of the distance functions
    weighted by cosine distance of the normal vectors"""
    FIRST = (ConceptEmbedding.first,)
    """First element of the result list"""

    DEFAULT = MEAN_NORMALIZED_DIST
    """The default instance to be used."""

    def __init__(self,
                 func: Callable[[Sequence[ConceptEmbedding]],
                                ConceptEmbedding]):
        """The init routine for enum members makes function available as
        instance fields.
        It is automatically called for all defined enum instances.
        """
        self.function: Callable[[Sequence[ConceptEmbedding]],
                                ConceptEmbedding] = func
        """Actual function that reduces a list of embeddings to a new one.

        .. note::
            The function is manually saved as attribute during ``__init__``
            due to the following issue:
            Enums currently do not support functions as values, as explained in
            `this
            <https://stackoverflow.com/questions/40338652>`_ and
            `this discussion
            <https://mail.python.org/pipermail/python-ideas/2017-April/045435.html>`_.
            The chosen workaround follows
            `this suggestion <https://stackoverflow.com/a/30311492>`_
            *(though the code is not used)*.
        """

    def __call__(self,
                 embeddings: Sequence[ConceptEmbedding]) -> ConceptEmbedding:
        """Call aggregation function behind the instance on the embeddings."""
        return self.function(embeddings)


class ConceptAnalysis:  # pylint: disable=too-many-instance-attributes
    r"""Handle for conducting a concept embedding analysis.
    Saves the analysis settings and can run a complete analysis.

    The core methods are:

    - :py:meth:`analysis`: plain analysis (collect
      :math:`\text{cross_val_runs}\cdot\text{num_val_splits}`
      embeddings for each layer in :py:attr`layer_infos`)
    - :py:meth:`best_embedding`: aggregate embeddings of an analysis per layer,
      then choose best one
    - :py:meth:`best_embedding_with_logging`: combination of the latter two
      with automatic logging and result saving
    """

    @property
    def num_runs(self) -> int:
        """The total number of runs that are conducted per layer."""
        return self.cross_val_runs * self.num_val_splits

    def __init__(self,
                 concept: Concept,
                 model: torch.nn.Module,
                 layer_infos: Union[Dict[str, Dict[str, Any]],
                                    Sequence[str]] = None,
                 cross_val_runs: int = 1,
                 num_val_splits: int = 5,
                 emb_reduction: EmbeddingReduction = EmbeddingReduction.DEFAULT,
                 concept_model_args: Dict[str, Any] = None,
                 train_val_args: Dict[str, Any] = None,
                 data_args: Dict[str, Any] = None,
                 after_layer_hook: Callable[[AnalysisResult,
                                             ConceptDetection2DTrainTestHandle],
                                            Any] = None
                 ):  # pylint: disable=too-many-arguments
        """Init.

        :param concept: concept to find the embedding of
        :param model: the DNN
        :param layer_infos: information about the layers in which to look for
            the best concept embedding; it may be given either as sequence of
            layer IDs or as dict where the indices are the layer keys in
            the model's :py:meth:`torch.nn.Module.named_modules` dict;
            used keys:

            - kernel_size: fixed kernel size to use for this layer
              (overrides value from ``concept_model_args``)
            - lr: learning rate to use

        :param num_val_splits: the number of validation splits to use for
            each cross-validation run
        :param cross_val_runs: for a layer, several concept models are
            trained in different runs; the runs differ by model initialization,
            and the validation data split;
            ``cross_val_runs`` is the number of cross-validation runs,
            i.e. collections of runs with num_val_splits distinct validation
            sets each
        :param emb_reduction: aggregation function to reduce list of
            embeddings to one
        :param concept_model_args: dict with arguments for the concept model
            initialization
        :param train_val_args: any further arguments to initialize the concept
            model handle; a loss and a metric are added by default
        :param data_args: any further arguments to initialize the training and
            eval data tuple using :py:meth:`data_for_concept_model`
        :param after_layer_hook: see :py:attr:`after_layer_hook`
        """
        # region Value checks
        if not concept.type == ConceptTypes.SEGMENTATION:
            raise NotImplementedError(
                ("Analysis only available for segmentation concepts,"
                 "but concept was of type {}").format(concept.type))
        if num_val_splits < 1:
            raise ValueError("num_val_splits must be positive int but was {}"
                             .format(num_val_splits))
        if cross_val_runs < 1:
            raise ValueError("cross_val_runs must be positive int but was {}"
                             .format(cross_val_runs))
        # endregion

        self.concept: Concept = concept
        """The concept to find the embedding for."""
        self.model: torch.nn.Module = model
        """The model in which to find the embedding."""
        self.layer_infos: Dict[str, Dict[str, Any]] = layer_infos \
            if isinstance(layer_infos, dict) \
            else {l_id: {} for l_id in layer_infos}
        """Information about the layers in which to look for the best concept
        embedding; the indices are the layer keys in the model's
        :py:meth:`torch.nn.Module.named_modules` dict"""
        self.cross_val_runs: int = cross_val_runs
        """The number of cross-validation runs to conduct for each layer.
        A cross-validation run consists of :py:attr:`num_val_splits` training
        runs with distinct validation sets. The resulting embeddings of all
        runs of all cross-validation runs are then used to obtain the layer's
        best concept embedding."""
        self.num_val_splits: int = num_val_splits
        """The number of validation splits per cross-validation run.
        If set to 1, no cross-validation is conducted but simply a number of
        :py:attr:`cross_val_runs` training runs.
        See :py:meth:`analysis_for_layer`."""
        self.emb_reduction: EmbeddingReduction = emb_reduction
        """Aggregation function to reduce a list of embeddings from several
        runs to one."""
        self.train_val_args: Dict[str, Any] = dict(train_val_args) \
            if train_val_args is not None else {}
        """Any training and evaluation arguments for the concept model
        initialization."""
        self.train_val_args.setdefault('loss_fn', losses.TverskyLoss())
        self.train_val_args.setdefault('metric_fns', {'set_iou': aggregating_kpis.SetIoU()})
        self.concept_model_args: Dict[str, Any] = concept_model_args \
            if concept_model_args is not None else {}
        """Any arguments for initializing a new concept model."""
        self.data_args: Dict[str, Any] = data_args \
            if data_args is not None else {}
        """Any arguments except for the concept model specifiers to the
        concept model data initializer.
        See :py:meth:`data_for_concept_model` for details."""

        if after_layer_hook is not None and not callable(after_layer_hook):
            raise ValueError("after_layer_hook given but not callable; was {}"
                             .format(after_layer_hook))
        self.after_layer_hook: Callable[
            [AnalysisResult, ConceptDetection2DTrainTestHandle], Any] = \
            after_layer_hook
        """Callable that is called after each layer analysis with the
        analysis result and the used training handle as arguments.
        Can e.g. be used to clear dataset caches or store the result."""

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings dict to reproduce instance."""
        return dict(
            concept=self.concept,
            model=self.model,
            layer_infos=self.layer_infos,
            cross_val_runs=self.cross_val_runs,
            num_val_splits=self.num_val_splits,
            emb_reduction=self.emb_reduction,
            train_val_args=self.train_val_args,
            concept_model_args=self.concept_model_args,
            data_args=self.data_args,
            after_layer_hook=self.after_layer_hook,
        )

    def __repr__(self):
        setts = self.settings
        # handle dict attribute representation
        for k, val in setts.items():
            if isinstance(val, dict) and len(val) > 0:
                setts[k] = '{\n' + ',\n'.join(
                    ["\t{!s}:\t{!s}".format(sub_k, sub_v)
                     for sub_k, sub_v in val.items()]) + '\n}'
        return (str(self.__class__.__name__) + '(\n' +
                ',\n'.join(
                    ["{!s} =\t{!s}".format(k, v) for k, v in setts.items()])
                + '\n)')

    def best_embedding(self,
                       analysis_results: Dict[
                           str, Dict[int, Tuple[ConceptEmbedding, pd.Series]]]
                       = None) -> ConceptEmbedding:
        """Conduct an analysis and from results derive the best embedding.

        :param analysis_results: optionally the results of a previously run
            analysis; defaults to running a new analysis via :py:meth:`analysis`
        :return: the determined best embedding of all layers analysed
        """
        analysis_results: AnalysisResult = analysis_results or self.analysis()
        best_embs_stds_stats: Dict[
            str, Tuple[ConceptEmbedding, Tuple, pd.Series]] = {}
        for _, results_per_run in analysis_results.items():
            best_embs_stds_stats.update(
                self.embedding_reduction(results_per_run).results)
        best_layer_id = self.best_layer_from_stats(
            BestEmbeddingResult(best_embs_stds_stats))
        LOGGER.info("Concept %s final layer: %s", self.concept.name,
                    best_layer_id)
        best_embedding, _, _ = best_embs_stds_stats[best_layer_id]
        return best_embedding

    def analysis(self) -> AnalysisResult:
        """Conduct a concept embedding analysis.

        For each layer in :py:attr:`layer_infos`:

        - train :py:attr:`cross_val_runs` x :py:attr:`num_val_splits`
          concept models,
        - collect their evaluation results,
        - convert them to embeddings.

        :return: a analysis result object holding a dictionary of
            ``{layer_id: {run: (embedding,
            pandas.Series with {pre_: metric_val}}}``
        """
        results_per_layer: Dict[
            str, Dict[int, Tuple[Sequence[ConceptEmbedding], pd.Series]]] = {}
        for layer_id in self.layer_infos:
            results_per_run: AnalysisResult = self.analysis_for_layer(layer_id)
            results_per_layer.update(results_per_run.results)
        return AnalysisResult(results_per_layer)

    def analysis_for_layer(self, layer_id: str
                           ) -> AnalysisResult:
        """Get a concept embedding of the given concept in the given layer.
        A number of :py:attr:`cross_val_runs` cross validation runs is conducted
        with each :py:attr:`num_val_splits` non-intersecting splits for the
        validation data.
        In case :py:attr:`num_val_splits` is 1, just :py:attr:`cross_val_runs`
        a normal training runs are conducted.

        After the analysis is completed, :py:attr:`after_layer_hook` is called.

        :param layer_id: ID of the layer to find embedding in; key in
            :py:attr:`layer_infos`
        :return: an analysis result object holding only information on
            this layer
        """
        c_model = self.concept_model_for_layer(layer_id)
        c_handle: ConceptDetection2DTrainTestHandle = \
            self.concept_model_handle(c_model)
        if 'lr' in self.layer_infos[layer_id]:
            c_handle.optimizer.lr = self.layer_infos[layer_id]['lr']

        stats_per_run = {}
        for cross_val_run in range(self.cross_val_runs):
            c_handle.model.reset_parameters()
            callback_context: Dict[str, Any] = {
                **c_handle.callback_context,
                'log_prefix': f"{c_handle.callback_context.get('log_prefix', '')}/"
                              f"cv{cross_val_run}"
            } if self.cross_val_runs > 1 else c_handle.callback_context
            if self.num_val_splits > 1:
                states, _, _ = zip(*c_handle.cross_validate(
                    num_splits=self.num_val_splits,
                    run_info_templ=("{}, cv {}/{}, ".format(
                        layer_id, cross_val_run + 1, self.cross_val_runs) +
                                    "run {run}/{runs}"),
                    callback_context=dict(callback_context)))
            else:
                c_handle.train()
                states = [c_handle.detached_state_dict(c_model)]

            for val_split, state_dict in enumerate(states):
                curr_cb_context = {
                    **callback_context,
                    'run': (val_split if self.num_val_splits > 1 else None)}

                c_model.load_state_dict(state_dict)

                if c_handle.model.use_laplace:
                    c_handle.second_stage_train(
                        callback_context=dict(curr_cb_context))

                embeddings: List[ConceptEmbedding] = c_model.to_embedding()
                run = val_split + cross_val_run * self.num_val_splits
                metrics: pd.Series = c_handle.evaluate(
                    callback_context=dict(curr_cb_context))
                # storing & logging
                stats_per_run[run] = (embeddings, metrics)
                context = "Concept {}, layer {}, run {}".format(
                    self.concept.name, layer_id, run)
                LOGGER.info(
                    "%s:\n%s", context,
                    AnalysisResult.emb_info_to_string(embeddings, metrics))

        layer_result = AnalysisResult({layer_id: stats_per_run})
        if self.after_layer_hook is not None:
            self.after_layer_hook(layer_result, c_handle)
        return layer_result

    def concept_model_handle(self,
                             c_model: ConceptDetectionModel2D = None,
                             emb: Union[ConceptEmbedding,
                                        Sequence[ConceptEmbedding]] = None,
                             layer_id: str = None
                             ) -> ConceptDetection2DTrainTestHandle:
        """Train and eval handle for the given concept model.
        The concept model to handle can either be specified directly or is
        created from an embedding or from a given ``layer_id``.

        :param c_model: the concept model to provide a handle for
        :param emb: if ``c_model`` is not given, it is initialized using
            :py:meth:`concept_model_from_embedding` on ``emb``
        :param layer_id: if c_model and emb is not given, it is initialized
            using :py:meth:`concept_model_for_layer` on ``layer_id``
        :return: a handle for the specified or created concept model
        """
        if c_model is None:
            if emb is not None:
                c_model = self.concept_model_from_embedding(emb)
            elif layer_id is not None:
                c_model = self.concept_model_for_layer(layer_id)
            else:
                raise ValueError("Either c_model, emb, or layer_id must "
                                 "be given.")

        data = self.data_for_concept_model(c_model)
        callback_context = self.train_val_args.get('callback_context', {})
        callback_context['log_prefix'] = "{}/{model}/{layer}/{concept}".format(
            callback_context.get('log_prefix', ""),
            model=c_model.main_model.__class__.__name__,
            layer=c_model.layer_id, concept=c_model.concept_name
        )

        return ConceptDetection2DTrainTestHandle(
            concept_model=c_model, data=data, **{
                **self.train_val_args,
                'callback_context': callback_context,
            }
        )

    def concept_model_for_layer(self, layer_id):
        """Return a concept model for the given layer ID.

        :param layer_id: ID of the layer the concept model should be attached
            to; key in :py:attr:`layer_infos`
        :returns: concept model for :py:attr:`concept` attached to given
            layer in :py:attr:`model`
        """
        args = {
            'kernel_size': self.layer_infos[layer_id].get("kernel_size", None),
            'in_channels': self.layer_infos[layer_id].get("out_channels", None),
            **self.concept_model_args
        }
        c_model: ConceptDetectionModel2D = ConceptDetectionModel2D(
            concept=SegmentationConcept2D.new(self.concept),
            model=self.model, layer_id=layer_id,
            **args
        )
        self.layer_infos[layer_id]['out_channels'] = c_model.in_channels
        return c_model

    def concept_model_from_embedding(self,
                                     emb: Union[ConceptEmbedding,
                                                Sequence[ConceptEmbedding]]
                                     ) -> ConceptDetectionModel2D:
        """Get concept model from embedding for training and eval."""
        embs = [emb] if isinstance(emb, ConceptEmbedding) else emb
        kwargs = {**dict(main_model=embs[0].main_model or self.model,
                         concept=embs[0].concept or self.concept),
                  **self.concept_model_args}
        return ConceptDetectionModel2D.from_embedding(embs, **kwargs)

    def data_for_concept_model(self, c_model: ConceptDetectionModel2D = None,
                               layer_id: str = None) -> DataTriple:
        """Get the concept model data for this instance."""
        assert c_model is not None or layer_id is not None

        # Choose default device for transformations to be the train device:
        data_args = {"device": self.train_val_args.get("device", None),
                     **self.data_args}

        # Wrap the concept data:
        if c_model:
            data: DataTriple = data_for_concept_model(
                concept_model=c_model, **data_args)
        else:
            data: DataTriple = data_for_concept_model(
                main_model_stump=ModelStump(self.model, layer_id),
                concept=self.concept, **data_args)
        return data

    def evaluate_embedding(self, embedding: Union[ConceptEmbedding,
                                                  Sequence[ConceptEmbedding]],
                           log_prefix: str = None):
        """Evaluate the embedding on its concept test data."""
        # Evaluation:
        with torch.no_grad():
            eval_model = self.concept_model_from_embedding(embedding)
            c_handle = self.concept_model_handle(eval_model)
            callback_context = None
            if log_prefix is not None:
                callback_context = dict(
                    log_prefix=f"{c_handle.callback_context.get('log_prefix', '')}/"
                               f"{log_prefix}")
            stats: pd.Series = \
                c_handle.evaluate(callback_context=callback_context)

        return stats

    def embedding_reduction(
            self,
            results_per_run: AnalysisResult
    ) -> BestEmbeddingResult:
        """Aggregate the embeddings collected in ``results_per_run``
        to a best one.
        This is a wrapper with standard deviation and stats collection and
        logging around a call to :py:func:`emb_reduction`.

        :param results_per_run: analysis result object as returned by
            :py:meth:`analysis`
        :return: a best embedding result object holding one tuple entry of

            - an aggregated ("mean") embedding for the concept and the layer,
            - the standard deviation values of the normal vectors,
            - the stats for the chosen "mean" embedding
        """
        if len(results_per_run.results) == 0:
            raise ValueError("Empty results dict")
        if len(results_per_run.results) > 1:
            raise ValueError(
                ("Results handle stores information on more than one layer, "
                 "thus selection is ambiguous! Layer keys: {}").format(
                    list(results_per_run.results.keys())))

        layer_id: str = list(results_per_run.results.keys())[0]
        all_embeddings: List[ConceptEmbedding] = [
            e for embs, _ in results_per_run.results[layer_id].values()
            for e in embs]
        best_embedding = self.emb_reduction(all_embeddings)

        # Variance and stats collection:
        std_dev: Tuple[np.ndarray, float, float] = \
            ConceptEmbedding.std_deviation(all_embeddings)
        stats: pd.Series = self.evaluate_embedding(best_embedding,
                                                   log_prefix='reduced')

        # Some logging:
        LOGGER.info("Concept %s, layer %s:\n%s",
                    self.concept.name, layer_id,
                    AnalysisResult.emb_info_to_string(best_embedding, stats,
                                                      std_dev=std_dev))
        return BestEmbeddingResult({layer_id: (best_embedding,
                                               std_dev, stats)})

    @staticmethod
    def best_layer_from_stats(
            results_per_layer: BestEmbeddingResult) -> str:
        """From the embedding quality results per layer, select the best layer.
        For segmentation concepts, select by set IoU.

        :param results_per_layer: a best embedding result object
        :return: layer ID with best stats
        """
        # DataFrame with layer-wise metric results
        # (col: layer_id, idx: metric_name)
        test_set_iou_key = ConceptDetection2DTrainTestHandle.test_("set_iou")
        test_loss_key = ConceptDetection2DTrainTestHandle.test_(ConceptDetection2DTrainTestHandle.LOSS_KEY)
        stats = pd.DataFrame({
            l_id: info[-1] for l_id, info in results_per_layer.results.items()})
        if len(stats.columns) == 1:
            return stats.columns[0]

        if test_set_iou_key in stats.index:
            best_layer_id = stats.loc[test_set_iou_key].astype('float32').idxmax()
        elif test_loss_key in stats.index:
            best_layer_id = stats.loc[test_loss_key].astype('float32').idxmin()
        else:
            raise KeyError(
                ("KPI key {} not in stats keys {}; Wrong concept type used?"
                 " (currently only segmentation concepts allowed)"
                 ).format(test_set_iou_key, stats.index))

        return str(best_layer_id)

    def fill_train_data_infos(self) -> pd.DataFrame:
        """Collect layer-wise information about the corresponding concept
        model.
        The results are stored into :py:attr:`layer_infos` and returned
        as :py:class:`pandas.DataFrame`.
        """
        layer_infos = {}
        for layer_id in self.layer_infos:
            c_model = self.concept_model_for_layer(layer_id)
            c_handle = self.concept_model_handle(c_model)
            layer_infos[layer_id] = {
                'kernel_size': c_model.kernel_size,
                'prop_neg_px': datavis.neg_pixel_prop(c_handle.data.train)}
            self.layer_infos[layer_id].update(layer_infos[layer_id])
        return pd.DataFrame(layer_infos).transpose()

    def best_embedding_with_logging(
            self,
            concept_exp_root: str,
            logger: logging.Logger = None,
            file_logging_formatter: logging.Formatter = None,
            log_file: str = 'log.txt',
            img_fp_templ: Optional[str] = "{}.png",
            visualization_transform: Optional[Callable[[Any, Any], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> ConceptEmbedding:
        # pylint: disable=line-too-long
        """Conduct an analysis, collect mean and best embeddings,
        and save and log all results.

        .. rubric:: Saved results

        - the embedding of each layer and run as .pt file;
          for format see
          :py:meth:`hybrid_learning.concepts.models.embeddings.ConceptEmbedding.save`;
          load with
          :py:meth:`hybrid_learning.concepts.models.embeddings.ConceptEmbedding.load`
        - the aggregated (best) embedding for each layer (see above)
        - the final best embedding amongst all layers
          (chosen from above best embeddings; see above)
        - statistics of the runs for each layer incl. evaluation results and
          infos on final embedding obtained by each run;
          for format see
          :py:meth:`~hybrid_learning.concepts.analysis.results.AnalysisResult.save`;
          load with
          :py:meth:`~hybrid_learning.concepts.analysis.results.AnalysisResult.load`;
        - statistics for the aggregated (best) embeddings;
          for format see
          :py:meth:`~hybrid_learning.concepts.analysis.results.BestEmbeddingResult.save`;

        .. rubric:: Saved visualizations

        - visualization of the training data
        - visualization of the final best embedding on some test data samples
        - visualization of the best embedding and each embedding in its layer
          for comparison (the best embedding is a kind of mean of the embeddings
          from its layer)
        - visualization of the aggregated embeddings of each layer
          for comparison

        :param concept_exp_root: the root directory in which to save results
            for this part
        :param logger: the logger to use for file logging; defaults to the
            module level logger; for the analysis, the logging level is set
            to :py:const:`logging.INFO`
        :param file_logging_formatter: if given, the formatter for the file
            logging
        :param log_file: the path to the logfile to use relative to
            ``concept_exp_root``
        :param img_fp_templ: template for the path of image files relative to
            ``concept_exp_root``; must include one ``'{}'`` formatting variable
        :param visualization_transform: a transformation applied to the tuple of
            concept model output and ground truth mask before visualization as mask
        :return: the found best embedding for that part
        """
        # pylint: enable=line-too-long
        os.makedirs(concept_exp_root, exist_ok=True)
        save_imgs: bool = img_fp_templ is not None
        if save_imgs and ('{}' not in img_fp_templ
                          or img_fp_templ.count('{}') > 1):
            raise ValueError("Invalid img_fp_templ {}; ".format(img_fp_templ) +
                             "must contain exactly one occurrence of '{}'")
        save_as: Callable[[str], str] = lambda desc: os.path.join(
            concept_exp_root,
            img_fp_templ.format(desc))

        # region Logging setup
        if logger is None:
            logger = logging.getLogger(__name__)
        orig_logging_level: int = logger.level
        logger.setLevel(logging.INFO)
        part_log_file_handler = logging.FileHandler(
            os.path.join(concept_exp_root, log_file))
        part_log_file_handler.setLevel(logging.INFO)
        if file_logging_formatter is not None:
            part_log_file_handler.setFormatter(file_logging_formatter)
        logger.addHandler(part_log_file_handler)
        # endregion

        # Some logging friendly settings of pandas
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.expand_frame_repr', False):
            # region Settings info
            logger.info("Starting concept %s", self.concept.name)
            # Concept
            logger.info("Concept data:\n%s", self.concept.data.info)
            logger.info("Mean proportion of negative pixels orig data: %f",
                        datavis.neg_pixel_prop(self.concept.test_data))
            # Analysis settings
            logger.info("Analysis settings:\n%s", str(self))
            logger.info("Layer-wise training data properties:\n%s",
                        self.fill_train_data_infos())
            if save_imgs:
                datavis.visualize_mask_transforms(
                    titled_datasets={
                        layer_id: self.data_for_concept_model(
                            layer_id=layer_id).train
                        for layer_id in self.layer_infos},
                    save_as=save_as("vis_train_data_transforms"))
            # endregion

            # Analysis:
            analysis_results = self.analysis()
            analysis_results.save(concept_exp_root)
            logger.info("Embedding results per run:\n%s",
                        analysis_results.to_pandas().infer_objects().select_dtypes(np.number))

            # Best embedding selection:
            best_embs_results: BestEmbeddingResult = \
                BestEmbeddingResult({layer_id: self.embedding_reduction(
                    analysis_results.result_for(layer_id)).results[layer_id]
                                     for layer_id in
                                     analysis_results.results.keys()})
            best_embs_results.save(concept_exp_root)
            best_emb_infos = best_embs_results.to_pandas()
            logger.info("Best embeddings per layer:\n%s", best_emb_infos)

            # The very best embedding:
            # Save it twice to find it more easily and store in best_embs
            best_layer_id = self.best_layer_from_stats(best_embs_results)
            best_layer_embs: List[ConceptEmbedding] = [
                e for ens_emb, _ in
                analysis_results.results[best_layer_id].values()
                for e in ens_emb]
            best_embedding: ConceptEmbedding = \
                best_embs_results.results[best_layer_id][0]
            best_embedding.save(os.path.join(concept_exp_root, "best.pt"))
            logger.info("Best embedding:\n%s",
                        best_emb_infos.loc[best_layer_id])

            # pair-wise cosines with last row and column the best_embedding:
            pairwise_cos = vis.pairwise_cosines(
                embs=best_layer_embs + [best_embedding],
                keys=list(range(len(best_layer_embs))) + ['best_emb'])
            pairwise_cos.to_csv(
                os.path.join(concept_exp_root, 'pairwise_cosines.csv'))
            if self.num_val_splits == 1 == self.cross_val_runs:
                logger.info(
                    "Mean cosine dist of best_embedding to other embeddings:"
                    "\n%s", pairwise_cos.iloc[:-1, -1].mean())
                logger.info(
                    'Pair-wise cosine dist between normal vectors of runs in '
                    'best layer:\n%s',
                    pairwise_cos)

        # visualizations:
        if save_imgs:
            vis.visualize_concept_model(
                self.concept_model_handle(emb=best_embedding),
                transform=visualization_transform,
                save_as=save_as("vis_best_embedding"))
            vis.visualize_concept_models(
                {**{"best": self.concept_model_handle(emb=best_embedding)},
                 **{"emb {}".format(i): self.concept_model_handle(emb=e)
                    for i, e in enumerate(best_layer_embs)}},
                transform=visualization_transform,
                save_as=save_as("vis_best_layer_embeddings"))
            vis.visualize_concept_models(
                {layer_id: self.concept_model_handle(emb=e)
                 for layer_id, (e, _, _) in best_embs_results.results.items()},
                transform=visualization_transform,
                save_as=save_as("vis_best_embeddings"))

        # Close part specific logging
        logger.removeHandler(part_log_file_handler)
        logger.setLevel(orig_logging_level)

        return best_embedding
