#  Copyright (c) 2022 Continental Automotive GmbH
"""Sacred default configuration and main function to conduct concept analysis.
To run as script, override the configurations from the ``to_be_overridden``
config scope:

- ``model_key``: the key of the model builder to use to obtain the model to analyse;
    the respective builder must be registered under this key using :py:func:`register_model_builder`;
- ``layer_infos``: a list with the layer IDs of the layers to analyse.
- ``part_infos``: a dict with part information
- ``get_data``: the (copyable, :py:class`DataGetter`) callable that will
    provide a tuple of train and test data when called with the
    person_size and the items of one ``parts_info`` entry as
    keyword arguments

This can e.g. be done by importing the experiment handle :py:data:`ex` from
this module and appending configuration/named configuration.
Then call ``ex.run_commandline()``.

Have a look at the configuration options for more details.

To add a file storage observer, add to as commandline argument
``-F BASEDIR`` or ``--file-storage=BASEDIR``.
"""

# pylint: disable=no-name-in-module
# pylint: disable=import-error,wrong-import-order,wrong-import-position
# pylint: disable=unused-variable,unused-argument,unused-import
# pylint: disable=too-many-locals

import abc
import logging
import os
import sys
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional, Callable, Sequence

import pandas as pd
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from torch.utils.data import Subset
from tqdm import tqdm

sys.path.insert(0, "")
# noinspection PyUnresolvedReferences
from hybrid_learning.concepts import analysis, models, kpis
from hybrid_learning.concepts.concepts import SegmentationConcept2D
from hybrid_learning import datasets
from hybrid_learning.datasets import transforms as trafos
from hybrid_learning.concepts.models import model_extension
from hybrid_learning.concepts.train_eval import callbacks as cb

ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.logger = logging.getLogger()

_MODEL_GETTER_MAP: Dict[str, Callable[..., torch.nn.Module]] = dict()
"""Model builder registry."""


def register_model_builder(model_key: str, model_getter: Callable[..., torch.nn.Module]):
    """Register a model builder to later be callable via get_model."""
    if not callable(model_getter):
        raise ValueError("model_getter ({}) must be callable, but was of type {}"
                         .format(model_getter, type(model_getter)))
    _MODEL_GETTER_MAP[model_key] = model_getter


def registered_model_keys() -> List[str]:
    """Return a list of valid model builder keys."""
    return list(_MODEL_GETTER_MAP.keys())


@ex.capture
def get_model(model_key: str, layer_infos: dict = None, check_layer_infos: bool = True) -> torch.nn.Module:
    """Return the result of a model builder previously registered via register_model."""
    if model_key not in registered_model_keys():
        raise KeyError(("model_key {} unknown -- was it registered (see register_model)? "
                        "Registered model keys: {}").format(model_key, registered_model_keys()))
    model: torch.nn.Module = _MODEL_GETTER_MAP[model_key]()

    # region value checks
    # Correct type?
    if not isinstance(model, torch.nn.Module):
        raise ValueError(("The model builder registered under key {} did not yield a torch.nn.Module but "
                          "an object of type {}:\n{}").format(type(model), model))
    # Are all specified model layers valid?
    if check_layer_infos:
        main_model_modules = list(dict(model.named_modules()).keys())
        if layer_infos is None:
            raise ValueError("layer_infos unset! Choose from the following layer keys:\n{}".format(main_model_modules))
        for layer_id in layer_infos:
            assert layer_id in main_model_modules, "Layer_id {} not in model".format(layer_id)
    # endregion

    return model


class DataGetter(abc.ABC):
    """Simple wrapper around a get_data function.
    To provide a data getter (needed for the main run of the experiment),
    implement the ``__call__`` function in a subclass and set the
    ``get_data`` config argument to an instance of this subclass.
    The ``__call__`` method is passed and must accept as arguments

    - ``person_size``
    - all keys from a ``parts_info`` entry.

    For further needed arguments, consider using the ``@ex.capture`` mechanism.
    """

    @staticmethod
    @abc.abstractmethod
    def __call__(person_size, **kwargs
                 ) -> Tuple[datasets.BaseDataset, datasets.BaseDataset]:
        """Get the train and test data."""
        raise NotImplementedError()

    def __repr__(self) -> str:
        return "{cl}()".format(cl=self.__class__.__name__)


# noinspection PyUnusedLocal
@ex.config
def to_be_overridden():
    # pylint: disable=unused-variable
    """Config placeholders that are to be overridden elsewhere."""
    # The key of the registered model builder with which to obtain the model to analyse
    model_key: str = None  # this should not be the final value!
    # Layer information: List or dict of the layer IDs to process
    layer_infos: list = None  # this should not be the final value!
    # Function to obtain data; should accept all keys of a part_infos value
    get_data: DataGetter = None
    # The part information; each entry must have one key 'label' which holds
    # a string value that describes the part (used for file names etc.)
    part_infos: dict = None
    # Which parts from part_infos to analyse
    part_keys: list = list(part_infos) \
        if part_infos is not None else None


# noinspection PyUnusedLocal
@ex.config
def default_configuration1():
    # pylint: disable=unused-variable
    """The default configuration part 1."""
    # Whether to show progress bars in general.
    # By default determines settings for train/eval progress bars and dataset progress bars.
    show_progress_bars: bool = True
    # Whether to resize the ground truth masks to fit the activation map size;
    # Used for the mask_size setting when initializing the dataset
    resize_gt_to_act_map_size: bool = False
    # A fixed size of the ground truth masks;
    # By default, the mask size is: the same as the image size, or,
    # if resize_gt_act_map_size is set to True, the size of the activation map.
    mask_size: tuple = None
    # Whether to clear the activation map caches after each layer
    clear_act_caches_after_layer: bool = True
    # Whether to clear the activation map caches after each concept of a layer
    clear_act_caches_after_concept: bool = True
    # The relative size of the concept compared to the image
    # respectively compared to the mean person height if this is given;
    # Used to determine the kernel size of the concept model (i.e. the
    # receptive field for one output pixel of the concept model).
    # Must be a tuple (rel_width, rel_height) with values in the interval [0,1];
    # Set to 0 to obtain constant kernel size 1x1
    default_rel_size: tuple = (0., 0.)
    # Some names for default ranges of person sizes (relative to image height)
    # each given as (min, max):
    default_person_sizes = dict(
        far=(0.2, 0.38),
        middle=(0.38, 0.71),
        close=(0.71, 1.33),
        very_close=(1.33, 2.5)
    )
    # The actual allowed range of a person's size
    # as (min, max) or set person_size_key to one of default_person_sizes
    person_size: tuple = None
    person_size_key: tuple = None
    # The input image size required by the model:
    img_size: tuple = (400, 400)

    # Path to the root directory for experiments.
    experiment_root: str = os.path.join("experiments")
    # The logging format in .format() style.
    logging_format: str = '---------- \t{asctime}\n{levelname}: {message}'

    # Any settings for the analysis to override (except for train_val_args)
    analysis_setts: dict = dict(
        # The ensembling scheme to reduce a set of embeddings to a kind of mean
        emb_reduction=analysis.EmbeddingReduction.FIRST,
        # How many concept models to train for each layer before taking mean:
        # The number of cross-validation runs to conduct (with distinct val set)
        cross_val_runs=1,
        # The number of validation data splits per cross-validation run
        # (validation data proportion then is 1/num_val_splits)
        num_val_splits=1,
    )

    # The settings used to initialize each concept model;
    # First bilinearly upsample, then apply sigmoid:
    concept_model_setts: dict = dict(
        apply_sigmoid=False,
        use_laplace=True,
        ensemble_count=1,
        use_bias=True)
    # The mean proportion of foreground pixels in the concept masks;
    # may be used e.g. for the loss function
    mean_prop_background_px: float = 0.999
    # Any simple settings for the analysis train_val_args to override
    # (applied to each single training run)
    training_setts: dict = dict(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        # The number of asynchronous workers to use; set to 0 to disable
        num_workers=1,
        # Slight early stopping
        early_stopping_handle=models.EarlyStoppingHandle(min_delta=0.001,
                                                         patience=3),
        # The maximum number of epochs to run
        max_epochs=7,
        # The used batch size
        batch_size=8,
        batch_size_val=64,
        batch_size_hessian=6,

        # Loss function
        # loss_fn=kpis.TverskyLoss(factor_false_positives=0.5, from_logit=True), #Tversky loss
        loss_fn=torch.nn.BCEWithLogitsLoss(),  # CE loss
        # loss_fn=kpis.BalancedBCELoss(factor_pos_class=mean_prop_background_px,from_logit=True), #BCE loss

        nll_fn=torch.nn.BCEWithLogitsLoss(),
        # Metrics
        metric_fns={'acc': kpis.Accuracy(),
                    'prec@050': kpis.Precision(),
                    'rec@050': kpis.Recall(),
                    'ECE': kpis.ECE(),
                    'MCE': kpis.ECE(aggregation="max", max_prob=False,
                                    class_conditional=True),
                    'MeanCE': kpis.ECE(aggregation="mean", max_prob=False,
                                       class_conditional=True),
                    'NLL': torch.nn.BCEWithLogitsLoss(),
                    # 'TACE001': kpis.ECE(
                    #     class_conditional=True, max_prob=False,
                    #     threshold_discard=0.01, adaptive=True),
                    'CC': kpis.CalibrationCurve(),
                    'PR': kpis.PrecisionRecallCurve(),
                    'SetIoUCurve': kpis.SetIoUThresholdCurve(),
                    'set_iou@050': kpis.SetIoU(),
                    'mean_iou@050': kpis.IoU(),
                    },

        # The optimizer including learning rate and weight regularization
        optimizer=models.ResettableOptimizer(
            torch.optim.Adam,
            lr=0.001,
            # The weight decay to apply (L2 regularization)
            weight_decay=0.),
        # Transformation applied to the tuples (model output, ground truth);
        # applied before calculating the loss;
        # applies to model output: bilin. up-sampling > sigmoid
        model_output_transform=(
            trafos.SameSize(resize_target=False, interpolation="bilinear")
            # + trafos.OnInput(trafos.Lambda(torch.nn.Sigmoid()))
        ),
        metric_input_transform=(
            trafos.OnInput(trafos.Lambda(torch.nn.Sigmoid()))
        ),
        show_progress_bars=show_progress_bars,
    )

    # The transformation applied to the concept model output before
    # visualization:
    visualization_transform: trafos.Transform = \
        trafos.OnInput(trafos.Lambda(torch.nn.Sigmoid()))

    # Whether to add metrics to a tensorboard:
    tensorboard_logdir: str = os.path.join(experiment_root, "tblogs")  # log dir for tensorboard callback
    csv_logdir: str = os.path.join(experiment_root, "metrics")  # log dir for metric logging
    sacred_logdir: str = os.path.join(experiment_root, "logs")  # log dir for sacred file observer


# noinspection PyUnusedLocal
@ex.config
def default_configuration2(img_size):
    # pylint: disable=unused-variable
    """The default configuration part 2."""
    # Any simple settings for the analysis data_args to override
    # (applied to each single training run)
    act_data_setts = dict(
        # Transformation applied to data tuples (input, target) before
        # using them in training
        transforms=None,
        # Whether to cache the datasets in-memory.
        # Best only use this if working only on CPU, not with CUDA.
        cache_in_memory=False,
        # Path for caching of all datasets (set to "" to disable caching)
        cache_root=os.path.join(
            "cache", "cache_{}x{}".format(*img_size))
    )
    # Cache root for transformed images and masks
    img_mask_cache_root = act_data_setts["cache_root"]


@ex.capture
def prepare_logging(logging_format: str = None
                    ) -> Tuple[logging.Logger, logging.Formatter]:
    """Set up and return a logging formatter and logger.
    Set pandas options to logging-friendly values."""

    # logging settings
    logger: logging.Logger = ex.logger
    log_stream_handler = logging.StreamHandler(stream=sys.stdout)
    log_stream_handler.setLevel(logger.getEffectiveLevel())
    log_format = logging.Formatter(logging_format, style='{')
    log_stream_handler.setFormatter(log_format)
    logger.addHandler(log_stream_handler)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)

    return logger, log_format


@ex.capture
def _get_act_map_size(model: torch.nn.Module, layer_id: str,
                      img_size: Tuple[int, int]):
    """Collect the output size of layer for input size ``img_size`` as
    ``(height, width)``."""
    return list(model_extension.output_size(model, input_size=[3, *img_size],
                                            layer_id=layer_id)[-2:])


@ex.capture
def get_concept(model: torch.nn.Module,
                get_data: Callable,
                label: str,
                rel_size: float,
                person_size: Tuple[float, float],
                img_size: Tuple[float, float] = None,
                layer_id: Optional[str] = None,
                resize_gt_to_act_map_size: bool = False,
                mask_size: Optional[Tuple[int, int]] = None,
                training_setts: Dict[str, Any] = None,
                **other_parts_specs
                ) -> Optional[SegmentationConcept2D]:
    """Load the datasets corresponding to a part, set caches, and return as
    concept together with a list of activation caches that are used.
    Mind that ``rel_size`` is understood as the relative width and height with
    respect to the mean person height, which is assumed to be the image
    height by default.
    This is transformed to the relative height wrt. the image width and height
    before setting it for the concept.
    """
    # Change rel_size from relative-to-person-height to relative-to-image-size
    rel_size = (rel_size, rel_size) if isinstance(rel_size, float) else rel_size
    if person_size is not None:
        p_mean_rel_height: float = (sum(person_size) / 2)
        rel_height = rel_size[0] * p_mean_rel_height
        rel_width = rel_size[1] * p_mean_rel_height * img_size[0] / img_size[1]
        rel_size = (rel_height, rel_width)

    # Collect the mask_size setting
    if resize_gt_to_act_map_size and mask_size is None:
        if layer_id is None:
            raise ValueError("resize_gt_to_act_map_size is True but no "
                             "layer_id was provided!")
        mask_size = _get_act_map_size(  # pylint: disable=no-value-for-parameter
            model=model, layer_id=layer_id
        )

    # Actual data retrieval
    train_d, test_d = get_data(
        label=label,
        person_size=person_size,
        mask_size=mask_size,
        device=(training_setts or {}).get("device", None),
        **other_parts_specs)
    # Quick exit if data has len 0:
    for data in (d for d in (train_d, test_d) if len(d) == 0):
        print("WARNING: Skipping, since data split {} has len 0!"
              .format(data.split.name))
        return None

    return SegmentationConcept2D(
        name=label, rel_size=rel_size,
        data=datasets.DataTriple(train_val=train_d, test=test_d)
    )


def get_act_caches(c_handle: models.ConceptSegmentation2DTrainTestHandle):
    """Clear the activation map cache of the activations dataset used in train
    handle.
    Assumes that the ``transforms_cache`` of train_val and test data is
    a CacheTuple of image and activations cache.
    """
    act_caches = []
    for data in (c_handle.data.train, c_handle.data.val, c_handle.data.test):
        if isinstance(data, Subset):
            # Note that this selects the cache of the full dataset,
            # even though just a subset is considered --- this is of no
            # interest here, as all subsets get selected (train, val, test).
            data = data.dataset
        assert isinstance(data, datasets.ActivationDatasetWrapper), \
            "Data {} no ActivationDatasetWrapper but {}".format(data,
                                                                type(data))
        cache: datasets.caching.CacheTuple = data.transforms_cache
        assert isinstance(cache, datasets.caching.CacheTuple), \
            "Activation data cache {} no CacheTuple but {}".format(cache,
                                                                   type(cache))
        act_caches.append(cache.caches[0])
    return act_caches


def get_cache_collect_hook(caches_to_clear: List[datasets.caching.Cache],
                           act_data_setts=None,
                           clear_act_caches_after_layer=False,
                           clear_act_caches_after_concept=False, ):
    """If cache clearing is requested, provide a hook that collects the
    act_data caches in caches_to_clear."""
    after_layer_hook = None
    if (clear_act_caches_after_layer or clear_act_caches_after_concept) and \
            (act_data_setts["cache_root"] or act_data_setts["cache_in_memory"]):
        after_layer_hook = (lambda _, handle:
                            caches_to_clear.extend(get_act_caches(handle)))
    return after_layer_hook


# noinspection PyUnusedLocal
@ex.config_hook
def custom_config_postproc(config: Dict[str, Any],
                           command_name, logger) -> Dict[str, Any]:
    # pylint: disable=unused-argument
    """In case the layer_infos are not set, infer them from the model."""
    config_addendum = {}

    # region Value checks
    if config["model_key"] is None:
        raise ValueError("Config parameter model_key not specified!"
                         "Make sure to register a model builder and specify its model_key.")

    for missing_key in (k for k in (config["part_keys"]) if
                        k not in config["part_infos"]):
        raise ValueError(
            ("Setting part_keys contains key {} missing from part_infos! "
             "Either add it to part_infos, or select one of: {}").format(
                missing_key, list(config["part_infos"].keys())))

    # person_size
    if config["person_size"] is None and config["person_size_key"] is not None:
        config_addendum["person_size"] = \
            config["default_person_sizes"][config["person_size_key"]]

    # Add tensorboard callback
    if config["tensorboard_logdir"]:
        config_addendum.setdefault("training_setts", config.get("training_setts", {}))
        config_addendum["training_setts"].setdefault("callbacks", []).append(
            cb.TensorboardLogger(log_dir=config["tensorboard_logdir"],
                                 log_sample_targets=True))

    if config["csv_logdir"]:
        config_addendum.setdefault("training_setts", config.get("training_setts", {}))
        config_addendum["training_setts"].setdefault("callbacks", []).append(
            cb.CsvLoggingCallback(log_dir=config["csv_logdir"]))

    # Add sacred file logger
    if config["sacred_logdir"]:
        ex.observers.append(FileStorageObserver(config["sacred_logdir"]))

    return config_addendum


@ex.capture
def get_analyzer(concept: SegmentationConcept2D, layer_ids: Sequence[str], model: torch.nn.Module,
                 analysis_setts: Dict[str, Any],
                 act_data_setts: Dict[str, Any],
                 training_setts: Dict[str, Any],
                 concept_model_setts: Dict[str, Any],
                 after_layer_hook: Callable = None,
                 ) -> analysis.ConceptAnalysis:
    """Get the analyzer from settings."""
    analyzer = analysis.ConceptAnalysis(
        concept=concept, model=model, layer_infos=layer_ids,
        after_layer_hook=after_layer_hook,
        **analysis_setts,
        train_val_args=training_setts,
        concept_model_args=concept_model_setts,
        data_args=act_data_setts,

    )
    return analyzer


@ex.command
def fill_caches(layer_infos, part_infos, part_keys, person_size,
                act_data_setts, img_mask_cache_root,
                no_act_data: bool = False):
    """Fill the activation map and other data transformation caches.
    The training and validation data is used for this."""
    model: torch.nn.Module = get_model()
    logger, log_format = prepare_logging()
    if act_data_setts["cache_root"] is None and \
            img_mask_cache_root is None:
        logger.warning("No cache root specified (act_data_setts.cache_root and "
                       "img_mask_cache_root are None). "
                       "Nothing to do, aborting.")
        return

    # Go through layer and concepts and get each dataset item once:
    for layer_id in layer_infos:
        for part_key in part_keys:
            parts_spec = part_infos[part_key]
            logger.info("Processing layer %s, part %s ...", layer_id, part_key)
            concept: Optional[SegmentationConcept2D] = \
                get_concept(model=model, **parts_spec, person_size=person_size,
                            layer_id=layer_id)
            if concept is None:
                logger.warning("Skipping empty concept %s", parts_spec['label'])
                continue
            if not no_act_data:
                analyzer: analysis.ConceptAnalysis = get_analyzer(  # pylint: disable=no-value-for-parameter
                    model=model, concept=concept, layer_ids=[layer_id]
                )
                data_triple: datasets.DataTriple = \
                    analyzer.data_for_concept_model(layer_id=layer_id)
            else:
                data_triple = concept.data
            for split, data in (("TRAIN,VAL", data_triple.train_val),
                                ("TEST", data_triple.test)):
                logger.info(("Iterating over items%s for part %s, layer %s, "
                             "split %s"),
                            '' if no_act_data else ' (act maps)', parts_spec["label"], layer_id, split)
                for i in tqdm(range(len(data)), desc="Items processed",
                              leave=False):
                    _ = data[i]
    logger.info("DONE.")


@ex.main
def concept_analysis(layer_infos, part_keys, part_infos, person_size,
                     clear_act_caches_after_concept,
                     experiment_root, visualization_transform):
    # pylint: disable=too-many-arguments
    """The main function executing the analysis."""
    # SETTINGS:
    # ---------
    # region Value init and check
    model: torch.nn.Module = get_model()
    logger, log_format = prepare_logging()

    exp_root_templ: str = os.path.join(
        experiment_root, "layers", "{layer_id}", "{model}_{time}".format(
            model=model.__class__.__name__.lower(),
            time=datetime.now().strftime("%Y-%m-%d_%H%M%S%f")))

    caches_to_clear = []
    after_layer_hook = get_cache_collect_hook(caches_to_clear)
    # endregion

    # CONDUCT CONCEPT ANALYSIS:
    # -------------------------
    logger.info("Starting concept analysis "
                "with experiment root folders: %s", exp_root_templ)

    for layer_id in layer_infos:
        exp_root = exp_root_templ.format(layer_id=layer_id)
        os.makedirs(exp_root)  # Throw error if exists
        for parts_spec in (part_infos[k] for k in part_keys):
            concept_exp_root = os.path.join(exp_root, parts_spec['label'])
            concept = get_concept(model=model, **parts_spec, person_size=person_size,
                                  layer_id=layer_id)
            if concept is None:
                continue

            if torch.cuda.is_available():
                logger.info(
                    f'Starting: GPU Memory {torch.cuda.memory_allocated() / 1024 ** 2}/'
                    f'{torch.cuda.memory_reserved() / 1024 ** 2} allocated/reserved')
            analyzer: analysis.ConceptAnalysis = get_analyzer(  # pylint: disable=no-value-for-parameter
                model=model, concept=concept, layer_ids=[layer_id],
                after_layer_hook=after_layer_hook
            )
            analyzer.best_embedding_with_logging(
                concept_exp_root=concept_exp_root,
                logger=logger, file_logging_formatter=log_format,
                visualization_transform=visualization_transform,
            )
            del analyzer
            if torch.cuda.is_available():
                logger.info(
                    f'Finished: GPU Memory {torch.cuda.memory_allocated() / 1024 ** 2}/'
                    f'{torch.cuda.memory_reserved() / 1024 ** 2} allocated/reserved')
                torch.cuda.empty_cache()
                logger.info(
                    f'Cleared: GPU Memory {torch.cuda.memory_allocated() / 1024 ** 2}/'
                    f'{torch.cuda.memory_reserved() / 1024 ** 2} allocated/reserved')
            if clear_act_caches_after_concept:
                for cache in caches_to_clear:
                    logger.info("Clearing cache %s", str(cache))
                    cache.clear()
                caches_to_clear.clear()
        for cache in caches_to_clear:
            logger.info("Clearing cache %s", str(cache))
            cache.clear()
        caches_to_clear.clear()
    logger.info("DONE.")


@ex.command
def reevaluate(layer_infos, part_keys, part_infos, person_size,
               experiment_root, reeval_desc):
    # pylint: disable=too-many-arguments,cell-var-from-loop
    """Re-evaluate a concept analysis experiment.
    This command will load all experiment results (concept embeddings) for
    the provided layers and concepts and evaluate them on the specified test
    set."""
    from hybrid_learning.experimentation import ca_exp_eval as exp_eval

    # SETTINGS:
    # ---------
    # region Value init and check
    model: torch.nn.Module = get_model()
    logger, log_format = prepare_logging()

    caches_to_clear = []
    after_layer_hook = get_cache_collect_hook(caches_to_clear)
    # endregion

    # CONDUCT CONCEPT ANALYSIS:
    # -------------------------
    logger.info("Starting reevaluating concept analysis results "
                "with experiment root folder: %s", experiment_root)
    # Layers
    layer_ids = exp_eval.get_layers(experiment_root,
                                    model_layers=list(layer_infos))
    if len(layer_ids) == 0:
        logger.warning("No results for any of the following layers found: %s",
                       str(list(layer_infos)))
    # Concepts
    part_labels = exp_eval.get_concepts(experiment_root, layers=layer_ids)
    parts_specs = [parts_spec for k, parts_spec in part_infos.items()
                   if parts_spec['label'] in part_labels and k in part_keys]
    if len(parts_specs) == 0:
        logger.warning("No results for any of the following concepts found: %s"
                       "\navailable concepts: %s",
                       str([p['label'] for p in part_infos.items()]),
                       str(part_labels))
    # CSV files
    new_results_file: str = os.path.join(
        experiment_root, f"{model.__class__.__name__.lower()}_test"
                         f"_{reeval_desc}.csv")
    if os.path.exists(new_results_file):
        raise FileExistsError("Results file {} exists, refusing to overwrite."
                              .format(new_results_file))
    new_best_results_file: str = os.path.join(
        experiment_root, f"{model.__class__.__name__.lower()}_test"
                         f"_{reeval_desc}-best.csv")
    if os.path.exists(new_best_results_file):
        raise FileExistsError("Results file {} exists, refusing to overwrite."
                              .format(new_results_file))

    new_results: List[Dict[str, Any]] = []
    new_best_results: List[Dict[str, Any]] = []
    for layer_id in layer_ids:
        for parts_spec in parts_specs:
            concept = get_concept(model=model, **parts_spec, person_size=person_size,
                                  layer_id=layer_id)
            if concept is None:
                continue
            analyzer = get_analyzer(  # pylint: disable=no-value-for-parameter
                model=model, concept=concept, layer_ids=[layer_id],
                after_layer_hook=after_layer_hook
            )
            analysis_root: str = exp_eval.analysis_root(
                layer_id=layer_id, concept_name=parts_spec['label'],
                root=experiment_root)

            def reeval(embedding, run, results_collection, filepath):
                """Re-evaluate embedding, log and save."""
                logger.info(
                    "Re-evaluating layer %s, concept %s, run %s (%d samples)",
                    layer_id, parts_spec['label'], run, len(concept.data.test))
                results_collection.append(dict(
                    layer=layer_id, concept=parts_spec['label'], run=run,
                    **analyzer.evaluate_embedding(
                        embedding=embedding,
                        log_prefix=f"{reeval_desc}_evalrun{run}")))
                pd.DataFrame(results_collection).to_csv(filepath)

            # Re-run evaluations on test set and save results:
            analysis_results = analysis.AnalysisResult.load(analysis_root)
            for i, (emb, _) in analysis_results.results[layer_id].items():
                reeval(emb, run=i, results_collection=new_results,
                       filepath=new_results_file)

            # Re-run evaluations on test set for best embeddings and save:
            best_emb_results = analysis.BestEmbeddingResult.load(analysis_root)
            best_emb, _, _ = best_emb_results.results[layer_id]
            reeval(best_emb, run="reduced", results_collection=new_best_results,
                   filepath=new_best_results_file)

        for cache in caches_to_clear:
            logger.info("Clearing cache %s", str(cache))
            cache.clear()
        caches_to_clear.clear()
    logger.info("DONE.")


@ex.command
def print_mean_prop_background_px(part_infos, part_keys, person_size,
                                  mask_size, resize_gt_to_act_map_size,
                                  show_progress_bars=True):
    """Collect the mean proportion of background pixels for each concept.
    The value can be used to initialize the ``mean_prop_background_px``
    config value, which is by default used for balancing in the loss.
    Will save the result to ``mean_prop_background_px.csv``.

    .. warning::
        This assumes that the masks have the same size as the images
        and are not generated / rescaled according to the activation map
        size! For layer-specific mask sizes, call this for each layer
        separately with the corresponding mask_size option.
    """
    model: torch.nn.Module = get_model()
    if resize_gt_to_act_map_size and mask_size is None:
        raise ValueError("Option resize_gt_to_act_map_size is set to True,"
                         "but mask_size is unset. "
                         "This command cannot handle layer dependent mask "
                         "sizes. So call it for each layer separately with "
                         "the desired mask_size config value set manually.")
    logger, _ = prepare_logging()
    all_props = {}
    for part_key in part_keys:
        parts_spec = part_infos[part_key]
        logger.info("Collecting info for part_key %s ...", part_key)
        concept = get_concept(model=model, **parts_spec, person_size=person_size)
        if concept is None:
            continue
        mean_prop_background_px: float = \
            datasets.neg_pixel_prop(concept.train_val_data,
                                    show_progress_bar=show_progress_bars,
                                    max_num_samples=None)
        logger.info("Mean proportion of negative pixels train_val data: %f",
                    mean_prop_background_px)
        all_props[part_key] = mean_prop_background_px
    all_props_pd: pd.Series = pd.Series(all_props)
    print("Mean proportion of negative (background) pixels in the train/val "
          "data per concept:")
    print(all_props_pd.to_string())
    all_props_pd.to_csv("mean_prop_background_px.csv", header=False)
