"""Sacred experiment training settings as used for the original Net2Vec
paper."""
#  Copyright (c) 2022 Continental Automotive GmbH

# pylint: disable=no-name-in-module,import-error
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
# pylint: disable=unused-variable,unused-argument,unused-import

import os
from tqdm import tqdm
from hybrid_learning.datasets import transforms as trafos, caching
from hybrid_learning.concepts import kpis, models, analysis
from typing import Dict, Any, List
import torch
import numpy as np

from .config_ca import ex, concept_analysis, get_analyzer, get_concept, \
    prepare_logging, get_act_caches


# noinspection PyUnusedLocal
@ex.named_config
def net2vec_config():
    # pylint: disable=unused-variable
    """Configuration as used in original Net2Vec paper."""
    # The mean proportion of foreground pixels in the concept masks:
    mean_prop_background_px: float = 0.95

    # Any simple settings for the analysis train_val_args to override
    # (applied to each single training run)
    training_setts: Dict[str, Any] = dict(
        # The maximum number of epochs to run
        max_epochs=30,
        # Don't apply early stopping
        early_stopping_handle=models.EarlyStoppingHandle(0.00001),
        # The used batch size
        batch_size=64,
        # Loss function
        loss_fn=kpis.BalancedBCELoss(factor_pos_class=mean_prop_background_px),
        # The optimizer including learning rate and weight regularization
        optimizer=models.ResettableOptimizer(
            torch.optim.SGD,
            momentum=0.9,
            lr=0.0001),
    )

    # Any settings for the analysis to override (except for train_val_args);
    # For Net2Vec use only one run per concept and layer, no cross-validation
    # (we already have 30 epochs per concept model!)
    analysis_setts: Dict[str, Any] = dict(
        # The ensembling scheme to reduce a set of embeddings to a kind of mean
        emb_reduction=analysis.EmbeddingReduction.MEAN_NORMALIZED_DIST,
        # How many concept models to train for each layer before taking mean:
        # The number of cross-validation runs to conduct (with distinct val set)
        cross_val_runs=1,
        # The number of validation data splits per cross-validation run
        # (validation data proportion then is 1/num_val_splits)
        num_val_splits=1,
    )

    # The path to the numpy file holding the threshold vector;
    # if a "{}" is contained in the file name, it is replaced by the layer ID
    thresholds_file = os.path.join("threshs", "thresholds_{}.npy")


@ex.capture
def get_net2vec_thresholding(layer_infos, thresholds_file
                             ) -> trafos.TupleTransforms:
    """Get the thresholding transformation used by Net2Vec."""
    if thresholds_file and len(layer_infos) > 1:
        raise ValueError(
            ("The layer_infos config value contains more than one layer "
             "(layer_infos={}), and Net2Vec thresholding is enabled "
             "(thresholds_file={}). Currently, only Net2Vec thresholds for one "
             "layer may be specified. "
             "Thus disable thresholding by setting thresholds_file=None "
             "or pass only one layer to process using layer_infos=[layer_id]."
             ).format(layer_infos, thresholds_file)
        )

    layer_id = list(layer_infos)[0]
    thresholds_file = thresholds_file.format(layer_id) \
        if "{}" in thresholds_file else thresholds_file
    if not os.path.exists(thresholds_file):
        raise FileNotFoundError("Couldn't find thresholds_file {}"
                                .format(thresholds_file))

    # The thresholds in the shape (filters, h, w):
    threshs = np.load(thresholds_file).reshape((-1, 1, 1))
    # The thresholding trafo:
    binarizer = trafos.Binarize(
        torch.tensor(threshs),  # pylint: disable=not-callable
        val_high_class=None)
    return trafos.OnInput(binarizer)


# noinspection PyUnusedLocal
@ex.config_hook
def add_net2vec_thresholding(config: Dict[str, Any],
                             command_name, logger) -> Dict[str, Any]:
    # pylint: disable=unused-argument
    """Add the thresholding model output transformation used by Net2Vec."""
    if command_name != concept_analysis.__name__ or \
            "thresholds_file" not in config or \
            not config["thresholds_file"]:
        return {}
    if len(config["part_keys"]) > 1:
        logger.warning("More than one part_key was specified (part_keys was {})"
                       " -- make sure that the global mean_prop_background_px "
                       "setting used for loss balancing applies to all "
                       "concepts! Consider separate runs for each concept.")
    return dict(
        act_data_setts={'transforms': get_net2vec_thresholding(
            layer_infos=config["layer_infos"],
            thresholds_file=config["thresholds_file"]
        )}
    )


# noinspection PyIncorrectDocstring
@ex.command
def generate_thresholds(layer_infos, part_infos, person_size,
                        act_data_setts, img_mask_cache_root,
                        top_quantile=0.005,
                        thresholds_file=os.path.join("new_threshs",
                                                     "thresholds_{}.npy"),
                        reduce_memory_consumption=False,
                        overwrite=False):
    """Collect and save the thresholds to use for the Net2Vec thresholding
    transformation on the given layer(s).

    Results are stored to the location(s) specified by the
    ``thresholds_file`` (config) argument.
    If a ``{}`` is in the ``thresholds_file`` config argument,
    one file for each given layer is generated, otherwise just one layer
    may be specified.
    The cache for the training data activations is used for this, so make
    sure to fill it (see ``fill_caches`` command) before executing this!

    It is tried to keep impact on the memory small by only loading activation
    maps for one filter at once into memory. This, however, may take some more
    time than doing all at once.

    :param reduce_memory_consumption: whether to choose a more memory-friendly
        looping (i.e. filter-wise map loading); may take much longer
    :param overwrite: whether to overwrite existing files
    """
    # region Value checks
    if '{}' not in thresholds_file and len(layer_infos) > 1:
        raise ValueError(
            ("The layer_infos config value contains more than one layer (is {})"
             " but the thresholds_file config value does not allow for layer "
             "specific formatting (is {})").format(layer_infos, thresholds_file)
            + " because it does not contain a {}."
              " Refusing to overwrite thresholds file, aborting."
        )
    logger, log_format = prepare_logging()
    if act_data_setts["cache_root"] is None and \
            img_mask_cache_root is None:
        logger.warning("No cache root specified (act_data_setts.cache_root and "
                       "img_mask_cache_root are None). "
                       "Nothing to do, aborting.")
        return
    # endregion

    # Go through layer and concepts and get each dataset item once:
    for layer_id in layer_infos:
        layer_threshs_file: str = thresholds_file \
            if '{}' not in thresholds_file else thresholds_file.format(layer_id)
        if os.path.exists(layer_threshs_file) and not overwrite:
            raise FileExistsError("Thresholds file {} exists and overwrite is "
                                  "set to False.".format(layer_threshs_file))
        logger.info("Processing layer %s (saving location: %s)...",
                    layer_id, layer_threshs_file)
        os.makedirs(os.path.dirname(layer_threshs_file), exist_ok=True)

        # region Get training activation map cache
        parts_spec = part_infos[list(part_infos)[0]]
        concept = get_concept(**parts_spec, person_size=person_size,
                              layer_id=layer_id)
        analyzer = get_analyzer(  # pylint: disable=no-value-for-parameter
            concept=concept, layer_ids=[layer_id]
        )
        c_handle = analyzer.concept_model_handle(layer_id=layer_id)
        act_cache: caching.NPYCache = get_act_caches(c_handle)[0]
        # endregion

        # region Get total number of filters and check cache isn't empty
        num_filters = None
        for example_act_map_desc in act_cache.descriptors():
            num_filters = act_cache.load(example_act_map_desc).shape[0]
            break
        assert num_filters is not None, "Cache empty for {}!".format(layer_id)
        # endregion

        # region Actually collect and save threshold values
        descriptors: List[str] = list(act_cache.descriptors())
        thresholds = []
        with torch.no_grad():
            if reduce_memory_consumption:
                # Filter-wise processing of act maps
                for f_idx in range(num_filters):
                    all_filter_acts = []
                    for descriptor in tqdm(descriptors, leave=False,
                                           desc=("Filter {} act maps loaded"
                                                 ).format(f_idx)):
                        acts = act_cache.load(descriptor)
                        all_filter_acts.append(np.array(acts[..., f_idx, :, :]))
                        del acts  # free memory as early as possible
                    all_filter_acts_np = np.array(all_filter_acts)
                    thresholds.append(np.quantile(all_filter_acts_np,
                                                  1 - top_quantile))
            else:
                # Collect ALL act maps, then do filter-wise operation
                all_acts = []
                for descriptor in tqdm(descriptors, leave=False,
                                       desc="Act maps loaded"):
                    acts = np.array(act_cache.load(descriptor))
                    all_acts.append(acts)
                all_acts_np = np.array(all_acts)
                for f_idx in range(num_filters):
                    # shape of all_acts_np is (#act_maps, ..., #filters, h, w)
                    thresholds.append(np.quantile(all_acts_np[..., f_idx, :, :],
                                                  1 - top_quantile))

        thresholds_np = np.array(thresholds)
        np.save(layer_threshs_file, thresholds_np)
        logger.info("Thresholds for layer %s (total %d; saved to %s):\n%s",
                    layer_id, len(thresholds), layer_threshs_file,
                    str(thresholds))
        # endregion

    logger.info("DONE.")
