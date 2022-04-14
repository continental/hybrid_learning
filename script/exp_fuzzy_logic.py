"""Experiment evaluating the occlusion robustness of Mask R-CNN.
Please call from project root.
Both predicate (e.g. concept model) results and the results of
the given logical formula are stored for each sample as dict,
pickled using torch.save.
Keys used in the dict:

- The model output has the key ``'pedestrian'``.
- The formula output has the key ``'final_'``.
- Each concept model output has the concept name as key (masks unscaled).

To start an experiment, provide

- concept_model_root: the previous experiment root under which to
  find the trained and stored concept models
- all_concepts_to_layers: mapping from concept names to the layer ID
  from which to take the trained concept model
  (or just the layer ID, if it is the same for all concepts).
  All concepts needed for predicates in the formula must be mapped.
- parser_key: The fuzzy logic parser to use. See :py:data:`PARSER`
  for available ones.
- formula_spec: String Boolean logical formula on atoms representing
  concept and main model outputs.
  For the syntax, esp. operator precedence, see the chosen parser.
  Atoms must be one of

  + concept names
  + the value of ``pedestrian_key``
  + the value of ``gt_pedestrian_key``.

- metrics: The AggregatingKpi metrics to aggregate during the run, e.g. the final quantifier;
  their final results are stored in the sacred logs and the metrics_logdir

If results_cache_dir is set, result masks are pickled to *.pt files
in the following folder structure

.. parsed-literal::

    results_cache_dir
     +-concept_name
     |  +-layer_id
     +-pedestrian_key
     |  +-fuzzy_logic_key
     +-printable formula_spec
        +-fuzzy_logic_key

For other settings see the config section.
"""
#  Copyright (c) 2022 Continental Automotive GmbH

import logging
import os
from pprint import pprint
import sys
from datetime import datetime
from typing import Callable, Dict, Any, Tuple, List, Iterable, Union, Set, Optional

import sacred.run
import torch
import torch.nn.functional
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
import hybrid_learning.experimentation.fuzzy_exp.fuzzy_exp_helpers as calc_helpers
from hybrid_learning import datasets
from hybrid_learning.concepts import kpis
from hybrid_learning.datasets import transforms as trafos, caching
from hybrid_learning import fuzzy_logic
from hybrid_learning.experimentation.model_registry.fuzzy_exp_models import \
    register_all_model_builders, get_efficientdet_config

sys.setrecursionlimit(100000000)

# ===================
# CONFIG
# ===================

ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds
from sacred.observers import FileStorageObserver

ex.logger = logging.getLogger()


# Register all model builders
register_all_model_builders()

# noinspection PyUnusedLocal
@ex.config
def exp_config():
    """Default configuration."""
    # Mandatory config:
    # Model key; set using named config (and don't forget to register the model!)
    # Also used to create key for the trafo model applied to the main model output
    # to obtain the segmentation masks (as f'{model_key}_trafo')
    model_key: str = None
    assert model_key is not None, "model_key must be set via named config (together with model registration)!"
    # Whether to use the boxes or the segmentations in the ground truth annotations
    use_boxes: bool = True
    # Root where to find previous experiment results
    concept_model_root: str = None
    # Mapping {concept_name: layer_id} or a single layer_id string;
    # This is verified and post-processed to the dict concept_to_layer
    # to exactly contain the needed concepts.
    all_concepts_to_layers: dict = None
    # Optional short pretty names for the concepts to use in formulas and cache dir names:
    concept_pretty_names: dict = {
        'LEFT_ANKLE-RIGHT_ANKLE': 'ankle',
        'LEFT_ARM-RIGHT_ARM': 'arm',
        'LEFT_EYE-RIGHT_EYE': 'eye',
        'LEFT_LEG-RIGHT_LEG': 'leg',
        'LEFT_WRIST-RIGHT_WRIST': 'wrist',
    }
    # Add additional modifiers to concept models, e.g. use_laplace=False
    concept_model_args: dict = dict(apply_sigmoid=True, use_laplace=False)  # default: normalize, don't use laplace

    # The logical OR to use to reduce instance segmentations to semantic segmentation;
    # values may be keys of MaskRCNNToSegMask.LOGICAL_OR
    fuzzy_logic_key: str = 'lukasiewicz'
    # The key used in the formula to name the model output segmentation mask
    pedestrian_key: str = 'pedestrian'
    gt_pedestrian_key: str = f'gt_{pedestrian_key}'
    # The settings for any custom predicates as {PREDICATE_SYMB: INIT_ARGS_DICT};
    # Used to determine formula cache directory names;
    # General settings that do not describe predicate init args should be stored in a dict under the key '_'
    predicate_setts = dict(
        IsPartofA=dict(
            kernel_size=25,  # little penalty for up to 4 pixel shift; choose loader_setts.batch_size sufficiently low!!
            thresh=0.1,
        ),
        AllNeighbors=dict(
            kernel_size=33,  # set square neighborhood window width to 16=maximum scale difference between masks
        ),
        BoolTorchOrNumpyOperation=dict(
            bool_thresh=0.5,  # the threshold for binarizing before Boolean operations
        ),
        **{'_': dict(  # any general settings not specifying init args to a logical ops
            same_size_mode='up_bilinear',  # resizing mode (see mode argument of SameSizeTensorValues class)}
            reduce_pred_masks=True,  # Whether to reduce the predicted annotation masks to a single mask using OR
            reduce_gt_masks=True,  # Whether to reduce the ground truth annotation masks to a single mask using OR
        )}
    )
    # The logical formula to evaluate for each sample to parse
    # (according to fuzzy_logic_key) and evaluate;
    # For the syntax have a look at the documentation of the Merge transformation
    formula_spec: str = "~LEFT_EYE-RIGHT_EYE||pedestrian"
    # The logical formula that encodes the ground truth values expected for formula_spec;
    # the result is then fed to the metrics; set to None to ignore
    gt_formula_spec: str = f"(~{pedestrian_key}&&{gt_pedestrian_key})>=ped_thresh" \
        if predicate_setts.get('_', {}).get('reduce_pred_masks', True) \
            or (not predicate_setts.get('_', {}).get('reduce_pred_masks', True)
                and not predicate_setts.get('_', {}).get('reduce_gt_masks', True) )\
        else f"(~(MasksOR({pedestrian_key}))&&{gt_pedestrian_key})>=ped_thresh"
    # Any constant values for formula evaluation;
    # Used to determine formula cache directory names;
    # Should only be key-number-pairs of keys (potentially) used in the formula string,
    # settings not occuring in the formula string should be set in predicate_setts['_']
    # (for proper cache dir naming)
    constants: dict = dict(
        ped_thresh=0.5,  # threshold above which the pedestrian mask counts as positive
        mon_thresh=0.5,  # threshold above which the monitor mask counts as "alarm"
        gt_mon_thresh=0.5,
        # img_mon_thresh=0.5,  # threshold above which the image-level monitor mask counts as "alarm"
        # gt_img_mon_thresh=0.5,
        mask_thresh=0.005,  # threshold for cutting low values of masks (if applied))
        bool_thresh=0.5,  # threshold for all predicate masks for default if_boolean_binarize_preds_at
    )

    # AUTOMATED FORMULA MODIFICATIONS (applied top to bottom)
    if_boolean_binarize_preds_at: str = 'bool_thresh'  # all predicate P mentions except for "{gt_pedestrian}" become "(P>={if_boolean_binarize_preds_at})"
    to_monitor: bool = True  # F becomes "~({F})"
    to_image_level: str = None  # e.g. 'AnyPixel(AllNeighbors({formula_spec}))'; may be tuple for (pred, gt)
    binarize_at: str = 'mon_thresh'  # F becomes "({F})>={binarize_at}", GT becomes "({GT})>=gt_{binarize_at}"

    dataset_root: str = os.path.join('dataset', 'coco')  # Root of the COCO dataset
    split: str = "TEST"  # Dataset split to use (must be key of enum DatasetSplit)
    img_size: list = [400, 400]  # Image size to which to pad and resize
    # Some names for default ranges of person sizes (relative to image height)
    # each given as (min, max);
    # select restriction to one size by setting predicate_setts['_']['person_size_key']
    default_person_sizes = dict(
        far=(0.2, 0.38),
        middle=(0.38, 0.71),
        close=(0.71, 1.33),
        very_close=(1.33, 2.5)
    )

    # Settings for data loader generation (see hybrid_learning.concepts.train_eval.train_eval_funs.loader)
    show_progress_bars: bool = True  # Show progress bar
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Device for inference
    loader_setts: dict = dict(  # Data loader settings
        batch_size=(1 if (any('IsPartOfA' in spec for spec in (formula_spec, gt_formula_spec))
                          or not predicate_setts.get('_', {}).get('reduce_pred_masks', True)
                          or not predicate_setts.get('_', {}).get('reduce_gt_masks', True))
                    else 16),
        shuffle=False,
        num_workers=1,  # The number of asynchronous workers to use; set to 0 to disable
    )
    # The AggregatingKpi metrics that are updated for each batch and
    # the final value of which is saved;
    # Should accept a tuple of (seg_mask, None) for update call
    metrics = {
        "all_mean": kpis.Mean(),
        "all_mean_gt": trafos.ReduceTuple(trafo=calc_helpers.SwapTupleArgs(), reduction=kpis.Mean()),
        "all_min": kpis.Minimum(),
        "all_binary_mean@050": trafos.ReduceTuple(trafo=trafos.OnInput(trafos.Binarize(threshold=0.5)),
                                                  reduction=kpis.Mean()),
        "exists_max": kpis.Maximum(),
        "histogram@1e3bins": kpis.Histogram(n_bins=1000),
        "std_dev": kpis.StandardDev(),
        **({"accuracy": kpis.Accuracy(),
            "f1score": kpis.F1Score(),
            "precision": kpis.Precision(),  # = 1 - false discovery rate
            "negpredictiveval": kpis.NegativePredictiveValue(),  # = 1 - false omission rate
            "recall": kpis.Recall(),  # = 1 - false silence rate
            "true_negative_rate": kpis.Specificity(),  # = 1 - false alarm rate
            } if gt_formula_spec is not None else {}),
    }

    # Logging settings (disable any of the logging types by setting to None or "")
    experiment_root: str = os.path.join("experiments", "occl_robustness", model_key, split)
    results_cache_dir: str = os.path.join("cache", "fuzzy_logic_masks", model_key, split)  # storage for .pt files
    metrics_logdir: str = None  # overwrite default metrics_logdir
    sacred_logdir: str = None  # set to False to disable logging into default sacred_logdir, or overwrite
    disable_formula_caching: bool = False  # Whether to disable caching of the formula trafos
    formulas_to_cache: list = []  # any additional intermediate outputs to cache


# =========================
# MODEL NAMED CONFIGS
# =========================

@ex.named_config
def maskrcnn():
    """Mask R-CNN model config."""
    model_key: str = 'maskrcnn'


@ex.named_config
def maskrcnn_box():
    """Mask R-CNN model config."""
    model_key: str = 'maskrcnn_box'
    img_size: list = [400, 400]


# noinspection PyUnusedLocal
@ex.named_config
def tf_efficientdet_d1():
    """Configuration for pretrained EfficientDet model variants
    from https://github.com/rwightman/efficientdet-pytorch.
    Install the effdet package for this:

    .. code:: shell

        # EfficientDet pytorch implementation
        pip install -e "git+https://github.com/rwightman/efficientdet-pytorch.git@75e16c2f#egg=effdet"
    """
    model_key = 'tf_efficientdet_d1'
    img_size: list = list(get_efficientdet_config()['image_size'])  # must be divisible by 2**7
    use_boxes: bool = True  # EfficientDet D1 only produces boxes


# =========================
# HELPER FUNCTIONS
# =========================

get_data = ex.capture(calc_helpers.get_data)


@ex.capture
def get_data_loader(loader_setts: Dict[str, Any], device: Union[str, torch.device],
                    show_progress_bars: bool, data: datasets.BaseDataset = None,
                    ) -> Iterable[Tuple[List[str], Tuple[torch.Tensor, torch.Tensor]]]:
    """Get the data loader for this experiment.
    The data loader returns tuples of the form ``(descriptors_list, image_batch)``.
    It is wrapped into a progressbar printer if ``show_progress_bars`` is true."""
    return calc_helpers.get_data_loader_for(data if data is not None else get_data(),
                               loader_setts, device, show_progress_bars)


mask_cache = ex.capture(calc_helpers.mask_cache)


@ex.capture
def save_metrics(metric_vals: Dict[str, float],
                 metrics_logdir: str, _run: sacred.run.Run):
    """Save dict ``{metric_name: val}`` to ``metrics_logdir/metrics.csv`` and within sacred observer."""
    calc_helpers.save_metrics_to(metric_vals=metric_vals, metrics_logdir=metrics_logdir, log_scalar=_run.log_scalar)


# ============================
# CONFIG POST-PROCESSING
# ============================


def _replace_all_with(orig: str, to_replace: List[str], replace_format_str: str):
    if not orig:
        return orig
    for key in to_replace:
        orig = orig.replace(key, replace_format_str.format(key))
    return orig

# noinspection PyUnusedLocal
@ex.config_hook
def config_postproc(config: Dict[str, Any],
                    command_name, logger) -> Dict[str, Any]:
    """Do some validation, add sacred file logger, parse the formula_spec, and
    post-process all_concepts_to_layers."""
    config_addendum: Dict[str, Any] = {}

    # Logdirs
    default_root = calc_helpers.default_logdir({**config, **config_addendum})
    # validate metrics_logdir
    config_addendum["metrics_logdir"] = config["metrics_logdir"] or os.path.join(
        default_root, "metrics", datetime.now().strftime("%Y-%m-%d_%H%M%S%f"))
    os.makedirs(config_addendum["metrics_logdir"])
    # add sacred file logging
    if config["sacred_logdir"] is not False:
        config_addendum["sacred_logdir"] = config["sacred_logdir"] or os.path.join(default_root, "logs")
        ex.observers.append(FileStorageObserver(config_addendum["sacred_logdir"]))

    # post-process formula_spec
    to_formula_obj: Callable[[str], fuzzy_logic.Merge] = lambda f: f if f is None else \
        calc_helpers.get_formula(f, config["fuzzy_logic_key"], config["predicate_setts"], warn_about_unused_setts=True)
    formula_spec = config["formula_spec"]
    gt_formula_spec = config["gt_formula_spec"]
    formulas_to_cache: List[str] = config["formulas_to_cache"]
    initial_parsed_formula: fuzzy_logic.Merge = to_formula_obj(formula_spec)
    initial_parsed_gt_formula: fuzzy_logic.Merge = to_formula_obj(gt_formula_spec)
    # binarize predicate variable names
    if config["if_boolean_binarize_preds_at"] and config["fuzzy_logic_key"].lower() == 'boolean':
        initial_parsed_formula = calc_helpers.to_bool_predicates(config, initial_parsed_formula)
        initial_parsed_gt_formula = calc_helpers.to_bool_predicates(config, initial_parsed_gt_formula)
    # normalize formula
    formula_spec = str(initial_parsed_formula)
    gt_formula_spec = str(initial_parsed_gt_formula)
    # to monitor
    if config["to_monitor"]:  # only for formula_spec
        formula_spec = f"~({formula_spec})"
    formulas_to_cache.append(formula_spec)
    if gt_formula_spec is not None:
        formulas_to_cache.append(gt_formula_spec)
    # to image level
    if config["to_image_level"]:
        to_image_level: Union[str, Tuple[str, str]] = config["to_image_level"]
        to_image_level, gt_to_image_level = (to_image_level, to_image_level) \
            if isinstance(to_image_level, str) else to_image_level
        formula_spec = to_image_level.format(formula_spec=formula_spec)
        gt_formula_spec = gt_to_image_level.format(formula_spec=gt_formula_spec) \
            if gt_formula_spec else None
    formulas_to_cache.append(formula_spec)
    if gt_formula_spec is not None:
        formulas_to_cache.append(gt_formula_spec)
    # binarize
    if config["binarize_at"]:
        formula_spec = f"({formula_spec})>={config['binarize_at']}"
        gt_formula_spec = f"({gt_formula_spec})>=gt_{config['binarize_at']}" \
            if gt_formula_spec else None
    # add modification results
    config_addendum["formula_spec"] = formula_spec
    config_addendum["gt_formula_spec"] = gt_formula_spec
    config_addendum["formulas_to_cache"] = formulas_to_cache
    sys.setrecursionlimit(1000000)  # LARGE formulas need a lot of recursion it seems ...
    formula: fuzzy_logic.Merge = to_formula_obj(formula_spec)
    config_addendum["formula"] = formula
    gt_formula: fuzzy_logic.Merge = to_formula_obj(gt_formula_spec)
    config_addendum["gt_formula"] = gt_formula

    # auto-fill concept_to_layer
    assert len(set(config["concept_pretty_names"])) == len(config["concept_pretty_names"]), \
        "Found duplicate concept pretty names: {}".format(config["concept_pretty_names"])
    concept_keys = [*formula.all_in_keys, *(gt_formula.all_in_keys if gt_formula is not None else [])]
    from_pretty_names = dict(zip(config["concept_pretty_names"].values(), config["concept_pretty_names"].keys()))
    needed_concepts: Set[str] = {from_pretty_names.get(k, k) for k in concept_keys
                                 if k not in [config["pedestrian_key"], config["gt_pedestrian_key"],
                                              *config["constants"].keys()]}
    concept_to_layer: Union[str, Dict[str, str]] = config["all_concepts_to_layers"]
    concept_to_layer: Dict[str, str] = {c: concept_to_layer for c in needed_concepts} \
        if isinstance(concept_to_layer, str) else dict(concept_to_layer)
    # verify concept_to_layer
    assert needed_concepts.issubset(set(concept_to_layer.keys())), \
        "Missing layer info for concepts {} from formula {}".format(
            needed_concepts.difference(set(concept_to_layer.keys())), config["formula_spec"])
    # prune concept_to_layer
    config_addendum["concept_to_layer"] = {c: l for c, l in concept_to_layer.items()
                                           if c in needed_concepts}
    return config_addendum


# ==================
# COMMAND
# ==================


@ex.capture
def print_info(formula_trafos: Dict[str, fuzzy_logic.Merge],
               formula_caches: Optional[Dict[str, caching.Cache]],
               metrics_logdir: str, sacred_logdir: str):
    """Print derived settings info."""
    print("Post-processed formula trafos:", '{')
    for key, trafo in formula_trafos.items():
        print(f' "{key}":')
        print(trafo.to_repr(indent_level=1, indent_first_child=True) if isinstance(trafo, fuzzy_logic.Merge) else repr(trafo))
    print('}')
    print("Formula caches:")
    pprint(formula_caches)
    print("Metrics logging dir:", metrics_logdir)
    print("Sacred logging dir:", sacred_logdir)


@ex.main
def occlusion_robustness(metrics,
                         gt_pedestrian_key,
                         formula, gt_formula,
                         # cache_gt_masks=False
                         _config
                         ):
    """Run evaluation on test and train/val sets for Mask R-CNN semantic segmentation."""
    
    formula: fuzzy_logic.Merge
    gt_formula: fuzzy_logic.Merge
    formula_str, gt_formula_str = str(formula), str(gt_formula) if gt_formula else None
    
    results_calc: calc_helpers.FormulaEvaluator = calc_helpers.FormulaEvaluator(
        conf=_config,
        formulas={formula_str: formula, gt_formula_str: gt_formula})
    # Derived settings print
    print_info(formula_trafos=dict(**results_calc.formulas, same_size=results_calc._same_size),
               formula_caches=results_calc.formula_caches)
    
    # # Ground truth cache (make sure that subsetting of annotations is respected in the naming)
    # gt_cache = mask_cache(
    #     subdir=gt_pedestrian_key + (predicate_setts.get('_', {}).get('person_size_key', None) or '')) \
    #     if cache_gt_masks else None

    # Eval loop
    with torch.no_grad():
        for descs, (images, masks) in get_data_loader():
            # if gt_cache:
            #     gt_cache.put_batch(descs, masks)
            batch_vals = results_calc.calc_results_for(
                images=images, descs=descs,
                initial_masks={gt_pedestrian_key: masks},
                only_formula_out=True)
            batch_formula_vals = batch_vals[formula_str]
            batch_gt_formula_vals = batch_vals[gt_formula_str]

            # update quantifier metrics
            for quantifier in metrics.values():
                quantifier(batch_formula_vals, batch_gt_formula_vals)

        save_metrics({k: quant.value() for k, quant in metrics.items()})


if __name__ == "__main__":
    ex.run_commandline()
