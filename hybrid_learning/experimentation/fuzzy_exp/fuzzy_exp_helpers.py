#  Copyright (c) 2022 Continental Automotive GmbH
"""Helper functions for experiment evaluating fuzzy logic rules.
This includes definition of a standard logic to use.
Most functions are based on receiving an experiment config mapping,
as is created by Sacred experiments (see respective Sacred experiment script for config format).
"""


import collections
import hashlib
import logging
import os
import numpy as np
import pandas as pd
from typing import Hashable, Mapping, NamedTuple, Optional, Sequence, Union, Type, Dict, List, Tuple, Any, Callable
from hybrid_learning.datasets import transforms as trafos
from hybrid_learning.datasets.custom import coco
from hybrid_learning.datasets.custom.coco.keypoints_processing import person_has_rel_size
from hybrid_learning.concepts.models import ConceptDetectionModel2D as ConceptModel
from hybrid_learning.concepts import train_eval
from hybrid_learning.concepts.models import model_extension, ConceptEmbedding
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate
from hybrid_learning.experimentation.model_registry import get_model
from hybrid_learning.experimentation import ca_exp_eval
from hybrid_learning import fuzzy_logic
from hybrid_learning.fuzzy_logic.predicates import custom_ops

# Only for type checking
import torch
import PIL.Image
from hybrid_learning import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from hybrid_learning.datasets import caching


# ====================
# CUSTOM LOGIC
# ====================

def _filter_predicate_setts(logic_op: Union[fuzzy_logic.MergeBuilder, Type[fuzzy_logic.Merge], fuzzy_logic.Merge],
                            predicate_setts: Dict[str, Dict[str, Any]]
                            ) -> Dict[str, Dict[str, Any]]:
    """Filter ``predicate_setts`` to only those entries relevant for ``logic_op``.
    Allowed keys: ``SYMB`` attribute value or class ``__name__`` property.
    The returned dict entries are sorted ascending by precedence of the respective settings dict.
    Precedence: Settings for more specific classes and the ``SYMB`` value are preferred."""
    symbs: List[str] = []
    if isinstance(logic_op, fuzzy_logic.Merge):
        mro_leaf: Type[fuzzy_logic.Merge] = type(logic_op)
    elif isinstance(logic_op, fuzzy_logic.MergeBuilder):
        mro_leaf: Type[fuzzy_logic.Merge] = logic_op.merge_class
        symbs.append(logic_op.SYMB)
    else:
        mro_leaf = logic_op
    logic_op_mro = [lop for lop in mro_leaf.__mro__ if fuzzy_logic.Merge in lop.__mro__]
    # Any symbols of this logic_op class and its base classes
    # that may be used as key in predicate_setts
    symbs += [s for bcls in logic_op_mro for s in [getattr(bcls, 'SYMB', bcls.__name__), bcls.__name__]]
    # Settings for this logic operation
    # Precedence: sorted descending by class specificity, [symbol, class name]
    additional_setts: Dict[str, Dict[str, Any]] = {s: predicate_setts[s]
                                                   for s in reversed(symbs) if s in predicate_setts}
    return additional_setts


def get_logic(fuzzy_logic_key: str, predicate_setts: Dict[str, Any] = None,
              warn_about_unused_setts: bool = True) -> fuzzy_logic.Logic:
    """Get a fuzzy logic from key and settings."""
    logic: fuzzy_logic.Logic = fuzzy_logic.logic_by_name(fuzzy_logic_key)

    def apply_predicate_setts(logic_op, pred_setts=predicate_setts):
        pred_setts = pred_setts or {}
        additional_setts: Dict[str, Type[fuzzy_logic.Merge]] = {
            key: val for setts_dict in _filter_predicate_setts(logic_op, pred_setts).values()
            for key, val in setts_dict.items()}
        if additional_setts:
            logic_op = logic_op.with_(**additional_setts)
        return logic_op

    def add_op(logic_op, pos: int = 0, pred_setts=predicate_setts, log: fuzzy_logic.Logic = logic):
        log.insert(pos, apply_predicate_setts(logic_op, pred_setts))

    # Common settings
    channel_dim, mask_dims = -3, (-2, -1)

    # Standard ones
    logic.operators = [apply_predicate_setts(op) for op in logic.operators]
    # CoveredBy
    add_op(custom_ops.CoveredBy.with_(logical_and=logic.logical_('AND'), logical_or=logic.logical_('OR'), mask_dims=mask_dims), -2)
    # IoUWith
    add_op(custom_ops.IoUWith.with_(logical_and=logic.logical_('AND'), logical_or=logic.logical_('OR'), mask_dims=mask_dims), -2)
    # BestIoUWith
    add_op(custom_ops.BestIoUWith.with_(logical_and=logic.logical_('AND'), logical_or=logic.logical_('OR'), mask_dims=mask_dims), -2)
    # IsPartOfA
    add_op(custom_ops.IsPartOfA.with_(logical_and=logic.logical_('AND')), -2)
    # AllNeighbors
    add_op(custom_ops.AllNeighbors, -2)
    # AllNeighborsGT
    add_op(custom_ops.AllNeighbors.with_().symb_('GTAllNeighbors'), -2)
    # Arithmetic comparisons
    logic.extend([apply_predicate_setts(op) for op in fuzzy_logic.predicates.arithmetic.ARITHMETIC_OP_PRECEDENCE])
    # Quantifiers
    for op in [fuzzy_logic.quantifiers.ALL.with_(dim=mask_dims).symb_('AllPixels'),
               fuzzy_logic.quantifiers.ALL.with_(dim=channel_dim).symb_('AllMasks'),
               fuzzy_logic.quantifiers.ANY.with_(dim=mask_dims).symb_('AnyPixel'),
               fuzzy_logic.quantifiers.ANY.with_(dim=channel_dim).symb_('AnyMask'),
               fuzzy_logic.quantifiers.ANY.with_(dim=channel_dim, reduction=logic.logical_('OR')).symb_('MasksOR'),
               fuzzy_logic.quantifiers.WHERE,
               ]:
        add_op(op)

    # Some warning about unused predicate_setts entries
    if warn_about_unused_setts:
        unused_keys = {key: setts for key, setts in predicate_setts.items()
                       if key != '_' and key not in [symb for op in logic
                                                     for symb in _filter_predicate_setts(op, predicate_setts)]}
        if len(unused_keys):
            logging.getLogger().warning("Found unused predicate_setts (maybe typo in key?): %s", unused_keys)

    return logic


def get_logic_from_conf(conf: Dict[str, Any], warn_about_unused_setts: bool = False) -> fuzzy_logic.Logic:
    """Prepare a logic object with the correct predicates from given sacred experiment ``conf``."""
    predicate_setts = conf.get('predicate_setts', {})
    if 'predicate_setts' not in conf and 'ispartofa_setts' in conf:
        predicate_setts.update({'IsPartofA': conf['ispartofa_setts']})
    for old_key, new_key in [('ispartofa', 'IsPartOfA'), ('allneighbors', 'AllNeighbors'), ('allneighborsgt', 'GTAllNeighbors')]:
        if old_key in predicate_setts:
            predicate_setts[new_key] = predicate_setts.pop(old_key)
    return get_logic(fuzzy_logic_key=conf['fuzzy_logic_key'], predicate_setts=predicate_setts, warn_about_unused_setts=warn_about_unused_setts)


def get_formula(formula_spec: str, fuzzy_logic_key: str, predicate_setts: Dict[str, Any],
                out_key: str = 'final_', warn_about_unused_setts: bool = False) -> fuzzy_logic.Merge:
    """Parse a formula specification to a formula object of the given logic."""
    logic: fuzzy_logic.Logic = get_logic(fuzzy_logic_key, predicate_setts,
                                         warn_about_unused_setts=warn_about_unused_setts)
    return logic.parser()(formula_spec, out_key=out_key, overwrite='noop')


def get_formula_obj(conf: Dict[str, Any], parse: Union[bool, str] = 'necessary', formula_spec: str = None) -> fuzzy_logic.Merge:
    """Parse a formula object from the settings in sacred experiment ``conf`` or load the original formula object.
    Uses keys-value-pairs ``predicate_setts`` (or legacy ``ispartofa_setts``), ``fuzzy_logic_key``, ``formula_spec``."""
    keys = {conf['gt_formula_spec']: 'gt_formula', conf['formula_spec']: 'formula'}
    if parse is True or (parse == 'necessary' and formula_spec and formula_spec not in keys):
        formula_spec = formula_spec or conf['formula_spec']
        logic: fuzzy_logic.Logic = get_logic_from_conf(conf)
        formula_obj: fuzzy_logic.Merge = logic.parser()(formula_spec)
    else:
        key = 'formula' if not formula_spec else keys[formula_spec]
        if key is None:
            raise ValueError("Cannot find original formula of spec {} in conf for spec {}".format(
                formula_spec, conf['formula_spec']))
        # Fix legacy formulas and issues from depickling
        formula_obj = fix_formula_obj(conf[key], conf['fuzzy_logic_key'])

    return formula_obj


def fix_formula_obj(formula_obj: fuzzy_logic.Merge, fuzzy_logic_key: str):
    """Fix loaded objects from legacy experiments.
    Changes that require fix:

    - introduction of cache_duplicates attribute.
    - logical_and attributes were turned into OrderedDict
    """
    all_formula_objs = []
    _inter_store = [formula_obj]
    while len(_inter_store) > 0:
        f = _inter_store.pop(0)
        all_formula_objs.append(f)
        _inter_store.extend(
            [k for k in f.in_keys if isinstance(k, fuzzy_logic.Merge)])
        if hasattr(f, 'logical_and'):
            if not isinstance(f.logical_and, fuzzy_logic.Merge):
                f.logical_and = fuzzy_logic.logic_by_name(
                    fuzzy_logic_key).logical_("AND")
            else:
                _inter_store.append(f.logical_and)
        if hasattr(f, 'logical_or'):
            if not isinstance(f.logical_and, fuzzy_logic.Merge):
                f.logical_or = fuzzy_logic.logic_by_name(
                    fuzzy_logic_key).logical_("OR")
            else:
                _inter_store.append(f.logical_or)
    for f in all_formula_objs:
        f.cache_duplicates = True
        # In some strange cases it seems like the dim info gets lost from sacred pickling:
        if isinstance(f, fuzzy_logic.quantifiers.AbstractQuantifier) and f.dim == collections.OrderedDict():
            f.dim = [-2, -1]
    return formula_obj


def to_bool_predicates(config: Dict[str, Any], formula_obj: fuzzy_logic.Merge, thresh_key: str = None):
    """Replace all occurences of predicate variables by a boolean version binarized at ``thresh_key``.
    Returns a new formula object."""
    thresh_key = thresh_key or config['if_boolean_binarize_preds_at']
    to_replace = [*config["concept_pretty_names"].values(), *config["concept_pretty_names"].keys(),
                  config["pedestrian_key"]]
    replace_map = {key: f"({key}>={thresh_key})" for key in to_replace}
    bool_formula_obj = formula_obj.treerecurse_replace_keys(**replace_map)
    return bool_formula_obj


class FormulaIDInfo(NamedTuple):
    obj: fuzzy_logic.Merge
    """The (parsed) formula object."""
    spec: str
    """The string specification of the formula object."""
    pretty_spec: str
    """The pretty string specification of the formula object."""
    id: str
    """A unique ID for the formula object taking into account the formula spec, logic, and constants."""
    hash: str
    """The hash of the ``id``."""
    dir: str
    """A unique experiment directory path based on the ``id`` and its ``hash``."""


def formula_spec_to_dir(formula_spec: Union[str, fuzzy_logic.Merge], fuzzy_logic_key: str,
                        constants: Dict[str, Any] = None,
                        predicate_setts: Dict[str, Any] = None,
                        use_all_setts: bool = False,
                        maxlen: int = 42,
                        formula_obj: fuzzy_logic.Merge = None,
                        **_) -> str:
    """Legacy shortcut for ``formula_spec_to_idinfo(...).dir``."""
    info: FormulaIDInfo = formula_spec_to_idinfo(
        formula_spec=formula_spec, formula_obj=formula_obj,
        fuzzy_logic_key=fuzzy_logic_key,
        constants=constants, predicate_setts=predicate_setts, use_all_setts=use_all_setts,
        maxlen=maxlen)
    return info.dir


def formula_spec_to_idinfo(formula_spec: Union[str, fuzzy_logic.Merge] = None,
                           formula_obj: Union[str, fuzzy_logic.Merge] = None,
                           fuzzy_logic_key: str = None,
                           constants: Dict[str, Any] = None,
                           predicate_setts: Dict[str, Any] = None,
                           use_all_setts: bool = False,
                           maxlen: int = 42, **_) -> FormulaIDInfo:
    """Return a unique infos on the given ``formula_spec`` based on the formula, logic, and constants.
    The ``dir`` path is determined by the ID of the formula (see :py:func:`formula_spec_to_idinfo`),
    and the used fuzzy logic.
    The format of the folder name is:
    ``f"{f_id[:(maxlen-10)]}..{f_id_md5hash[:8]}/{fuzzy_logic_key}"``

    :param formula_spec: formula object or string
    :param fuzzy_logic_key: identifier for fuzzy logic;
        also used to parse ``formula_spec`` if that is a str
    :param constants: key-value pairs of grounded variables
    :param predicate_setts: collection of predicate settings of the format
        ``{pred_symbol: {init_key: init_val}}``
    :param use_all_setts: legacy setting to use all ``constants`` and
        ``predicate_setts`` entries without filtering
    :param maxlen: maximum total length of the string
    :param formula_obj: alternative to setting ``formula_spec`` for compatibility
    """
    if fuzzy_logic is None:
        raise ValueError("fuzzy_logic_key must be given.")
    if formula_spec is None and formula_obj is None:
        raise ValueError("Either formula_spec of formula_obj must be given.")
    
    # Formula object
    formula_obj: Union[str, fuzzy_logic.Merge] = formula_obj or formula_spec
    if not isinstance(formula_obj, fuzzy_logic.Merge):
        formula_obj: fuzzy_logic.Merge = get_formula(formula_spec, fuzzy_logic_key, predicate_setts)

    # String and pretty string spec for formula
    formula_spec: str = str(formula_obj)
    formula_pretty_str: str = formula_obj.to_pretty_str()

    # Filter constants and predicate_setts
    if not use_all_setts:
        constants = {con: val for con, val in constants.items() if con in formula_pretty_str}
        predicate_setts = {**{symb: setts for op in [formula_obj, *formula_obj.all_children]
                              for symb, setts in _filter_predicate_setts(op, predicate_setts).items()},
                           **({'_': predicate_setts['_']} if '_' in predicate_setts else {})}
    # Pretty strings for constants and predicate_setts
    consts: str = ("_" + "_".join([f"{key}{val}" for key, val in sorted(constants.items())])) if constants else ""
    pred_setts: str = ("_" + "_".join(["_".join([f"predicate_setts.{pred}.{k}{v}" for k, v in sorted(setts.items())])
                                       for pred, setts in sorted(predicate_setts.items())])) if predicate_setts else ""
    # ID string
    formula_id: str = formula_pretty_str + consts + pred_setts

    # Calc hash and print <first part of str>_<first 8 digits of hex hash>
    formula_id_hash: str = hashlib.md5(formula_id.encode()).hexdigest()
    formula_dir: str = os.path.join(f'{formula_id[:maxlen - 10]}..{formula_id_hash[:8]}',
                                    fuzzy_logic_key)
    return FormulaIDInfo(
        obj=formula_obj,
        spec=formula_spec,
        pretty_spec=formula_pretty_str,
        id=formula_id,
        hash=formula_id_hash,
        dir=formula_dir,
    )

# ====================
# DATA HANDLING
# ====================

class SwapTupleArgs(trafos.TupleTransforms):
    """Swap the two arguments in a tuple."""

    def apply_to(self, inp, target) -> Tuple:
        """Swap target and input."""
        return target, inp


class KeypointsDatasetWithDesc(coco.KeypointsDataset):
    """Wrapper around coco dataset class returning tuples of ``(descriptor, (image, anns))``.
    Default trafo turns images into tensors, anns into mask tensors and resizes all to ``img_size``.
    In case a transform is given that should apply on ``(image, anns)``,
    don't forget to use a ``trafos.UnfoldTuple()`` transform."""

    def __init__(self, img_size: Tuple[int, int], device: torch.device = None,
                 use_boxes: bool = True, reduce_gt_masks: bool = True,
                 **kwargs):
        """Overwrite the default transform."""
        super().__init__(device=device, img_size=img_size, **kwargs)
        # Turn annotations into masks and resize
        anns_to_seg_trafo = coco.COCOBoxToSegMask(img_size=img_size, merge_masks=reduce_gt_masks)\
            if use_boxes else coco.COCOSegToSegMask(img_size=img_size, merge_masks=reduce_gt_masks)
        self.transforms = trafos.OnTarget(
            (trafos.UnfoldTuple()
             + self.transforms  # pad and resize to img_size
             + trafos.OnTarget(anns_to_seg_trafo)
             + trafos.OnBothSides(trafos.ToTensor(device=device))))

    def getitem(self, i: int) -> Tuple[str, Tuple[PIL.Image.Image, List[Dict]]]:
        """Return tuple of ``(descriptor, (image, anns))``."""
        return self.descriptor(i), super().getitem(i)

    @classmethod
    def _get_default_after_cache_trafo(cls, device=None):
        return None


def get_data_loader_for(dataset: datasets.BaseDataset, loader_setts: Dict[str, Any],
                        device: Union[str, torch.device], show_progress_bars: bool):
    """Uncaptured version of ``get_data_loader``."""
    dataloader: DataLoader = train_eval.loader(dataset, device=device, **loader_setts)
    if show_progress_bars:
        dataloader: tqdm = tqdm(dataloader)
    return dataloader


def get_data(img_size: Tuple[int, int], dataset_root: str, split: str,
             root: str = None,
             predicate_setts: Dict[str, Any] = None, default_person_sizes: Dict[str, Tuple[str, str]] = None,
             reduce_gt_masks: Optional[bool] = None,
             device: str = None, show_progress_bars: bool = True,
             use_boxes: bool = True, _log: logging.Logger = None, **_) -> coco.KeypointsDataset:
    """Return data triple of COCO keypoint datasets
    with output transformed to semantic segmentation of persons.
    Allow to subset the annotations to only contain persons of a certain
    estimated relative ``(min, max)`` size range.
    Cache subsetted annotations."""
    if root:
        dataset_root = os.path.join(root, dataset_root)
    person_size: Optional[Tuple[float, float]] = None \
        if (not default_person_sizes or not predicate_setts.get('_', {}).get('person_size_key', None)) \
        else default_person_sizes[predicate_setts['_']['person_size_key']]
    reduce_gt_masks = reduce_gt_masks if reduce_gt_masks is not None else \
        predicate_setts.get('_', {}).get('reduce_gt_masks', True)

    img_base = {'TRAIN_VAL': 'train2017', 'TEST': 'val2017'}[split]
    imgs_root = os.path.join(dataset_root, 'images', img_base)
    ann_fp: str = os.path.join(
        dataset_root, "annotations",
        "person_keypoints_" + coco.ConceptDataset.settings_to_str(
            dataset_root=imgs_root,
            img_size=img_size,
            person_rel_size_range=person_size
        ) + ".json")
    if os.path.isfile(ann_fp):  # load from file without subsetting
        condition, annotations_fp = None, ann_fp
    else:  # enable subsetting by person size
        annotations_fp = None
        condition = None if person_size is None else \
            lambda i, a: person_has_rel_size(
                i, a, min_rel_height=person_size[0],
                max_rel_height=person_size[1], img_target_size=img_size)

    # Actual data acquisition
    _log.info("Loading and subsetting dataset split %s", split) if _log else None
    dataset: KeypointsDatasetWithDesc = KeypointsDatasetWithDesc(
        dataset_root=imgs_root,
        split=split, img_size=img_size,
        annotations_fp=annotations_fp,
        device=device,
        use_boxes=use_boxes,
        reduce_gt_masks=reduce_gt_masks,
    )
    if condition and not annotations_fp:
        dataset = dataset.subset(condition=condition,
                                 show_progress_bar=show_progress_bars,
                                 license_ids=None)

    # Annotations caching
    if person_size is not None and len(dataset) > 0 \
            and not os.path.isfile(ann_fp):
        dataset.to_raw_anns(description=(
            "Subset of MS COCO dataset with persons of relative size in"
            " the range [{}, {}] assuming images are padded & scaled "
            "to HxW={}x{}").format(*person_size, *img_size),
                            save_as=ann_fp)

    return dataset

# ====================
# RESULTS HANDLING
# ====================

def save_metrics_to(metric_vals: Dict[str, float],
                    metrics_logfile: str = None, metrics_logdir: str = None,
                    log_scalar: Callable[[str, float], Any] = None
                    ) -> Tuple[Dict[str, float], Dict[str, plt.Figure]]:
    """Save dict ``{metric_name: val}`` to ``metrics_logdir/metrics.csv`` and/or using ``log_scalar`` method.
    Return two dicts, on with the numerical values, one with the figures contained in ``metric_vals`` dict.
    """
    # clean up types, especially tensors
    for key, val in metric_vals.items():
        val = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else val
        val = val.item() if isinstance(val, np.ndarray) and val.size == 1 else val
        metric_vals[key] = val
    numeric_data: Dict[str, float] = {k: v for k, v in metric_vals.items() if np.isscalar(v)}

    # csv log of numeric values
    metrics_logfile = metrics_logfile or \
        (os.path.join(metrics_logdir, 'metrics.csv') if metrics_logdir else None)
    if metrics_logfile is not None:
        if os.path.exists(metrics_logfile):
            raise FileExistsError("Log CSV file {} already exists!".format(metrics_logfile))
        pd.Series(numeric_data).reset_index().to_csv(metrics_logfile, index=False)

    # sacred log
    if log_scalar is not None:
        for quant_name, res in numeric_data.items():
            log_scalar(quant_name, res)

    # fig logs
    figs: Dict[str, plt.Figure] = {k: v for k, v in metric_vals.items() if isinstance(v, plt.Figure)}
    if metrics_logdir is not None:
        for title, fig in figs.items():
            fig.savefig(os.path.join(metrics_logdir, f'{title}.png'))
            fig.savefig(os.path.join(metrics_logdir, f'{title}.svg'))
            torch.save(fig, os.path.join(metrics_logdir, f'{title}.pt'))  # enable later recovery of figure values
    
    return numeric_data, figs


def default_logdir(conf: Dict[str, Any]) -> str:
    """Return the default logging directory root."""
    return os.path.join(  # metrics logging dir
        conf["experiment_root"],
        formula_spec_to_dir(conf["formula_spec"], conf["fuzzy_logic_key"], conf["constants"], conf["predicate_setts"]),
    )


# ====================
# MODEL HANDLING AND CACHING
# ====================

def get_predicates(concept_model_root: str, concept_model_args: dict,
                   concept_to_layer: Dict[str, str], concept_pretty_names: Dict[str, str],
                   pedestrian_key: str,
                   root: str = None,
                   # For loading models via config and model registry:
                   _config: Dict[str, Any] = None, model_key: str = None,
                   # For handing over models directly:
                   main_model: torch.nn.Module = None, main_model_trafo: torch.nn.Module = None, device: torch.device = None,
                   **_):
    """Get the predicate function: Takes an image and returns a dict with all predicate values."""
    assert model_key is not None or (main_model is not None and main_model_trafo is not None)
    if root:
        concept_model_root = os.path.join(root, concept_model_root)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    # Concept models (with sigmoid activation)
    c_models_by_layer: Dict[str, Dict[str, ConceptModel]] = {l_id: {} for l_id in concept_to_layer.values()}
    for concept_name, layer_id in concept_to_layer.items():
        c_emb: ConceptEmbedding = ca_exp_eval.get_best_emb(
            layer_id, root=ca_exp_eval.analysis_root(layer_id, concept_name, root=concept_model_root))
        c_model = ConceptModel.from_embedding(c_emb, legacy_warnings=False, **concept_model_args).eval().to(device)
        assert c_model.concept_name == concept_name, \
            "concept_name for model {} differs from the one used for loading {}".format(c_model, concept_name)
        c_models_by_layer[layer_id].update({concept_pretty_names.get(concept_name, concept_name): c_model})

    # Main model
    main_model: torch.nn.Module = main_model or \
        get_model(model_key=model_key, layer_infos=c_models_by_layer, config=_config)
    main_model_trafo: torch.nn.Module = main_model_trafo or \
        get_model(model_key=f'{model_key}_trafo', config=_config, check_layer_infos=False)

    # Joint model: return dict of {'concept': seg_mask, ...} (the predicate values)
    predicates: model_extension.ModelExtender = model_extension.ModelExtender(
        main_model.eval().to(device), return_orig_out=False, extensions={
            '': {pedestrian_key: main_model_trafo.eval().to(device)},
            **c_models_by_layer
        })
    return predicates


def default_uncollate(batch_out: Union[torch.Tensor, Dict[str, torch.Tensor]],
                      batch_size: int = None) -> Union[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """Uncollate a batch of predicate outputs previously collated by torch default_collate"""
    if isinstance(batch_out, torch.Tensor):
        return batch_out  # tensor already is iterable over batch
    if isinstance(batch_out, dict):
        batch_size: int = batch_size if batch_size is not None else \
            list(batch_out.values())[0].size()[0]
        return [{k: v[i] for k, v in batch_out.items()}
                for i in range(batch_size)]
    raise NotImplementedError("Uncollate currently only defined for dict but got {}".format(type(batch_out)))


def cached_eval(descriptors: List[str], input_batch: Any,
                model: Callable[[Any], Union[torch.Tensor, Dict[str, torch.Tensor]]],
                cache: caching.Cache = None,
                device: Union[str, torch.device] = None):
    """Given ``descriptors`` load ``model`` outputs either from ``cache``
    or generate them from ``input_batch``."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    # load from cache if possible:
    cached_out: Optional[List[Dict[str, torch.Tensor]]] = cache.load_batch(descriptors) \
        if cache is not None else None
    if cached_out:
        model_out = default_collate(cached_out)
    else:  # else evaluate predicates
        if isinstance(model, torch.nn.Module):
            model = model.eval().to(device)
            input_batch = input_batch.to(device)
        model_out = model(input_batch)
        # stack c-model list output
        if isinstance(model_out, dict):
            model_out = {k: (v.squeeze(0) if len(v.size()) == 5 else v)  # TODO: remove special treatment for concepts
                         for k, v in model_out.items()}
        # put
        if cache is not None:
            cache.put_batch(descriptors, default_uncollate(model_out))
    return model_out


def mask_cache(subdir: str, results_cache_dir: str,
               device: Union[str, torch.device]) -> caching.Cache:
    """Standard mask cache."""
    if not results_cache_dir:
        return caching.NoCache()
    return caching.PTCache(cache_root=os.path.join(results_cache_dir, subdir),
                           dtype=torch.bfloat16, device=device)
                           

def predicates_cache_for(concept_to_layer: Dict[str, str], concept_pretty_names: Dict[str, str],
                         fuzzy_logic_key: str, pedestrian_key: str,
                         results_cache_dir: str, root: str = None, device: Union[str, torch.device] = None,
                         reduce_pred_masks: bool = None, predicate_setts: Dict[str, Any] = None, **_
                         ) -> caching.CacheDict:
    """Provide caches with folder structure ``<pedestrian_key>/<logic>/*.pt`` and ``<concept>/<layer>/*.pt``"""
    device = device or 'cpu'
    if root:
        results_cache_dir = os.path.join(root, results_cache_dir)
    if reduce_pred_masks is None:
        reduce_pred_masks: bool = (predicate_setts or {}).get('_', {}).get('reduce_pred_masks', True)
    pred_cache: caching.CacheDict = caching.CacheDict({
        pedestrian_key: mask_cache(subdir=os.path.join(pedestrian_key + ('_unreduced' if not reduce_pred_masks else ''),
                                                       fuzzy_logic_key),
                                   results_cache_dir=results_cache_dir, device=device),
        **{concept_pretty_names.get(cn, cn): mask_cache(subdir=os.path.join(concept_pretty_names.get(cn, cn), l_id),
                                                        results_cache_dir=results_cache_dir, device=device)
           for cn, l_id in concept_to_layer.items()}})
    return pred_cache


def formula_trafo_caches_for(formulas_to_cache: List[str], fuzzy_logic_key: str,
                             constants: Dict[str, Any], predicate_setts: Dict[str, Any],
                             results_cache_dir: str, root: str = None, device: Union[str, torch.device] = None,
                             keys_allowed_from: Sequence[Optional[fuzzy_logic.Merge]] = None,
                             disable_formula_caching: bool = False,
                             _log: logging.Logger = None, **_) -> Optional[Dict[str, caching.Cache]]:
    """Get caches for formula transformation outputs."""
    device = device or 'cpu'
    if root:
        results_cache_dir = os.path.join(root, results_cache_dir)
    # Cache structure: <formula>_<constants>/<logic>/*.pt
    if disable_formula_caching:
        return None
    # Filter out keys not occurring as child
    keys_allowed_from = [f for f in (keys_allowed_from or []) if f is not None and isinstance(f, fuzzy_logic.Merge)]
    allowed_keys: Optional[List[str]] = None if not keys_allowed_from else \
        set([child.out_key for f in keys_allowed_from for child in f.all_children] \
            + [f.out_key for f in keys_allowed_from]
            + [str(f) for f in keys_allowed_from])
    # Collect caches
    caches = {}
    for formula_spec in set(formulas_to_cache):
        curr_formula: fuzzy_logic.Merge = get_formula(formula_spec, fuzzy_logic_key, predicate_setts)
        curr_formula_str: str = str(curr_formula)
        if _log and allowed_keys and curr_formula_str not in allowed_keys:
            _log.warning("Intermediate formula string specified for caching is no intermediate output: %s",
                              curr_formula_str)
            continue
        # Cache structure: <gt_formula>_<constants>/<logic>/*.pt
        caches[curr_formula_str] = mask_cache(subdir=formula_spec_to_dir(
            curr_formula, fuzzy_logic_key, constants, predicate_setts),
            results_cache_dir=results_cache_dir, device=device)

    return caches


class FormulaEvaluator:
    """Evaluate predicates and formulas with intermediate values cached according to a config."""

    def __init__(self, conf: Mapping[str, Any] = None,
                 formulas: Union[Sequence[fuzzy_logic.Merge], Dict[str, fuzzy_logic.Merge]] = None,
                 additional_formulas: Union[Sequence[fuzzy_logic.Merge], Dict[str, fuzzy_logic.Merge]] = None,
                 predicates: torch.nn.Module = None,
                 formula_caches: Optional[Dict[str, caching.Cache]] = None,
                 predicates_cache: caching.Cache = None,
                 constants: Dict[str, Any] = None,
                 changed_constants: Dict[str, Any] = None,
                 changed_root: str = None,
                 device: Union[str, torch.device] = None,
                 ):
        """Init.
        To disable caching, set ``conf['results_cache_dir']=None`` or assign ``caching.NoCache()``
        to ``formula_caches`` and ``predicates_cache``.
        
        :param conf: the experiment config to use for setting defaults;
            may be skipped if all other settings are given
        :param formulas: the formulas to evaluate (see :py:attr:`formulas`);
            defaults to ``[conf['formula'], conf['gt_formula']]``;
            may be given as dict ``{new_out_key: formula_object}`` or as sequence of formula objects,
            in which case the ``formula.out_key`` is used as ``new_out_key``.
        :param additional_formulas: add these to ``formulas`` (after setting defaults to ``formulas``);
            as ``formulas``, may be dict or sequence
        :param predicates: the predicates model (see :py:attr:`predicates`);
            defaults to the output of :py:func:`get_predicates`
        :param formula_caches: dictionary of ``{out_key: cache}`` for caching formula (intermediate)
            outputs (see :py:attr:`formula_caches`);
            defaults to output of :py:func:`formula_trafo_caches_for`;
            set to ``{}`` to disable
        :param predicates_cache: the cache used for the ``predicates`` outputs;
            defaults to output of :py:func:`predicates_cache_for`;
            set to ``caching.NoCache()`` to disable
        :param constants: constants to use; defaults to ``conf['constants']``
        :param changed_constants: update ``constants`` (also default value)
        :param changed_root: prepend ``changed_root`` to all used paths from ``conf``
        :param device: device to use for evaluation;
            defaults to ``conf['device']``
        """
        self.conf: Mapping[str, Any] = conf or {}
        """The configuration mapping used to produce any default settings."""
        self.device: Union[str, torch.device] = device or \
            self.conf.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        """The device to use for evaluation."""
        
        # Predicates & Caches
        self.predicates: torch.nn.Module = (predicates or get_predicates(
            root=changed_root, **{**self.conf, 'device': self.device}, _config=self.conf)
        ).to(self.device)
        """The model producing the dictionary of predicate outputs."""
        self.predicates_cache: caching.Cache = predicates_cache or predicates_cache_for(root=changed_root, **self.conf)
        """Cache for predicate outputs."""

        # Constants
        constants = constants if constants is not None else conf.get('constants', {})
        self.constants: Dict[str, Any] = {**constants, **(changed_constants or {})}
        """The constants to supply to the formula evaluation.
        Have higher precendence than newly calculated predicate outputs."""

        # Formulas & Caches
        to_dict = lambda fs: fs if isinstance(fs, collections.Mapping) else {str(f): f for f in fs}
        formulas = formulas or [get_formula_obj(conf), *([get_formula_obj(conf, formula_spec=conf['gt_formula_spec'])] if conf['gt_formula_spec'] else [])]
        self.formulas: Dict[str, fuzzy_logic.Merge] = {
            name: f for name, f in {**to_dict(formulas), **to_dict(additional_formulas or {})}.items()
            if f is not None}
        """The formula objects to evaluate given as dictionary of the form ``{new_out_key: formula_object}``.
        The ``new_out_key`` is used instead of the ``formula_object.out_key`` for the final output.
        The latter is only kept if there is a cache registered for it, see :py:attr:`formula_caches`."""

        self.formula_caches: Optional[Dict[str, caching.Cache]] = formula_caches or formula_trafo_caches_for(
            root=changed_root,
            **{**conf, 'formulas_to_cache': conf.get('formulas_to_cache', {})})
        """Dictionary of ``{out_key: cache}`` for caching formula (intermediate) outputs."""
        if self.formula_caches:
            for formula in self.formulas.values():
                formula.keep_keys = [*(formula.keep_keys or []), *self.formula_caches.keys()]
        
        self._same_size = trafos.SameSizeTensorValues(mode=conf.get('predicate_setts', {}).get('_', {}).get('same_size_mode', None))

    @property
    def used_constants(self) -> Dict[str, Any]:
        """The constants actually used in one of the formulas."""
        return {k: v for k, v in self.constants.items()
                if any(k in formula.all_in_keys for formula in self.formulas.values())}

    def restrict_to_used_constants(self) -> 'FormulaEvaluator':
        """Set ``constants`` to a dict only containing the used constants."""
        used_constants = self.used_constants
        self.constants = used_constants
        return self

    def calc_results_for(self, images: torch.Tensor,
                         initial_masks: Dict[str, torch.Tensor] = None,
                         descs: Sequence[Hashable] = None,
                         only_formula_out: bool = False) -> Dict[str, Any]:
        """Calculate predicate and formula outputs for a batch of input images.

        :param images: batch of input images of shape ``[batch, channels, height, width]``
        :param initial_masks: any further (pre-calculated) inputs to the formula calculation,
            e.g. the ground truth masks;
            these take precedence over newly calculated predicate outputs and :py:attr:`constants`
        :param descs: the descriptors of the images for obtaining respective predicate
            outputs from :py:attr:`predicates_cache`
        :param only_formula_out: whether to prune the output to only contain the output values
            of the formulas from :py:attr:`formulas`
        :return: dictionary containing the outputs of the formulas as ``{out_key: output}``
            with ``out_key`` from :py:attr:`formulas`;
            if ``only_formula_out`` is ``False``, also the following intermediate outputs are added:
            predicate outputs indexed by the prediate name,
            all formula outputs and intermediate outputs according to the formula ``keep_keys`` settings
            (by default, all formula intermediate outputs registered for caching are added to this),
            all ``initial_masks``,
            :py:attr:`constants`
        """
        # calc predicate values
        batch_pred_out: Dict[str, torch.Tensor] = \
            cached_eval(descs, images, model=self.predicates, cache=self.predicates_cache, device=self.device)
        calced_formula_inp: Dict[str, torch.Tensor] = self._same_size({
            **batch_pred_out, **self.constants, **initial_masks})

        # get cached formula (intermediate) outputs
        formula_vals_from_cache: Dict[str, Optional[List[torch.Tensor]]] = \
            {k: c.load_batch(descs) for k, c in self.formula_caches.items()} if self.formula_caches else {}
        cached_formula_out: Dict[str, torch.Tensor] = \
            {k: torch.stack(v) for k, v in formula_vals_from_cache.items() if v is not None}

        # evaluate formulas on predicate outputs & ground truth
        all_vals: Dict[str, torch.Tensor] = {**cached_formula_out, **calced_formula_inp}
        batch_formula_out: Dict[str, torch.Tensor] = {}
        for formula_str, formula in self.formulas.items():
            all_vals.update(formula({**cached_formula_out, **all_vals, **calced_formula_inp}))
            # change out_key of formula according to formula_str in formulas
            all_vals[formula_str] = batch_formula_out[formula_str] = all_vals[formula.out_key]
            if formula.out_key not in self.formula_caches:
                del all_vals[formula.out_key]
        
        # store yet uncached stuff:
        if self.formula_caches:
            to_cache = {**all_vals, **batch_formula_out}
            for key in [k for k, v in formula_vals_from_cache.items()
                        if v is None and k in to_cache]:
                self.formula_caches[key].put_batch(descs, to_cache[key])

        if only_formula_out:
            return batch_formula_out
        else:
            return all_vals

    def calc_result_for(self, image: torch.Tensor,
                        initial_masks: Dict[str, torch.Tensor],
                        desc: Hashable = None,
                        only_formula_out: bool = False):
        """Wrapper around ``calc_results_for`` that handles single inputs instead of a batch.
        
        :return: output of :py:meth:`calc_results_for` but with singular batch dimension
            (first dimension) squeezed for all tensors
        """
        results = self.calc_results_for(
            images = image.unsqueeze(0),
            initial_masks = {k: mask.unsqueeze(0) if isinstance(mask, torch.Tensor) else mask
                             for k, mask in (initial_masks or {}).items()},
            descs = [desc] if desc is not None else None,
            only_formula_out=only_formula_out,
        )
        return {key: val.squeeze(0) if isinstance(val, torch.Tensor) and val.size()[0] == 1 else val
                for key, val in results.items()}

