#  Copyright (c) 2022 Continental Automotive GmbH
"""
Helper functions to evaluate fuzzy logic experiments.
Includes functions for loading experiment results, post-processing them,
and to do further evaluations.
"""
import json
import logging
import os
import warnings
from typing import Dict, Any, Optional, Tuple, Iterable, List, Callable, Union, Sequence

import PIL.Image
import matplotlib
import pandas as pd
import torch
from tqdm import tqdm

import hybrid_learning.datasets.transforms as trafos
from hybrid_learning.datasets import caching
from hybrid_learning import fuzzy_logic
from hybrid_learning.experimentation.model_registry.fuzzy_exp_models import register_all_model_builders
from . import fuzzy_exp_helpers as calc_helpers
from ..exp_eval_common import auc, do_for, constrain_pd

try:
    from IPython.display import display
except ImportError:
    display = print

try:
    from sacred.config import load_config_file as load_sacred_config_file
except ImportError:
    def load_sacred_config_file(*_args, **_kwargs):
        raise ImportError("Loading sacred config files requires sacred package to be installed.")

to_tens = trafos.ToTensor(sparse=False)
FIG_SAVE_DIR = "plots"
PROJECT_ROOT: str = ".."

# Suppress the unnecessary deprecation warning about reusing an axis
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

register_all_model_builders()


# ===============================
# RAW EXPERIMENT RESULT LOADING
# ===============================

def get_exp_conf(sacred_logdir: str, conf_validation_vals: Dict[str, str] = None) -> Dict[str, Any]:
    """Get the sacred config dict for the experiment under ``sacred_logdir``."""

    # Select the logdir
    logs_dirnames = [d for d in os.listdir(sacred_logdir) if d != "_sources"]
    completed, failed, running = [], [], []
    for log_dirn in logs_dirnames:
        runfile = os.path.join(sacred_logdir, log_dirn, 'run.json')
        if not os.path.isfile(runfile) or not os.path.getsize(runfile) > 0:
            failed.append(log_dirn)
            continue
        with open(runfile, 'r') as rfile:
            content = json.load(rfile)
            if content.get('status', False) == 'COMPLETED':
                completed.append(log_dirn)
            elif content.get('status', False) == 'RUNNING':
                running.append(log_dirn)
            else:
                failed.append(log_dirn)
    if len(failed) > 0:
        logging.getLogger().warning("Found %d log dirs for failed or interrupted experiments in %s:\n%s",
                                    len(failed), sacred_logdir, '\n'.join([os.path.join(sacred_logdir, d) for d in failed]))
    if len(running) > 0:
        logging.getLogger().warning("Found %d log dirs for running experiments in %s:\n%s",
                                    len(running), sacred_logdir, '\n'.join([os.path.join(sacred_logdir, d) for d in running]))
    if len(completed) > 1:
        raise ValueError("Found log directories for more than one completed experiment in {}:\n{}".format(
            sacred_logdir, '\n'.join([os.path.join(sacred_logdir, d) for d in completed])))
    elif len(completed) < 1:
        raise ValueError("Found no log directories for a completed experiment in {}".format(sacred_logdir))
    conf_file: str = os.path.join(sacred_logdir, completed[0], "config.json")
    conf: dict = load_sacred_config_file(conf_file)

    # Legacy compatibility
    if 'model_key' not in conf:
        conf['model_key'] = 'maskrcnn'

    # Validation
    formula_infos: calc_helpers.FormulaIDInfo = calc_helpers.formula_spec_to_idinfo(**conf)

    err_msg = f"Failed loading conf for {sacred_logdir};" + "\nproblem: {}" + "\nconf content:\n{}".format(
        '\n'.join([f'{key}: {type(val)}' for key, val in conf.items()]))
    for key, expected_val in (conf_validation_vals.items() if conf_validation_vals is not None else []):
        assert key in conf, err_msg.format("Loaded configuration missing key {}".format(key))
        assert conf[key] == expected_val, err_msg.format(
            "conf[{}] ({}) != expected_val ({})".format(key, conf[key], expected_val))
    logdir_root_root = os.path.basename(os.path.dirname(os.path.dirname(sacred_logdir)))
    if ".." in logdir_root_root:
        formula_dir = formula_infos.dir
        logdir_root_root_front, _ = logdir_root_root.split("..")
        assert formula_dir[:min(len(formula_dir), len(logdir_root_root_front))] in logdir_root_root_front, \
            err_msg.format("formula_dir ({}) not in logdir_root_root ({})"
                           .format(formula_dir, logdir_root_root))
    # Legacy compatibility
    else:
        formula_dir = formula_infos.id
        assert formula_dir in logdir_root_root, \
            err_msg.format("formula_dir ({}) not in logdir_root_root ({})"
                           .format(logdir_root_root, formula_dir))
    concept_pretty_names = conf.get('concept_pretty_names', {})
    formula_concept_keys = (formula_infos.obj.all_in_keys -
                            {conf['pedestrian_key'], conf.get(
                               'gt_pedestrian_key', None), *conf.get('constants', {}).keys()})
    concept_to_layer_keys = [concept_pretty_names.get(key, key) for key in conf['concept_to_layer'].keys()]
    assert sorted(formula_concept_keys) == sorted(concept_to_layer_keys), \
        err_msg.format("(formula_concept_keys ({}) != concept_to_layer_keys ({})"
                       .format(sorted(formula_concept_keys), sorted(concept_to_layer_keys)))

    return conf


formula_display_replace = {
    'LEFT_EYE-RIGHT_EYE': 'eye',
    'LEFT_LEG-RIGHT_LEG': 'leg',
    'LEFT_ARM-RIGHT_ARM': 'arm',
    'LEFT_WRIST-RIGHT_WRIST': 'wrist',
    'LEFT_ANKLE-RIGHT_ANKLE': 'ankle',
    'pedestrian': 'person'
}


def formula_to_display_name(formula: str, max_len: int = 40):
    """Return a possibly shortened title version of the given formula string."""
    for replace_a, with_b in formula_display_replace.items():
        formula = formula.replace(replace_a, with_b)
    if max_len and len(formula) > max_len:
        formula = formula[:max_len-3] + "..."
    return formula


def load_orig_and_masks(img_fn: str, orig_size: Optional[Tuple[int, int]] = None,
                        imgs_dir: str = None, pedestrian_cache_dir: str = None, gt_pedestrian_cache_dir: str = None,
                        concept_cache_dirs: Dict[str, str] = None, final_cache_dir: str = None,
                        pedestrian_key: str = "pedestrian", gt_pedestrian_key: str = "gt_pedestrian",
                        device=None, use_pretty_names: bool = True, raise_on_missing: bool = False,
                        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Load the original image and masks at index ``image_id`` from configured cache directories.

    :param orig_size: size in ``(height, width)`` to which to pad and resize the original image;
        if not given defaults to size of pedestrian mask, and no resizing if this is unknown
    :param final_cache_dir: cache directory for the formula output files
    :param use_pretty_names: whether to transform mask names using :py:func:`formula_to_display_name`
    :param pedestrian_key: mask key for person mask
    :param gt_pedestrian_key: mask key for ground truth person mask
    :param device: onto which device to load the masks
    :param raise_on_missing: whether to raise when a mask file is missing or just set the respective value to ``None``
    :returns: tuple ``(original_image, masks_dict)``
        with the special mask dict keys ``'formula'`` for the formula mask, and
        ``pedestrian_key`` for the DNN output."""
    # get images / masks
    masks = {}

    def torch_load_from(d: str, mandatory: bool = raise_on_missing):
        fp = os.path.join(d, img_fn + '.pt')
        if os.path.exists(fp):
            return to_tens(torch.load(fp, map_location=device)).float()
        elif mandatory:
            raise FileNotFoundError(fp)
        else:
            return None
    if pedestrian_cache_dir:
        masks[pedestrian_key] = torch_load_from(pedestrian_cache_dir)
    if gt_pedestrian_cache_dir:
        masks[gt_pedestrian_key] = torch_load_from(gt_pedestrian_cache_dir, mandatory=False)
    if concept_cache_dirs:
        masks.update({c: torch_load_from(c_dir)
                     for c, c_dir in concept_cache_dirs.items()})
    if final_cache_dir:
        masks["formula"] = torch_load_from(final_cache_dir)
    img_t = None
    if imgs_dir:
        orig_img_fp = os.path.join(imgs_dir, img_fn)
        if os.path.exists(orig_img_fp):
            orig_img = PIL.Image.open(orig_img_fp)
            if orig_img.mode != 'RGB':
                orig_img = orig_img.convert('RGB')
            img_t = to_tens(orig_img)
            orig_size = orig_size or (masks[pedestrian_key].size()[-2:]
                                      if masks.get(pedestrian_key, None) is not None else None)
            if orig_size is not None:
                img_t = trafos.pad_and_resize(img_t, orig_size).float()
        elif raise_on_missing:
            raise FileNotFoundError(orig_img_fp)
    if use_pretty_names:
        masks = {formula_to_display_name(key): val for key, val in masks.items()}
    return img_t, masks


def get_caches(root: str, conf: Dict[str, Any], discard_non_existing: bool = False) -> Dict[str, Union[str, Dict[str, str]]]:
    """Given root and config provide the cache arguments needed for ``load_orig_and_masks``"""
    reduce_pred_masks: bool = conf['predicate_setts'].get('_', {}).get('reduce_pred_masks', True)
    pred_caches: caching.CacheDict = calc_helpers.predicates_cache_for(**conf, reduce_pred_masks=reduce_pred_masks)
    c_caches: Dict[str, Optional[str]] = {k: getattr(c, 'cache_root', None)
                                          for k, c in pred_caches.cache_dict.items()}
    ped_cache_dir: Optional[str] = c_caches.pop(conf['pedestrian_key'], None)
    formula_str = str(calc_helpers.get_formula_obj(conf, parse=False))
    gt_formula_str = str(calc_helpers.get_formula_obj(conf, parse=False, formula_spec=conf['gt_formula_spec']))
    formula_caches: Dict[str, caching.PTCache] = calc_helpers.formula_trafo_caches_for(**{**conf, 'formulas_to_cache': [formula_str, gt_formula_str]})
    
    other_cache_dirs: Dict[str, str] = dict(
        pedestrian_cache_dir=os.path.join(root, ped_cache_dir) if ped_cache_dir else None,
        gt_pedestrian_cache_dir=os.path.join(root, formula_caches[gt_formula_str].cache_root)
            if 'gt_pedestrian_key' in conf else None,  # and conf['gt_pedestrian_key'] in calc_helpers.get_formula_obj(conf, parse=False).all_in_keys
        final_cache_dir=os.path.join(root, formula_caches[formula_str].cache_root),
        imgs_dir=os.path.join(root, conf['dataset_root'], 'images',
                            {'TRAIN_VAL': 'train2017', 'TEST': 'val2017'}[conf['split']]),
    )
    concept_cache_dirs: Dict[str, str] = {c: os.path.join(root, d) for c, d in c_caches.items()}
    if discard_non_existing:
        other_cache_dirs = {k: v for k, v in other_cache_dirs.items() if os.path.isdir(v)}
        concept_cache_dirs = {k: v for k, v in concept_cache_dirs.items() if os.path.isdir(v)}
    return dict(**other_cache_dirs, concept_cache_dirs=concept_cache_dirs)


class ResultsIterator:
    """Allow to get all output infos and potentially additionally calculated values for one sample."""

    def __init__(self, conf: dict, root: str = PROJECT_ROOT,
                 load_images: bool = True, device: str = None,
                 only_filenames: Iterable = None, raise_on_missing: bool = False,
                 recalc: Union[bool, str] = "necessary",
                 additional_formulas: Dict[str, fuzzy_logic.Merge] = None,
                 additional_formula_mods: Dict[str, Callable[[
                     fuzzy_logic.Merge, fuzzy_logic.Merge], fuzzy_logic.Merge]] = None,
                 changed_constants: Dict[str, Any] = None,
                 verbose: bool = False, use_pretty_names: bool = True):
        """Given a sacred experiment ``conf`` dict, yield pairs of
        image file name, original image, and dict of output masks for that experiment.
        This is a convenience wrapper around :py:func:`load_orig_and_masks` and :py:func:`recalc_formula_masks`.
        Included pairs can be restricted by setting the ``only_filenames`` list.
        Included masks are the concept model outputs and the fuzzy logic formula output.
        Formula masks may be recalculated or further ones calculated and added via
        ``additional_formula_mods`` argument.

        :param conf: the experiment config dict (loaded using ``get_exp_config``)
        :param root: the root folder in which the experiment was run
        :param load_images: whether to also load the original images; else set to ``None``
        :param recalc: whether to recalculate missing ``'formula'`` mask;
            set to ``True`` or ``'always'`` to always recalculate the formula mask,
            set to ``False`` or ``None`` to never calculate masks (also disable calculation of ``additional_formula_mods``),
            set to ``"necessary"`` to calculate ``additional_formula_mods`` and only recalculate missing formula masks.
        :param additional_formula_mods: dict of callables that accept the original formula object and
            the ground truth formula object of the experiment, and return a new formula object
            that shall be calculated (cf. ``recalc``).
            Recalculated formulas are added to the output masks using the keys from the ``additional_formula_mods`` dict.
            This can also be used to overwrite the original formula(s) by using the keys
            ``'formula'`` and ``'gt_formula'``.
            Example application: For a monitor formula, add its underlying formula body
            ``{'body': lambda f, gt_f: f.in_keys[0]}`` (assuming ``f=NOT(some_body)``).
        :param verbose: in case of recalculation, print some recalcuation information
        :param use_pretty_names: whether to transform mask names using :py:func:`formula_to_display_name`
        :param raise_on_missing: see :py:func:`load_orig_and_masks`
        :yields: tuples of ``(image_filename, (original_image_or_none, dict_of_masks))``;
            for the format of the second tuple entry have a look at ``load_orig_and_masks``.
        """
        self.conf: Dict[str, Any] = conf

        # CACHE DIRS
        self.caches: Dict[str, Union[str, Dict[str, str]]] = get_caches(root, self.conf, discard_non_existing=True)
        if not load_images:
            self.caches.update(imgs_dir=None)
        self.img_fns: List[str] = only_filenames if only_filenames is not None \
            else [img_pt.rsplit('.pt', maxsplit=1)[0] for img_pt in os.listdir(self.caches['pedestrian_cache_dir'])]

        # RECALCULATION SETTINGS
        self.additional_formulas: Dict[str, trafos.fuzzy_logic.Merge] = additional_formulas or {}
        self.backup_formulas: Dict[str, trafos.fuzzy_logic.Merge] = {}
        self.changed_constants: Dict[str, Any] = changed_constants or {}
        self.predicates: Optional[torch.nn.Module] = None
        self.masks_dict_trafo: Optional[trafos.DictTransform] = None
        self.device: Union[str, torch.device] = device
        
        # Feature flags
        self.raise_on_missing: bool = raise_on_missing
        self.use_pretty_names: bool = use_pretty_names

        recalc = True if changed_constants or additional_formula_mods or self.additional_formulas else recalc
        self._prepare_recalc(recalc=recalc, additional_formula_mods=additional_formula_mods, root=root, verbose=verbose)
    
    def __len__(self):
        return len(self.img_fns)

    def __getitem__(self, i: Union[str, int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, bool, float]]]]:
        img_fn: str = i if isinstance(i, str) else self.img_fns[i]
        return img_fn, self.get_all_masks(img_fn)

    def _prepare_recalc(self, recalc: Union[str, bool] = 'necessary',
                        root: str = None, additional_formula_mods=None, verbose=False):
        if not recalc: return
        self.predicates = calc_helpers.get_predicates(**{**self.conf,
                                        'concept_model_root': os.path.join(root, self.conf['concept_model_root']),
                                        'device': self.device,},
                                        _config=self.conf)
        formula_obj: fuzzy_logic.Merge = calc_helpers.get_formula_obj(self.conf, parse=False)
        gt_formula_obj: fuzzy_logic.Merge = calc_helpers.get_formula_obj(
            self.conf, formula_spec=self.conf['gt_formula_spec'], parse=False)
        self.masks_dict_trafo = trafos.SameSizeTensorValues(
            mode=self.conf.get('predicate_setts', {}).get('_', {}).get('same_size_mode', None))
        self.additional_formulas = {
            **{key: mod(formula_obj, gt_formula_obj) for key, mod in (
               additional_formula_mods or {}).items()},
            **self.additional_formulas}
        self.backup_formulas = {'formula': formula_obj,
                                **({'gt_formula': gt_formula_obj} if gt_formula_obj else {})}
        for form in self.additional_formulas.values():
            form.overwrite = True
        if verbose:
            print("Recalculating parts of the formula values.",
                "\nOriginal formula spec:", self.conf['formula_spec'],
                "\nOriginal formula:", formula_obj,
                "\nOriginal gt_formula:", gt_formula_obj,
                "\nAdditionally calculated formula outputs:", self.additional_formulas,
                "\nChanged constants for formula evaluation:", self.changed_constants)
        if recalc == 'formula':
            self.caches.update(final_cache_dir=None)
            self.additional_formulas = {**self.backup_formulas, **self.additional_formulas}
        elif recalc == True or recalc == "always":
            self.caches = dict(imgs_dir=self.caches['imgs_dir'])


    def from_cache(self, img_fn: str) -> Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, bool, float]]]:
        img_t, masks_t = load_orig_and_masks(
            img_fn=img_fn,
            **self.caches,
            device=self.device,
            raise_on_missing=self.raise_on_missing,
            orig_size=self.conf['img_size'],
            pedestrian_key=self.conf['pedestrian_key'], gt_pedestrian_key=self.conf['gt_pedestrian_key'],
            use_pretty_names=False,
        )
        # use correct concept names
        masks_t = {self.conf['concept_pretty_names'].get(k, k): v
                   for k, v in masks_t.items()}
        # filter non-existing files
        masks_t = {k: m for k, m in masks_t.items() if m is not None}
        return img_t, masks_t

    def recalc_predicate_masks(self, img_t: torch.Tensor):
        assert img_t is not None
        if len(img_t.size()) == 2:
            img_t.unsqueeze(0)
        mask_updates: Dict[str, torch.Tensor] = \
            calc_helpers.cached_eval(input_batch=img_t.unsqueeze(0), model=self.predicates,
                                     descriptors=None, cache=None, device=self.device)
        return {k: m.squeeze(0) for k, m in mask_updates.items()}

    def recalc_formula_masks(self, masks_t: Dict[str, torch.Tensor],
                             recalc_formulas: Dict[str, fuzzy_logic.Merge]
                             ) -> Dict[str, Union[torch.Tensor, float, bool]]:
        return recalc_formula_masks(
            self.conf, masks_t,
            additional_formulas=recalc_formulas,
            changed_constants=self.changed_constants,
            masks_dict_trafo=self.masks_dict_trafo,
            do_postfix=False, do_broadcast=False) or {}

    def get_all_masks(self, img_fn: str):
        try:
            img_t, masks_t = self.from_cache(img_fn)
            curr_recalc_formulas: Dict[str, trafos.fuzzy_logic.Merge] = {
                **{k: f for k, f in self.backup_formulas.items() if k not in masks_t},
                **self.additional_formulas}
            # recalc missing formula input masks
            if img_t is not None and \
                any(masks_t.get(k, None) is None for form_obj in curr_recalc_formulas.values()
                    for k in form_obj.all_in_keys):
                masks_t.update(self.recalc_predicate_masks(img_t))
            # recalc formula masks
            masks_t.update(self.recalc_formula_masks(masks_t, curr_recalc_formulas))
            if self.use_pretty_names:
                masks_t = {formula_to_display_name(key): val for key, val in masks_t.items()}
        except Exception as e:
            raise RuntimeError("Error during processing image file {} for experiment {}"
                               .format(img_fn, self.conf.get('sacred_logdir', None))) from e
        return img_t, masks_t


def to_logic_dirs(experiment_root: str, model_key: str, split: str,
                  formula_dirs: str = None, logic_type: str = None,
                  skip_missing: bool = True) -> List[str]:
    """Get the sub-directories for logic experiments based on given information.

    :param formula_dirs: list of (full) directories to include; by default excluded are ".*", "_*", "*results_cache"
    :param model_key: key and directory name of the model 
    :return: list of logic dirs (full paths)
    """
    logic_type = logic_type or ""
    exp_dir = os.path.join(experiment_root, model_key, split)
    if not os.path.exists(exp_dir) and skip_missing:
        logging.getLogger().warning("Skipping missing experiment dir {}".format(exp_dir))
        return []
    elif not os.path.exists(exp_dir):
        raise ValueError("Experiment dir {} does not exist".format(exp_dir))

    formula_dirs = [os.path.join(exp_dir, f_dir) for f_dir in formula_dirs] if formula_dirs is not None else \
        [f_dir for f_dir in [os.path.join(exp_dir, f) for f in os.listdir(exp_dir)]
         if os.path.isdir(f_dir)
         and not os.path.basename(f_dir).startswith(".")
         and not os.path.basename(f_dir).startswith("_")
         and not f_dir.endswith("results_cache")]
    for f_dir in formula_dirs:
        if not os.path.exists(f_dir) and skip_missing:
            logging.getLogger().warning("Skipping missing formula dir {}".format(f_dir))
        elif not os.path.exists(f_dir):
            raise ValueError("Formula dir {} does not exist".format(f_dir))
    logic_dirs_by_fdir = {f_dir: [os.path.join(f_dir, l) for l in os.listdir(
        f_dir)] for f_dir in formula_dirs if os.path.exists(f_dir)}
    logic_dirs = [ldir for ldirs in logic_dirs_by_fdir.values()
                  for ldir in ldirs if logic_type in ldir]
    return logic_dirs


# =====================
# EXPERIMENT LOADING
# =====================

def get_metrics(experiment_root: str, model_key: str, split: str, logic_type: str,
                formula_dirs: list = None, skip_missing: bool = True,
                ) -> List[Dict[str, Any]]:
    """Given a sacred ``experiment_root``, collect all experiment settings and results for the respective filters.
    Filters: model, formula(s), logic type.
    Experiment settings: formula formulation (``formula_dirs``), predicate settings, constant values.
    Metrics: all result metrics saved by the respective formula verification experiment.

    :param skip_missing: if unset, raise in case no metrics subdirectory can be found for a matching experiment dir
    :return: list of dictionaries, each entry corresponding to a single experiment (filter/settings & results);
        can directly be used to create a ``pd.DataFrame``
    """
    logic_dirs = to_logic_dirs(experiment_root=experiment_root, model_key=model_key, split=split,
                               formula_dirs=formula_dirs, logic_type=logic_type, skip_missing=skip_missing)
    # Collect infos & plot visualizations
    all_infos = []
    for logic_dir in logic_dirs:
        curr_infos = dict(logic_dir=logic_dir)

        # METRICS
        sacred_logdir = os.path.join(logic_dir, "logs")
        metrics_logdirs = [mdir for mdir in os.listdir(
            os.path.join(logic_dir, "metrics")) if mdir.startswith("20")]
        empty_metrics_logdirs = [mdir for mdir in metrics_logdirs if len(
            os.listdir(os.path.join(logic_dir, "metrics", mdir))) == 0]
        non_empty_metrics_logdirs = [mdir for mdir in metrics_logdirs if len(
            os.listdir(os.path.join(logic_dir, "metrics", mdir))) > 0]

        # Some sanity checks regarding folder structure
        if len(empty_metrics_logdirs) > 0:
            logging.getLogger().warning("Found %d empty metrics logdir entries (failed or running experiments?) in logdir %s:\n%s", len(empty_metrics_logdirs), os.path.join(logic_dir, "metrics"),
                                        '\n'.join([os.path.join(logic_dir, "metrics", d) for d in empty_metrics_logdirs]))
        if len(non_empty_metrics_logdirs) > 1:
            raise ValueError("Found metrics logdir with {} non-empty entries (several succeeded experiments?):\nlogdir: {}\nentries: {}".format(
                len(metrics_logdirs), os.path.join(logic_dir, "metrics"), metrics_logdirs))
        if len(non_empty_metrics_logdirs) < 1:
            if not skip_missing:
                raise ValueError(
                    "No non-empty metrics logdir found in {}".format(os.path.join(logic_dir, "metrics")))
            else:
                logging.getLogger().warning("Skipping metrics logdir (no non-empty subdirs found) %s",
                                            os.path.join(os.path.join(logic_dir, "metrics")))
                continue

        metrics_logdir = non_empty_metrics_logdirs[0]
        metrics_csv = os.path.join(logic_dir, "metrics", metrics_logdir, "metrics.csv")
        if not os.path.isfile(metrics_csv) and skip_missing:
            logging.getLogger().warning("Skipping missing file {}".format(metrics_csv))
            continue
        metrics = pd.read_csv(metrics_csv, index_col='index')['0']
        curr_infos.update(metrics)

        # FORMULA SPECS
        try:
            conf = get_exp_conf(sacred_logdir)
        except ValueError as e:
            if skip_missing:
                logging.getLogger().warning(
                    "Skipping logic dir %s due to error during configuration loading:\n%s", logic_dir, e)
                continue
            raise e
        formula_obj = calc_helpers.get_formula_obj(conf)
        formula: str = formula_to_display_name(
            formula_obj.to_str(), max_len=None)
        curr_infos['formula'] = formula
        gt_formula: str = ""
        if 'gt_formula_spec' in conf:
            gt_formula_obj = calc_helpers.get_formula_obj(conf, formula_spec=conf['gt_formula_spec'])
            gt_formula: str = formula_to_display_name(gt_formula_obj.to_str(), max_len=None)
        curr_infos['gt_formula'] = gt_formula

        # CONSTANTS
        curr_infos.update(conf['constants'])

        # PREDICATE SETTINGS
        predicate_setts = conf.get(
            'predicate_setts', {'ispartofa': conf.get('ispartofa_setts', {})})
        curr_infos.update({f'predicate_setts.{pred}.{k}': v for pred,
                          setts in predicate_setts.items() for k, v in setts.items()})

        all_infos.append(curr_infos)
    return all_infos


def gather_results(metrics_csv: str = None, **get_metrics_args) -> pd.DataFrame:
    """Gather and cache DataFrame containing experiment metric results.
    Convenience shortcut for calling :py:func:`get_metrics`, standard post-processing, and CSV caching.

    :param metrics_csv: where to load/save metric results from/to
    :param get_metrics_args: see :py:func:`get_metrics`
    :return: DataFrame with columns for all experiment settings and metrics
    """
    get_metrics_args_defaults: Dict[str, Any] = dict(
        model_key=["maskrcnn_box", "tf_efficientdet_d1"],
        split="TEST",
        logic_type=["boolean", "lukasiewicz", "product", "goedel"],)
    if metrics_csv is None or not os.path.isfile(metrics_csv):
        metrics = do_for(
            get_metrics,
            **{**get_metrics_args_defaults,
               **get_metrics_args}
        )
        metrics_flat = [
            {**setts, **metr_info}
            for setts, formula_metr in metrics
            for metr_info in formula_metr
        ]
        metrics_pd = add_cols_to_metrics_pd(pd.DataFrame(metrics_flat))
        # Filter out the Boolean dot at mask_thresh==0.5:
        metrics_pd = metrics_pd[~(~(metrics_pd.formula_attrs.str.contains(
            '(Boolean)')) & (metrics_pd.logic_type == 'boolean'))]
        # Cache results:
        if metrics_csv is not None:
            os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)
            metrics_pd.to_csv(metrics_csv)
    else:
        metrics_pd = pd.read_csv(metrics_csv)
    return metrics_pd


def gather_aucs(
    metrics_pd: pd.DataFrame,
    other_by: str = 'img_mon_thresh',
    formulas=('formula with S-implies, (Boolean)', 'formula with S-implies',
              'formula with S-implies, calibrated, (Boolean)', 'formula with S-implies, calibrated'),
    constraints: Dict[str, Any] = None,
    **aucs_for_kwargs,
) -> pd.DataFrame:
    """Gather area under curve values for several formulas.
    See :py:func:`aucs_for` for details.
    """
    constraints = constraints if constraints is not None else {
        'predicate_setts.AllNeighbors.kernel_size': 33,
        'predicate_setts.GTAllNeighbors.kernel_size': 33,
        f'gt_{other_by}': 0.5,
    }
    constrained_metrics_pd: pd.DataFrame = constrain_pd(metrics_pd, constraints)
    aucs = do_for(
        auc_for, allow_looping=('model_key', 'logic_type', 'formula'), verbose_looping=False,
        metrics_pd=constrained_metrics_pd,
        other_by=other_by,
        model_key=['maskrcnn_box', 'tf_efficientdet_d1'],
        logic_type=['product',
                    'goedel',
                    'lukasiewicz',
                    'boolean',
                    ],
        formula=formulas,
        **aucs_for_kwargs
    )
    aucs_flat = [{**setts, **auc} for setts, auc in aucs]
    aucs_pd = add_cols_to_metrics_pd(pd.DataFrame(aucs_flat))
    aucs_pd = (aucs_pd[aucs_pd.num_points > 0]
               .drop(columns=({*aucs_for_kwargs, 'other_by', 'other_at', 'implies', 'CloseBy', 'calibrated', 'denoised', 'Boolean', 'formula_attrs'}
                              .intersection(aucs_pd.columns)))
               .drop('metrics_pd', axis=1)
               .sort_values(['model_key', 'logic_type', 'formula']))
    return aucs_pd


# ================================
# EXPERIMENT POST-EVALUATION
# ================================

def recalc_formula_masks(conf, masks: Dict[str, torch.Tensor],
                         additional_formulas: Dict[str, fuzzy_logic.Merge],
                         changed_constants: Dict[str, Any] = None,
                         masks_dict_trafo: trafos.Transform = None,
                         do_postfix: bool = True, do_broadcast: bool = True) -> Optional[Dict[str, torch.Tensor]]:
    """Given a sacred experiment ``conf``ig, changes to constants, and predicate ``masks``,
    calculate the values of the given ``additional_formulas``.

    :param mask_dict_trafo: transformation applied to the dictionary of torch tensor input masks
        before feeding them to the ``additional_formulas``;
        defaults to the default one according to the experiment
    :param changed_constants: mapping of constant key to new value
    :return: dictionary of torch tensor masks from ``masks``, and the newly calculated ones
        using the keys from ``additional_formulas``
    """
    if not len(additional_formulas):
        return {}

    masks_dict_trafo = masks_dict_trafo or trafos.SameSizeTensorValues(
        mode=conf.get('predicate_setts', {}).get('_', {}).get('same_size_mode', None))
    changed_constants = changed_constants or {}
    masks = {key: mask if len(mask.size()) >= 3 else torch.broadcast_to(mask, [1, *conf['img_size']])
             for key, mask in masks.items() if mask is not None}
    postfix: str = '_lp' if do_postfix and conf.get('concept_model_args', {}).get(
        'use_laplace', False) else ''
    all_masks = {
        conf['pedestrian_key']: masks.get(formula_to_display_name(conf['pedestrian_key']), None),
        conf['gt_pedestrian_key']: masks.get(formula_to_display_name(conf['gt_pedestrian_key'], None)),
        **{k+postfix: m.to(torch.float) for k, m in masks.items() if isinstance(m, torch.Tensor)}}
    inp = {**masks_dict_trafo(all_masks),
           **conf['constants'],
           **changed_constants}

    calced_masks = {}
    for title, form in reversed(additional_formulas.items()):
        calced: torch.BoolTensor = form(inp)[form.out_key]
        if calced is None:  # Any input mask missing?
            continue
        calced = calced if len(calced.size()) >= 3 or not do_broadcast \
            else torch.broadcast_to(calced, [1, *conf['img_size']])
        calced_masks[title] = calced.to(torch.float)
    return {k: m for k, m in calced_masks.items()
            if k in masks.keys() or k in additional_formulas.keys() and isinstance(m, torch.Tensor)}


def gather_exp_stats(conf, iterator: Iterable[Tuple[str, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]],
                     thresh: float = 1e-3, save_as: str = None,
                     force_reevaluate: bool = False,
                     show_progress: bool = True,
                     ) -> pd.DataFrame:
    """Create a DataFrame summary of statistics of the masks created during a fuzzy logic experiment.
    The dataframe will feature

    - rows: indexed by image file names
    - columns: indexed by double index of ``(mask_type, value_type)``;
      ``mask_type``s: formula, ped, [gt, ] <concepts>;
      ``value_type``s: stddev, max, min, num_entries,
      num_interesting, mean_interesting, stddev_interesting,
      mean_formula_non_one, stddev_formula_non_one,
      mean_ped_non_zero, stddev_ped_non_zero,
      mean_concepts_non_zero, stddev_concepts_non_zero;
      ``*_interesting`` / ``*_ped_non_zero`` / ``*_concepts_non_zero``
      means values after application of masks where the
      formula mask is below ``1-thresh``/ person mask above ``thresh`` / all concept masks above ``thresh``.
    """
    data = {}
    if not force_reevaluate and save_as is not None and os.path.exists(save_as):
        df = pd.read_csv(save_as, header=[0, 1], index_col=0)
        return df
    if show_progress:
        iterator = tqdm(iterator, desc="Masks processed")
    for img_fn, (_, masks) in iterator:
        masks_trafo = trafos.Resize(img_size=conf['img_size'])

        formula = masks.pop('formula')
        ped = masks.pop(formula_to_display_name(conf['pedestrian_key']))
        gt = masks.pop(formula_to_display_name(
            conf['gt_pedestrian_key']), None) if 'gt_pedestrian_key' in conf else None
        concepts: Dict[str, torch.Tensor] = masks

        # binary masks for interesting values
        formula_non_one = formula < (1-thresh)
        ped_non_zero = ped > thresh
        concepts_non_zeros: Dict[str, torch.BoolTensor] = {
            c: tens > thresh for c, tens in concepts.items()}  # original size
        concepts_non_zero: torch.BoolTensor = torch.any(torch.stack([masks_trafo(c.float()) > 0.5
                                                                     for c in concepts_non_zeros.values()]), dim=0).bool()
        # Add ground-truth, false positive and false negative info if available:
        gt_masks = {}
        if gt is not None:
            gt_non_zero = (gt > thresh)
            fpos = torch.where(~gt_non_zero, ped, torch.tensor(0., dtype=ped.dtype, device=ped.device))    # ~gt & ped
            fpos_non_zero = fpos > thresh
            fneg = torch.where(gt_non_zero, 1.-ped, torch.tensor(0.,
                               dtype=ped.dtype, device=ped.device))  # gt & ~ped
            fneg_non_zero = fneg > thresh
            gt_masks = dict(gt=(gt, gt_non_zero), fpos=(
                fpos, fpos_non_zero), fneg=(fneg, fneg_non_zero))

        # actual data summary
        data[img_fn] = {}
        for label, (mask, mask_interesting) in dict(formula=(formula, formula_non_one),
                                                    ped=(ped, ped_non_zero),
                                                    **gt_masks,
                                                    **{c: (concepts[c], concepts_non_zeros[c]) for c in concepts},).items():
            data[img_fn].update({
                **dict(zip([(label, 'stddev'), (label, 'mean')], torch.std_mean(mask))),  # mean + stddev per image
                (label, 'max'): mask.max(),  # maximum pixel value
                (label, 'min'): mask.min(),  # minimum pixel value
                (label, 'num_entries'): mask.numel(),  # num entries in mask
                # num interesting entries in mask
                (label, 'num_interesting'): mask_interesting.sum(),
                **dict(zip([(label, 'stddev_interesting'), (label, 'mean_interesting')],
                           torch.std_mean(torch.masked_select(mask, mask_interesting)))),  # mean + stddev per image (only for interesting entries)
                **dict(zip([(label, 'stddev_formula_non_one'), (label, 'mean_formula_non_one')],
                           torch.std_mean(torch.masked_select(masks_trafo(mask), formula_non_one)))),  # mean + stddev per image (only non-1-entries of formula)
                **dict(zip([(label, 'stddev_ped_non_zero'), (label, 'mean_ped_non_zero')],
                           torch.std_mean(torch.masked_select(masks_trafo(mask), ped_non_zero)))),  # mean + stddev per image (only non-0-entries of pedestrian)
                **dict(zip([(label, 'stddev_concepts_non_zero'), (label, 'mean_concepts_non_zero')],
                           torch.std_mean(torch.masked_select(masks_trafo(mask), concepts_non_zero)))),  # mean + stddev per image (only non-0-entries of any concept mask)
            })

    df = pd.DataFrame({img_fn: {key: tens.item() if isinstance(tens, torch.Tensor) else tens
                                for key, tens in img_data.items()} for img_fn, img_data in data.items()}).T.rename_axis("filename")
    # Mean number of interesting pixels
    for key in df.columns.get_level_values(0).unique():
        df[(key, 'proportion_interesting')] = df.apply(
            lambda row: row[(key, 'num_interesting')] / row[(key, 'num_entries')], axis=1)
    df = df.sort_index(level=0, axis=1)

    if save_as:
        df.to_csv(save_as)
    return df


def summarize_exp_stats(df: pd.DataFrame, conf: dict, verbose: bool = False) -> pd.DataFrame:
    """Gather mean, std, min, max for the experiment stats DataFrame into
    a summary DataFrame (with these columns).

    :param df: DataFrame of the format provided by ``gather_exp_stats``
    :return: DataFrame with

        - column indices:
          level 1: ``proportion_interesting``, ``mean``, ``stddev``, ``mean_interesting``, ``stddev_interesting``;
          level 2: ``mean``, ``std``, ``max``, ``min``
        - row indices: mask types
    """
    cols = ['proportion_interesting', 'mean', 'stddev',
            'mean_interesting', 'stddev_interesting']

    # Mean and standard deviation per column:
    summaries = {}
    # .style.set_caption("Mean values per col.")
    summaries["mean"] = df.mean().unstack(level=1)[cols].T
    # .style.set_caption("Global standard deviation values per col."))
    summaries["std"] = df.std().unstack(level=1)[cols].T

    # Minimum and maximum per column:
    summaries["max"] = df.max().unstack(level=1)[cols].T
    summaries["min"] = df.min().unstack(level=1)[cols].T
    summaries_df = pd.concat(summaries, axis=1).swaplevel(
        0, 1, axis=1).sort_index(level=0, axis=1).T

    if verbose:
        display(summaries_df)
    return summaries_df


def auc_for(metrics_pd: pd.DataFrame,
            model_key: str, logic_type: str, formula: str, formula_name_col: str = 'formula_attrs',
            x: str = 'false_positive_rate', y: str = 'recall',
            other_metrics: Dict[str, str] = {'f1score': 'F1', 'precision': 'precision', 'recall': 'recall'},
            other_by: str = 'img_mon_thresh',
            other_at: Sequence[float] = (0.1, 0.5, 0.9),
            precision: int = 3) -> Dict[str, float]:
    """Collect area under curve of x-y-plots for the given experiment series.
    The output dictionary may contain further values at specific points on the curve.
    Precisely, values of the ``other_metrics`` are collected at the values ``other_at`` of the setting ``other_by``
    (by default: F1 score, precision and recall at values 0.1, 0.5 and 0.9 of the threshold ``img_mon_thresh``).

    :param metrics_pd: DataFrame with metric results (see :py:func:`get_metrics`)
    :param model_key: the model key and directory name
    :param logic_type: the logic type
    :param formula: the formula specifier
    :param x: column with x-values in the x-y-plot
    :param y: column with y-values in the x-y-plot
    :param other_metrics: dictionary where keys are column names of other metrics to sample points values from,
        dict values are pretty names thereof to use as keys in the output dict
    :param formula_name_col: column with values to match with ``formula``
    :param precision: number display precision used to create keys for the ``other_metrics`` values in the output dict
    :return: dictionary of the form ``{'auc': float, 'num_points': int, '<other_metric>@<other_by_value>': float}``
    """

    constraints: Dict[str, Any] = {'model_key': model_key, 'logic_type': logic_type, formula_name_col: formula}
    constrained_metrics_pd: pd.DataFrame = constrain_pd(metrics_pd, constraints)
    xy: Dict[str, List[float]] = constrained_metrics_pd[[x, y]]\
        .sort_values([x])\
        .astype({x: float, y: float})\
        .to_dict(orient='list')
    other_vals = {}
    for m, m_pretty in other_metrics.items():
        for t in other_at:
            curr = constrained_metrics_pd[constrained_metrics_pd[other_by].astype(
                float) == t]
            if len(curr.index) == 1:
                col = ('{m_pretty}@{t:.'+str(precision) + 'f}').format(m_pretty=m_pretty, t=t)
                other_vals[col] = curr[m].astype(float).item()
                if pd.isna(other_vals[col]):
                    logging.getLogger().warning(
                        f'WARNING: NaN metrics value for: {curr.experiment_root.item()=} {model_key=} {logic_type=} {formula=} {t=} {m=} {other_vals[col]=}')  # {curr.to_dict()=}
    return {'auc': auc(xy[x], xy[y]), 'num_points': len(xy[x]),
            # 'max_f1': constrained_metrics_pd['f1score'].astype(float).max(),
            # 'max_f1_thresh': constrained_metrics_pd.loc[constrained_metrics_pd['f1score'].astype(float).idxmax(), other_by],
            **other_vals,
            }


def to_formula_attrs(formula: str) -> str:
    """Extract key properties from the formula string.
    Format: {
    'implies': Literal['R','S'],
    'CloseBy': Literal['CoveredBy', 'downscaled'],
    'calibrated': bool, 'denoised': bool, 'Boolean': bool
    }"""
    attrs = {}
    attrs['implies'] = 'R' if '>>' in formula or '<<' in formula else 'S'
    if 'IsPartOfA' in formula:
        attrs['CloseBy'] = 'CoveredBy'
    elif '_down' in formula:
        attrs['CloseBy'] = 'downscaled'
    else:
        attrs['CloseBy'] = 'none'
    attrs['calibrated'] = ('_lp' in formula)
    attrs['denoised'] = ('((ankle_lp||arm_lp||eye_lp||leg_lp||wrist_lp)>=mask_thresh)&&(ankle_lp||arm_lp||eye_lp||leg_lp||wrist_lp)' in formula
                         or '((ankle||arm||eye||leg||wrist)>=mask_thresh)&&(ankle||arm||eye||leg||wrist))' in formula)
    attrs['Boolean'] = ('bool_thresh' in formula)
    return attrs


def to_formula_attrs_str(formula: str) -> str:
    """Turn formula in nice short string listing its attributes."""
    attrs: Dict[str, Any] = to_formula_attrs(formula)
    attr = []
    attr.append(f'{attrs["implies"]}-implies')
    attr.append(attrs["CloseBy"]) if attrs['CloseBy'] != 'none' else None
    attr.append('calibrated') if attrs['calibrated'] else None
    attr.append('denoised') if attrs['denoised'] else None
    attr.append('(Boolean)') if attrs['Boolean'] else None
    return 'formula' + (f' with {", ".join(attr)}' if len(attr) else '')


def add_cols_to_metrics_pd(metrics_pd: pd.DataFrame):
    """Add some standard aliases and derived values to outputs of ``get_metrics``."""
    def cols(): return metrics_pd.columns
    
    # Fix types:
    types = {
        **{col: float for col in [
            # 'all_min', 'exists_max', # bools that got parsed to string :-/
            'all_mean', 'all_mean_gt', 'all_binary_mean@050',
            'std_dev', 'accuracy', 'f1score', 'precision',
            'negpredictiveval', 'recall',
            *[c for c in metrics_pd.columns if 'thresh' in c or 'rate' in c]
       ]}, **{col: int for col in [
           *[c for c in metrics_pd.columns if 'kernel_size' in c]
       ]},
    }
    metrics_pd = metrics_pd.astype({col: dtype for col, dtype in types.items()
                                    if col in metrics_pd.columns})

    # TPR
    if 'true_positive_rate' not in cols() and 'recall' in cols():
        metrics_pd.loc[:, 'true_positive_rate'] = metrics_pd['recall'].astype(float)
    # TNR
    if 'true_negative_rate' not in cols() and 'specificity' in cols():
        metrics_pd.loc[:, 'true_negative_rate'] = metrics_pd['specificity'].astype(float)
    elif 'true_negative_rate' in cols() and 'specificity' in cols():
        metrics_pd.loc[:, 'true_negative_rate'] = metrics_pd['true_negative_rate'].fillna(
            metrics_pd['specificity'].astype(float))
    # FPR
    if 'false_positive_rate' not in cols() and 'true_negative_rate' in cols():
        metrics_pd.loc[:, 'false_positive_rate'] = 1 - \
            metrics_pd['true_negative_rate'].astype(float)
    elif 'false_positive_rate' in cols() and 'true_negative_rate' in cols():
        metrics_pd.loc[:, 'false_positive_rate'] = metrics_pd['false_positive_rate'].fillna(
            1 - metrics_pd['true_negative_rate'].astype(float))
    # monitor threshold
    if 'mon_thresh' not in cols() and 'monitor_thresh' in cols():
        metrics_pd.loc[:, 'mon_thresh'] = 1 - \
            metrics_pd['monitor_thresh'].astype(float)
    elif 'mon_thresh' in cols() and 'monitor_thresh' in cols():
        metrics_pd.loc[:, 'mon_thresh'] = metrics_pd.loc[:, 'mon_thresh'].fillna(
            1 - metrics_pd['monitor_thresh'].astype(float))
    # same_size_mode
    if 'same_size_mode' in cols():
        metrics_pd.loc[:, 'same_size_mode'] = metrics_pd['same_size_mode'].fillna('up_bilinear')
    # nice formula names
    additional_formula_names: pd.DataFrame = pd.DataFrame(
        metrics_pd['formula'].apply(lambda f: to_formula_attrs(f)).to_dict()).T
    metrics_pd = metrics_pd.join(additional_formula_names[[
        c for c in additional_formula_names.columns if c not in metrics_pd.columns]])
    if 'formula_attrs' not in cols() and 'formula' in cols():
        metrics_pd.loc[:, 'formula_attrs'] = metrics_pd['formula'].apply(
            lambda f: to_formula_attrs_str(f))
    if 'gt_formula_attrs' not in cols() and 'gt_formula' in cols():
        metrics_pd.loc[:, 'gt_formula_attrs'] = metrics_pd['gt_formula'].apply(
            lambda f: to_formula_attrs_str(f))
    return metrics_pd
