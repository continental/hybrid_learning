"""Helper functions for evaluation of standard concept embedding analysis experiments.
Experiment results are assumed to have a folder structure of
``<root>/layers/<layer_id>/<timestamp>/<concept_name>``.
For details see the respective sample Sacred experiment scripts.
"""
#  Copyright (c) 2022 Continental Automotive GmbH

import os
from typing import Callable, List, Sequence, Optional, Dict, Union, Mapping, \
    Iterable, Tuple

import PIL.Image
import pandas as pd
import torchvision as tv
from matplotlib import pyplot as plt
from pandas.io.formats.style import Styler

# pylint: disable=import-error,wrong-import-position
from hybrid_learning.concepts.analysis import AnalysisResult, \
    BestEmbeddingResult
from hybrid_learning.concepts.models import ConceptEmbedding

to_img: Callable = tv.transforms.ToPILImage()
to_tens: Callable = tv.transforms.ToTensor()

# ======================================
# DEFAULT MODEL LAYERS
# ======================================

# ALexNet
ALEXNET_LAYERS: Tuple[str, ...] = (
    'features.1', 'features.2', 'features.4', 'features.5', 'features.7',
    'features.9', 'features.11', 'features.12', 'avgpool')
"""Layer IDs of pytorch AlexNet model in correct order."""

# VGG16
VGG16_LAYERS: Tuple[str, ...] = (
    'features.4', 'features.6', 'features.8', 'features.9', 'features.11',
    'features.13', 'features.15', 'features.16', 'features.18', 'features.20',
    'features.22', 'features.23', 'features.25', 'features.27', 'features.29',
    'features.30', 'avgpool')
"""Layer IDs of pytorch VGG16 model in correct order."""

# Mask R-CNN
MASK_RCNN_LAYERS: Tuple[str, ...] = (
    'backbone.body.relu', 'backbone.body.maxpool', 'backbone.body.layer1',
    'backbone.body.layer1.0.bn1', 'backbone.body.layer1.0.bn2',
    'backbone.body.layer1.0.relu', 'backbone.body.layer1.0',
    'backbone.body.layer1.1.bn1', 'backbone.body.layer1.1.bn2',
    'backbone.body.layer1.1.relu', 'backbone.body.layer1.1',
    'backbone.body.layer1.2.bn1', 'backbone.body.layer1.2.bn2',
    'backbone.body.layer1.2.relu', 'backbone.body.layer1.2',
    'backbone.body.layer2', 'backbone.body.layer2.0.bn1',
    'backbone.body.layer2.0.bn2', 'backbone.body.layer2.0.relu',
    'backbone.body.layer2.0', 'backbone.body.layer2.1.bn1',
    'backbone.body.layer2.1.bn2', 'backbone.body.layer2.1.relu',
    'backbone.body.layer2.1', 'backbone.body.layer2.2.bn1',
    'backbone.body.layer2.2.bn2', 'backbone.body.layer2.2.relu',
    'backbone.body.layer2.2', 'backbone.body.layer2.3.bn1',
    'backbone.body.layer2.3.bn2', 'backbone.body.layer2.3.relu',
    'backbone.body.layer2.3', 'backbone.body.layer3',
    'backbone.body.layer3.0.bn1', 'backbone.body.layer3.0.bn2',
    'backbone.body.layer3.0.relu', 'backbone.body.layer3.0',
    'backbone.body.layer3.1.bn1', 'backbone.body.layer3.1.bn2',
    'backbone.body.layer3.1.relu', 'backbone.body.layer3.1',
    'backbone.body.layer3.2.bn1', 'backbone.body.layer3.2.bn2',
    'backbone.body.layer3.2.relu', 'backbone.body.layer3.2',
    'backbone.body.layer3.3.bn1', 'backbone.body.layer3.3.bn2',
    'backbone.body.layer3.3.relu', 'backbone.body.layer3.3',
    'backbone.body.layer3.4.bn1', 'backbone.body.layer3.4.bn2',
    'backbone.body.layer3.4.relu', 'backbone.body.layer3.4',
    'backbone.body.layer3.5.bn1', 'backbone.body.layer3.5.bn2',
    'backbone.body.layer3.5.relu', 'backbone.body.layer3.5',
    'backbone.body.layer4', 'backbone.body.layer4.0.bn1',
    'backbone.body.layer4.0.bn2', 'backbone.body.layer4.0.relu',
    'backbone.body.layer4.0', 'backbone.body.layer4.1.bn1',
    'backbone.body.layer4.1.bn2', 'backbone.body.layer4.1.relu',
    'backbone.body.layer4.1', 'backbone.body.layer4.2.bn1',
    'backbone.body.layer4.2.bn2', 'backbone.body.layer4.2.relu',
    'backbone.body.layer4.2', 'backbone.body.layer1.0.downsample',
    'backbone.body.layer2.0.downsample', 'backbone.body.layer3.0.downsample',
    'backbone.body.layer4.0.downsample', 'backbone.fpn',
    'backbone.fpn.inner_blocks.0', 'backbone.fpn.inner_blocks.1',
    'backbone.fpn.inner_blocks.2', 'backbone.fpn.inner_blocks.3',
    'backbone.fpn.layer_blocks.0', 'backbone.fpn.layer_blocks.1',
    'backbone.fpn.layer_blocks.2', 'backbone.fpn.layer_blocks.3',
    'backbone.fpn.extra_blocks')
"""Layer IDs of pytorch Mask R-CNN model in correct order."""


# ======================================
# FILEPATHS
# ======================================


def _experiments(layer_id: str, root: str) -> List[str]:
    """List of experiment folders for given layer and root.
    Ones starting with underscore are excluded."""
    layer_root = os.path.join(root, "layers", layer_id)
    return [os.path.join(layer_root, exp) for exp in os.listdir(layer_root)
            if not exp.startswith('_')]


def analysis_root(layer_id: str, concept_name: str, root: str) -> str:
    """Get the analysis root folder path for given
    ``root``, ``layer_id``, and ``concept_name``."""
    exps: List[str] = [exp for exp in _experiments(layer_id=layer_id, root=root)
                       if os.path.isdir(os.path.join(exp, concept_name))]
    if len(exps) == 0:
        raise ValueError(
            "Concept results not found for concept {} and layer {} in root {}"
                .format(concept_name, layer_id, root))
    if len(exps) > 1:
        raise ValueError(
            "Ambiguous results for concept {}: Found experiments {}"
                .format(concept_name,
                        [os.path.join(exp, concept_name) for exp in exps]))
    concept_root = os.path.join(exps[0], concept_name)
    return concept_root


def get_layers(root: str, model_layers: List[str] = None,
               verbose: bool = False) -> List[str]:
    """Return all available layers for the given analysis root."""
    available_layers = sorted(os.listdir(os.path.join(root, "layers")))
    model_layers = model_layers or available_layers
    layers = [l for l in model_layers if l in available_layers]
    if verbose and len(layers) != len(model_layers):
        print("WARNING: Skipping non-available layers {}".format(
            [l for l in model_layers if l not in available_layers]))
    return layers


def get_concepts(root: str, layers: List[str] = None) -> List[str]:
    """Get all the concept names available for the given experiment root."""
    layers = layers or get_layers(root)
    return sorted({c for layer in layers
                   for exp in _experiments(layer, root=root)
                   for c in os.listdir(exp)})


def get_common_concepts(root: str, layers: List[str] = None) -> List[str]:
    """Get all the concept names available for ALL layers in the
    given experiment ``root``."""
    layers = layers or get_layers(root)
    return sorted(set.intersection(*[
        set(concept for exp in _experiments(layer, root=root)
            for concept in os.listdir(exp))
        for layer in layers]))


# ======================================
# RESULTS RETRIEVAL
# ======================================


def get_stats(root: str) -> pd.DataFrame:
    """Stats of an analysis run as DataFrame."""
    return AnalysisResult.load(root).to_pandas()


def get_best_emb_stats(root: str) -> pd.DataFrame:
    """Stats of the best embeddings of an analysis as DataFrame."""
    return BestEmbeddingResult.load(root).to_pandas()


def get_embs(root: str) -> pd.Series:
    """Embeddings of an analysis as series indexed by ``layer_id``.
    Each embedding is encapsulated in a 1-element list to keep pandas
    working."""
    analysis_results = AnalysisResult.load(root).results
    return pd.Series({(layer, run): [emb]
                      for layer, layer_stats in analysis_results.items()
                      for run, (emb, s) in layer_stats.items()}).transpose()


def get_best_emb(layer_id: str, root: str) -> ConceptEmbedding:
    """Best embedding for given layer."""
    best_emb_result = BestEmbeddingResult.load(root)
    return best_emb_result.results[layer_id][0]


def get_vis_best_embedding(root: str) -> PIL.Image.Image:
    """Get the saved visualization of the best embedding."""
    return PIL.Image.open(os.path.join(root, "vis_best_embedding.png"))


def get_all_best_emb_stats(root: str, layers: List[str],
                           concepts: List[str] = None) -> pd.DataFrame:
    """Gather a DataFrame with the stats of all best embeddings.
    The frame is indexed by ``concept`` and ``layer``, and columns are the
    stats names.
    Stats are retrieved from experiment ``root``, for given ``layers`` and
    ``concepts``."""
    concepts = concepts or get_concepts(root, layers=layers)
    best_emb_stats = pd.DataFrame(
        {(concept, layer): get_best_emb_stats(
            analysis_root(layer, concept, root=root)).loc[layer]
         for layer in layers for concept in concepts
         if os.path.isdir(analysis_root(layer, concept, root=root))}
    ).transpose().infer_objects()
    best_emb_stats.index.names = ['concept', 'layer']
    best_emb_stats.reset_index(inplace=True)
    return best_emb_stats


def get_all_stats(root: str, layers: List[str],
                  concepts: List[str] = None) -> pd.DataFrame:
    """Gather a DataFrame with the stats of all embedding runs.
    Same as ``get_all_best_emb_stats``, only indexed by
    ``concept``, ``layer``, *and* ``run``."""
    concepts = concepts or get_concepts(root, layers=layers)
    stats_dict = {}
    for layer in layers:
        for concept in [c for c in concepts if
                        os.path.isdir(analysis_root(layer, c, root=root))]:
            results = get_stats(analysis_root(layer, concept, root=root))
            stats_dict.update(
                {(concept, layer, run): results.loc[(layer, run)] for run in
                 results.index.get_level_values(1).unique()})
    stats = pd.DataFrame(stats_dict).transpose().infer_objects()
    stats.index.names = ['concept', 'layer', 'run']
    stats.reset_index(inplace=True)
    return stats


def gather_stats(roots: Union[List[str], Dict[str, str]] = None,
                 root_templ: str = None,
                 root_params: Iterable[Union[str, Sequence[str]]] = None,
                 model_layers: List[str] = None,
                 metric: str = 'set_iou'
                 ):
    """Gather dicts with all_stats, best_emb_stats, and merged_stats for
    different roots."""
    # Prepare roots:
    if roots is not None:
        if not isinstance(roots, Mapping):
            roots: Dict[str, str] = {
                os.path.basename(os.path.dirname(root)): root
                for root in roots}
    elif root_templ is None or root_params is None:
        raise ValueError("Either root, or root_templ and root_params must be "
                         "given")
    else:
        root_params = [[p] if isinstance(p, str) else p for p in root_params]
        roots = {"_".join(param): root_templ.format(*param)
                 for param in root_params}
    for root in roots.values():
        assert os.path.isdir(root), root

    all_stats, best_stats, merged_stats = dict(), dict(), dict()
    for graph_name, root in roots.items():
        layers = get_layers(root, model_layers=model_layers)
        all_stats[graph_name] = get_all_stats(root=root, layers=layers)
        best_stats[graph_name] = get_all_best_emb_stats(root=root,
                                                        layers=layers)
        merged_stats[graph_name] = merge_to_overview(best_stats[graph_name],
                                                     all_stats[graph_name],
                                                     metric=metric)

    return all_stats, best_stats, merged_stats


# ======================================
# PANDAS DISPLAY
# ======================================

def highlight_max_blue(series: pd.Series) -> List[str]:
    """Highlight the maximum in a Series with red font color."""
    return ['color: blue; font-weight: bold' if v else ''
            for v in series == series.max()]


def merge_to_overview(best_ious: pd.DataFrame, all_stats: pd.DataFrame,
                      metric: str = 'set_iou', layers: List[str] = None):
    """Merge the stats from all runs and from the best embeddings
    to an overview over the test values of ``metric``.
    The resulting DataFrame is indexed by the ``layer_id``, and columns are a
    multi-index of ``(concept_name, stats_name)``, with ``stats_name`` one of

    - ``best_emb``: the best embedding performance
    - ``mean``: the mean performance of the runs for that concept & layer
    - ``std``: the corresponding standard deviation

    :param all_stats: should an output of ``get_all_stats``
    :param best_ious: should an output of ``get_all_best_emb_stats``
    :param metric: the (column) name of the metric to use
    :param layers: optionally restrict to given layers
    """
    if all_stats.index.names != ['layer', 'concept']:
        all_stats = all_stats.set_index(['layer', 'concept'])
    if best_ious.index.names != ['layer', 'concept']:
        best_ious = best_ious.set_index(['layer', 'concept'])
    layers = layers or all_stats.index.get_level_values('layer').unique()
    best_set_iou_display = pd.concat({
        'best_emb': best_ious[['test_' + metric]],
        'mean': all_stats[['test_' + metric]].mean(level=[0, 1]),
        'std': all_stats[['test_' + metric]].std(level=[0, 1]),
    }, axis=1) \
        .T.reset_index(level=1, drop=True).T \
        .unstack(level=1) \
        .swaplevel(0, 1, 1).sort_index(1) \
        .loc[layers]
    return best_set_iou_display


def display_overview(best_set_iou_display: pd.DataFrame) -> Styler:
    """Provide a nice display style of ``best_set_iou_display``.
    E.g. highlight maxima, apply color gradients, and take care of precision.

    :param best_set_iou_display:
    """
    best_emb_cols = best_set_iou_display.loc[:,
                    (slice(None), 'best_emb')].columns.unique()
    std_cols = best_set_iou_display.loc[:, (slice(None), 'std')] \
        .columns.unique()
    mean_cols = best_set_iou_display.loc[:, (slice(None), 'mean')] \
        .columns.unique()
    return best_set_iou_display.style \
        .set_caption("Set IoU values by layer and concept (both the set IoU of "
                     "the mean embedding and the mean set IoU with standard "
                     "deviation)") \
        .set_precision(4) \
        .background_gradient(cmap='Greens', axis=0, subset=best_emb_cols,
                             vmin=0, vmax=0.5) \
        .apply(highlight_max_blue,
               subset=list(best_emb_cols) + list(mean_cols)) \
        .format({col: "±{:.4f}" for col in std_cols})


# ======================================
# PLOTTING
# ======================================


def plot_best_ious_wt_std(best_emb_ious: pd.DataFrame,
                          iou_stds: pd.DataFrame, save_as: str = None,
                          max_val: float = None, metric: str = None,
                          **plot_args):
    """Provide a plot of metric values against layers for all concepts in
    ``best_emb_ious``.
    Plotted points are the best embedding performances, the plotted standard
    deviation is that of the runs. Optionally save the figure."""
    plot_args = {**dict(capsize=3, rot=90,
                        xticks=range(len(best_emb_ious.index))), **plot_args}
    layers: Sequence[str] = best_emb_ious.index
    max_val = max_val if max_val is not None else \
        best_emb_ious.max().max() + 0.01
    plt.figure(figsize=(len(layers) * 0.5, max_val * 8))
    plt.ylim(top=max_val)
    plt.title("Embedding performance")
    axis: Optional[plt.Axes] = None
    for concept in best_emb_ious.columns.get_level_values(0).unique():
        axis = best_emb_ious[(concept, 'best_emb')] \
            .plot(yerr=iou_stds[concept, 'std'], **plot_args)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    axis.set_xticklabels(layers)
    if metric is not None:
        axis.set_ylabel(metric.replace("_", " "))
    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight')


def plot_best_iou_comparison(metric_mean_map: Dict[str, pd.DataFrame],
                             metric_std_map: Dict[str, pd.DataFrame],
                             concept: str = None,
                             max_val: float = None,
                             save_as: str = None,
                             metric: str = 'set_iou',
                             axis: plt.Axes = None,
                             set_title: bool = True,
                             **plot_args):
    """Provide a plot of metric values against layers for one concept and all
    settings in ``best_emb_ious``.
    Plotted points are the best embedding performances, the plotted standard
    deviation is that of the runs. Optionally save the figure."""
    layers: Sequence[str] = list(metric_mean_map.values())[0].index
    concept: str = concept or list(metric_mean_map.values())[0] \
        .columns.get_level_values(0).unique()[0]
    max_val: float = max(max_val or 0,
                         max(ious.max().max() + 0.01 for ious in
                             metric_mean_map.values()))
    fig = None
    if axis is None:
        fig = plt.figure(figsize=(len(layers) * 0.5, max_val * 12))
        if set_title:
            plt.title(concept)
        plt.ylim(top=max_val)
    else:
        if set_title:
            axis.set_title(concept)
        axis.set_ylim(top=max_val)
    for graph_name, best_emb_ious in metric_mean_map.items():
        values: pd.Series = pd.Series(best_emb_ious[(concept, 'best_emb')],
                                      name=graph_name)
        # Make sure layers are all aligned!!
        values.sort_index(key=lambda idx: [list(layers).index(i) for i in idx],
                          inplace=True)
        values.name = " ".join(str(values.name).split("_"))
        plot_args = {**dict(rot=90, capsize=3, xticks=range(len(values.index))),
                     **plot_args}
        axis = values.plot(yerr=metric_std_map[graph_name][concept, 'std'],
                           ax=axis, **plot_args)
    axis.set_xticklabels(layers)
    axis.set_ylabel(metric.replace("_", " "))
    if fig is not None:
        axis.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight')


def plot_overview(best_set_iou_display: pd.DataFrame,
                  model_name: str = None, metric: str = 'set_iou',
                  max_val: float = None, save_as: str = None,
                  **plot_args):
    """Plot and save the output of ``merge_to_overview`` using
    ``plot_best_ious_wt_std``.

    :param best_set_iou_display: DataFrame in the format as provided by
        ``merge_to_overview``
    :param model_name: used to determine default for save_as
    :param metric: used to determine default for save_as
    :param max_val: see ``plot_best_ious_wt_std``
    :param save_as: the filename to save plot as; no saving if set to ``False``;
        if a directory or None, a default file name is used (under directory);
        if ends with ``.svg``, both an SVG and a PNG image are saved
    """
    # Default save position
    root = None
    if save_as and os.path.isdir(save_as):
        root, save_as = save_as, None
    model_name = None if root is None else model_name or os.path.basename(root)
    if save_as is None:
        save_as: Optional[str] = None if root is None else \
            os.path.join(root, f"{model_name}_{metric}.svg")
    save_as = save_as or None

    plot_best_ious_wt_std(
        best_emb_ious=best_set_iou_display.loc[:, (slice(None), 'best_emb')],
        iou_stds=best_set_iou_display.loc[:, (slice(None), 'std')],
        save_as=save_as, max_val=max_val, metric=metric, **plot_args
    )
    plt.close()
    plot_best_ious_wt_std(
        best_emb_ious=best_set_iou_display.loc[:, (slice(None), 'best_emb')],
        iou_stds=best_set_iou_display.loc[:, (slice(None), 'std')],
        save_as=save_as.replace(".svg", ".png") if save_as else None,
        max_val=max_val, metric=metric, **plot_args
    )
    plt.show()


def plot_overview_for_concept(merged_stats_map: Dict[str, pd.DataFrame],
                              concepts: Union[str, Iterable[str]],
                              max_val: float = None,
                              save_as: str = None, model_name: str = None,
                              metric: str = 'set_iou',
                              one_plot: bool = True, legend_in: int = -1,
                              fig_size: Tuple[float, float] = None,
                              axes: Optional[Sequence[plt.Axes]] = None,
                              **plot_args):  # pylint: disable=too-many-branches
    """Plot and save the output of ``merge_to_overview`` using
    ``plot_best_ious_wt_std``.

    :param merged_stats_map: dict with values the output of merge_to_overview;
        each value will get its own graph
    :param concepts: the concept or concepts to plot
    :param max_val: the common x-axis limit for all created plots;
        defaults to the maximum metric value for a concept
    :param save_as: an optional save location;
        must contain ``{concept}`` if ``one_plot`` is not ``True``;
        if a directory is given, the file name is auto-inferred from the
        other arguments
    :param model_name: for file name auto-inference
    :param metric: for axis label and file name auto-inference
    :param one_plot: whether to put all into one common plot
        (save_as must not contain formatting string for concept)
    :param legend_in: if ``one_plot``, index of the axis the legend should be
        placed in; if set to None, legend is placed outside of plot
    :param axes: plot into given axes
    :param fig_size: if ``one_plot``, used ``fig_size``
    """
    assert axes is None or save_as is None
    # Collect data
    metric_mean_map, metric_std_map = dict(), dict()
    for graph_name, merged_stats in merged_stats_map.items():
        metric_mean_map[graph_name] = \
            merged_stats.loc[:, (slice(None), 'best_emb')]
        metric_std_map[graph_name] = \
            merged_stats.loc[:, (slice(None), 'std')]

    max_val_by_concept: Dict[str, float] = {
        concept: max(max_val or 0,
                     max(vals.loc[:, (concept, 'best_emb')].max().max() + 0.01
                         for vals in metric_mean_map.values()))
        for concept in concepts}

    if save_as is not None and os.path.isdir(save_as):
        save_as: str = os.path.join(
            save_as, f"{model_name or ''}_{metric or ''}"
                     f"{'' if one_plot else '_{concept}'}.svg")

    if one_plot and axes is None:
        layers: Sequence[str] = list(metric_mean_map.values())[0].index
        max_val: float = max(max_val_by_concept.values())
        max_val_by_concept: Dict[str, float] = {c: max_val for c in concepts}
        _, axes = plt.subplots(
            1, len(concepts), sharey='all',
            figsize=fig_size or
                    (len(layers) * len(concepts) * 0.7, max_val * 10))
        if isinstance(axes, plt.Axes):
            axes: Sequence[plt.Axes] = [axes]

    for i, concept in enumerate(concepts):
        assert all(concept in means.columns.get_level_values('concept')
                   for means in metric_mean_map.values())
        assert all(concept in stds.columns.get_level_values('concept')
                   for stds in metric_std_map.values())

        max_val: float = max_val_by_concept[concept]
        curr_save_as: Optional[str] = save_as.format(concept=concept) \
            if save_as is not None and not one_plot else None
        axis: Optional[plt.Axes] = axes[i] if axes is not None else None
        if curr_save_as is not None and curr_save_as.endswith('.svg'):
            plot_best_iou_comparison(
                metric_mean_map=metric_mean_map,
                metric_std_map=metric_std_map,
                concept=concept, max_val=max_val, metric=metric,
                save_as=(curr_save_as.replace('.svg', '.png') if curr_save_as
                         else None),
                axis=axis, **plot_args
            )
        if not axis:
            plt.close()
        plot_best_iou_comparison(
            metric_mean_map=metric_mean_map,
            metric_std_map=metric_std_map,
            concept=concept, max_val=max_val, metric=metric,
            save_as=curr_save_as,
            axis=axis, **plot_args
        )
        if not axis:
            plt.show()
    if one_plot:
        # handles, labels = axes[-1].get_legend_handles_labels()
        # fig.legend(handles, labels, loc='center left', bbox_to_anchor=(.9,0.5))
        if legend_in is not None:
            axes[legend_in].legend()
        else:
            axes[-1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        if save_as is not None:
            if save_as.endswith('.svg'):
                plt.savefig(save_as.replace('.svg', '.png'),
                            bbox_inches='tight')
            plt.savefig(save_as, bbox_inches='tight')
