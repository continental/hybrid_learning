#  Copyright (c) 2022 Continental Automotive GmbH
"""Helper functions for visualization of fuzzy logic experiment results."""

import logging
import os
from typing import Sequence, Literal, List, Dict, Tuple, Callable, Optional, Union, Set, Iterable, Any

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import functional as F

from hybrid_learning.datasets import transforms as trafos, data_visualization
from hybrid_learning import fuzzy_logic
from hybrid_learning.datasets.transforms.common import lazy_format
from hybrid_learning.experimentation.exp_eval_common import show_and_save_plot, constrain_pd
from hybrid_learning.experimentation.fuzzy_exp import fuzzy_exp_helpers as calc_helpers
from hybrid_learning.experimentation.fuzzy_exp.fuzzy_exp_eval import to_logic_dirs, get_exp_conf, ResultsIterator, \
    gather_exp_stats, summarize_exp_stats, formula_to_display_name, to_tens, display

to_img = F.to_pil_image


def summarize_plot_visualize_exp(
    experiment_root: str, model_key: str, split: str, logic_type: str, formula_dirs: list = None,
    # file path relative to logic_dir/metrics
    per_image_stats_csv: str = "per_image_stats.csv",
    force_reevaluate: bool = True,
    visualize: Sequence[Literal['best', 'worst']] = None,  # ('worst', 'best'),
    num_interesting_samples: int = 15,
    verbose: bool = False,
    only_filenames: List[str] = None,
):
    """Concatenation of per image stats gathering, stats summary, and optional visualization of formula body masks.
    Summary of
    :py:func:`hybrid_learning.experimentation.fuzzy_exp.fuzzy_exp_eval.gather_exp_stats`,
    :py:func:`hybrid_learning.experimentation.fuzzy_exp.fuzzy_exp_eval.summarize_exp_stats`,
    :py:func:`plot_summaries`, and
    :py:func:`visualize_most_interesting_samples` with option to loop over several formulas.
    Formula body values are reevaluated.

    :param per_image_stats_csv: file path to a CSV file for caching the per image statistics
        (will contain information for one image per row);
        file path relative to the respective ``logic_type/metrics`` results directory
    :param visualize: list of different arguments for key ``best`` in :py:func:`visualize_most_interesting_samples`;
        set to ``None`` or an empty sequence to disable visualization of example images
    :param num_interesting_samples: see :py:func:`visualize_most_interesting_samples`
    :param force_reevaluate: see :py:func:`hybrid_learning.experimentation.fuzzy_exp.fuzzy_exp_eval.gather_exp_stats`
    :param only_filenames: only take into account the files with given names in analysis and visualization
    """
    iterator_args = dict(recalc=True,
                         # apply analytics to formula body
                         additional_formula_mods={
                             'formula': lambda f, _: f.in_keys[0].in_keys[0]},
                         verbose=verbose)
    if only_filenames:
        iterator_args.update(only_filenames=only_filenames)

    # Derived config
    logic_dirs = to_logic_dirs(experiment_root=experiment_root, model_key=model_key, split=split,
                               formula_dirs=formula_dirs, logic_type=logic_type)
    if len(logic_dirs) == 0:
        raise ValueError("Could not find any logic results directory for setting: {}".format(
            dict(experiment_root=experiment_root, model_key=model_key, split=split,
                 formula_dirs=formula_dirs, logic_type=logic_type)))

    # Collect infos & plot visualizations
    all_summaries = {}
    for logic_dir in logic_dirs:
        formula_dir = os.path.dirname(logic_dir)
        summary_csv = os.path.join(
            logic_dir, "metrics", per_image_stats_csv) if per_image_stats_csv else None
        try:
            conf = get_exp_conf(os.path.join(logic_dir, 'logs'), conf_validation_vals=dict(
                model_key=model_key, split=split))
            # use the continuous formula outout
            iterator = ResultsIterator(conf, load_images=False, **iterator_args)
            df = gather_exp_stats(conf, iterator, show_progress=True, thresh=1e-3,
                                  save_as=summary_csv,
                                  force_reevaluate=force_reevaluate)
        except Exception as e:
            raise RuntimeError(
                "Failed for logic_dir {}".format(logic_dir)) from e
        all_summaries[conf['formula_spec']] = summarize_exp_stats(df, conf)

        # Make sure save names are unique:
        for bestorworst in visualize:
            display(f"Visualizations of the {bestorworst} examples")
            visualize_most_interesting_samples(
                df, conf, num_samples=num_interesting_samples, one_figure=True,
                best=(bestorworst == 'best'),
                save_as=os.path.join(logic_dir, "visualizations",
                                     f"{model_key}-{logic_type}-{formula_to_display_name(os.path.basename(formula_dir))}-{bestorworst}_samples.png"),
                **iterator_args)

    # Summary plots
    mask_types = sorted(set(mask_type for summaries in all_summaries.values()
                            for mask_type in summaries.index.get_level_values(0).unique()) - set(['formula']))
    part_dfs = {formula_to_display_name(mask_type): ([s for s in all_summaries.values()
                                                      if mask_type in s.index.get_level_values(0)][0])
                for mask_type in mask_types}
    formula_dfs = {formula_to_display_name(formula): summaries
                   for formula, summaries in all_summaries.items()}

    # Make sure save names are unique:
    fp = os.path.join(experiment_root, model_key, split,
                      f"{model_key}-{logic_type}-per_image_stats_summary" + "-{}.svg")
    plot_summaries(part_dfs, save_as=fp.format("parts"))
    plot_summaries(formula_dfs, save_as=fp.format("formulas"))


def to_displayable_masks(masks: Dict[str, torch.Tensor], target_size: Tuple[int, int] = None) -> Dict[str, torch.Tensor]:
    """Turn all mask items of ``masks`` into shape ``[h,w]``, potentially splitting up stacked masks. """
    masks = {k: torch.as_tensor(m) for k, m in masks.items() if isinstance(m, (torch.Tensor, float))}
    target_size = target_size or [max(m.size()[-2] if len(m.size()) > 1 else 0 for m in masks.values()),
                                  max(m.size()[-1] if len(m.size()) > 1 else 0 for m in masks.values())]
    new_masks = {}
    for key, mask in masks.items():
        # squeeze leading ones in size():  [1, ..., h, w] -> [..., h, w]
        while len(mask.size()) > 0 and mask.size()[0] == 1:
            mask = mask.squeeze(0)
        # squeeze ones in channel dim: [X, 1, ..., h, w] -> [X, ..., h, w]
        while len(mask.size()) > 3 and mask.size()[1] == 1:
            mask = mask.squeeze(1)
        # broadcast 1D tensors: [channel,] -> [channel, h, w]
        if len(mask.size()) < 2:
            while len(mask.size()) < 3:
                mask = mask.unsqueeze(-1)
            mask = torch.broadcast_to(mask, [mask.size()[0], *target_size])

        if len(mask.size()) == 2:
            new_masks[key] = mask
        # split 3D tensors into separate masks: [channel, h, w] -> ([h, w], [h, w], ...)
        elif len(mask.size()) == 3:
            if mask.size()[0] == 1:  # just one channel
                new_masks[key] = mask.squeeze(0)
            else:
                new_masks.update({f'{key}_{i}': mask[i] for i in range(mask.size()[0])})
        else:
            raise ValueError("Encountered mask of size {}. Cannot split of broadcast this to mask(s) of size [height, width].".format(mask.size()))

    return new_masks


def to_custom_displayable_masks(out: Dict[str, torch.Tensor],
                                person_key: str = 'pedestrian',
                                skip_if_max_lower_than: Dict[str, float] = None,
                                mark_if_score_higher_than: Dict[str, float] = None,
                                scores_to_add_to_key: Sequence[Sequence[str]] = None,
                                reduce_masks: Sequence[str] = None,
                                compare_masks: Sequence[Sequence[str]] = None,
                                mask_union: Callable[[Sequence[torch.Tensor]], torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Custom post-process of a formula calculation output dict for easy display with ``compare_orig_with_masks``.
    Modifications applied:

    - stacked masks are unstacked or reduced via mask union
    - the person mask(s) potentially get scores added to their keys according to ``scores_to_add_to_key``
    - the person mask(s) potentially get marked with color if condition ``mark_if_score_higher_than`` is met
    - in case conditions are not met (``skip_if_max_lower_than``), ``None`` is returned

    Assumptions:

    - tensor of ``len(shape)==1`` is a list of scores (one per prediction)
    - tensor of ``len(shape)==2`` is a standard masks
    - tensor of ``len(shape)==3`` are standard masks stacked in dim 0

    :param out: the dict output of a formula evaluation; only tensors values therein are used
    :param person_key: the key of the (stacked) person mask(s)
    :param skip_if_max_lower_than: dict of the form ``{tensor_key: minimum_value_of_max}``;
        return ``None`` if the max value of tensors in ``out`` at keys are lower
        than the ``minimum_value_of_max``
    :param scores_to_add_to_key: add values of given scores to the string key(s) of the (unstacked) person mask(s)
    :param reduce_masks: reduce the masks at given keys if they are stacked;
        defaults to all stacked masks except for the person mask(s)
    :param compare_masks: each item is a list of keys; for each item add a comparison image
        to the output comparing the masks at keys in that item;
        if the key of a later unstacked mask is given, the unstacked masks are merged
        via union for comparison
    :return: dict of ``{title: mask}`` of 2D and 3D tensors representing masks and images for plotting
    """
    mask_union = mask_union or (lambda t: torch.max(*t, dim=-3))
    # Discard non-tensors
    masks_t: Dict[str, torch.Tensor] = {k: v for k, v in out.items() if isinstance(v, torch.Tensor)}
    person_mask: torch.Tensor = masks_t[person_key]
    assert len(person_mask.shape) == 3
    num_preds: int = person_mask.shape[0] if person_mask.shape[0] > 1 else 1

    # Reduce masks as requested
    reduce_masks = [k for k in reduce_masks if k in masks_t] if reduce_masks is not None else \
        [key for key, m in masks_t.items() if len(m.shape) == 3 and m.shape[0] > 1 if key != person_key]
    for key in reduce_masks:
        mask = masks_t[key]
        masks_t[key] = mask_union([mask])

    # Skip uninteresting images: No person prediction or no false positive prediction?
    for key, min_max_val in skip_if_max_lower_than.items():
        if key in masks_t and key in masks_t and masks_t[key].max() < min_max_val:
            return None

    # Extract prediction scores (single values per prediction)
    per_pred_scores: Dict[str, torch.Tensor] = {k: masks_t.pop(k) for k, v in dict(masks_t).items()
                                                if len(v.shape) == 1 and len(v) == num_preds}

    # Explode stacked, unreduced masks
    masks_t = to_displayable_masks(masks_t)

    # Some custom derived infos
    add_scores: Sequence[str] = [k for k in (scores_to_add_to_key or []) if k in per_pred_scores]
    for i in range(num_preds):
        # Again filter uninteresting masks:
        curr_person_key = f'{person_key}_{i}' if num_preds > 1 else person_key
        new_mask = masks_t.pop(curr_person_key)
        min_max_val = skip_if_max_lower_than.get(person_key, skip_if_max_lower_than.get(curr_person_key, None))
        if min_max_val is not None and new_mask.max() < min_max_val:
            continue

        # add scores to key
        score_strings: Sequence[str] = [f"{per_pred_scores[s][i]:.3f}" for s in add_scores]
        new_key = f'{"ped"+(curr_person_key.split("_")[-1] if num_preds > 1 else "")}'
        if len(add_scores):
            new_key += f'/{"/".join(score_strings)}'

        # potentially mark mask
        for key, lower_limit in (mark_if_score_higher_than or {}).items():
            if key in per_pred_scores and per_pred_scores[key][i].item() >= lower_limit:
                #new_key = '!'+new_key
                new_mask = to_tens(data_visualization.compare_masks(new_mask, colors=('yellow',)))
        masks_t[new_key] = new_mask

    # Add mask comparisons
    colors = ('red', 'blue', 'yellow', 'green')
    for mask_keys in compare_masks or []:
        curr_masks_t = dict(masks_t)
        for key in [*reduce_masks, person_key]:
            if key in mask_keys:
                curr_masks_t.update({key: mask_union([out[key]])})
        if any(k not in curr_masks_t for k in mask_keys):
            continue
        assert 0 < len(mask_keys) < len(colors)
        key = ' vs.\n '.join([f'{key} ({colors[i]})' for i, key in enumerate(mask_keys)])
        masks_t = {key: to_tens(data_visualization.compare_masks(*[curr_masks_t[key] for key in mask_keys], colors=colors)),
                   **masks_t}
    return masks_t


def compare_orig_wt_masks(img_t: torch.Tensor, masks: Dict[str, torch.Tensor], masks_trafo: trafos.Transform = None,
                          label: str = None, fig_scale: float = 3, verbose: bool = False, axes: List[plt.Axes] = None,
                          save_as: str = None, show_fig: bool = True) -> plt.Figure:
    """Plot a figure comparing the original ``img_t`` with the created masks (values in [0,1]).
    If the list of axes to plot the masks into is given, it must have the length of ``masks``
    (plus one for the original image if ``img_t`` is not ``None``).

    :param masks_trafo: transformation to be applied to ``masks`` before plotting
    :param label: y-label of the row
    :param axes: list of (column) axes into which to plot the images and masks
    :param save_as: see :py:func:`~hybrid_learning.experimentation.exp_eval_common.show_and_save_plot`
    :param show_fig: see :py:func:`~hybrid_learning.experimentation.exp_eval_common.show_and_save_plot`
    :return: the plotted figure (the one of ``axes`` if these are given)
    """
    is_subplot = axes is not None
    if not is_subplot:
        num_cols = (len(masks) + 1) if img_t is not None else len(masks)
        fig, axes = plt.subplots(1, num_cols, figsize=(
            fig_scale * num_cols, fig_scale))
    else:
        fig = axes[0].figure
    if label:
        axes[0].set_ylabel(label)
    # original
    if img_t is not None:
        axes[0].imshow(to_img(img_t))
        axes[0].set_title("orig")
    # masks
    for i, (title, tens) in enumerate(masks.items()):
        i = i+1 if img_t is not None else i
        tens = masks_trafo(tens) if masks_trafo else tens
        is_grayscale: bool = len(tens.size()) == 2 or (len(tens.size()) == 2 and tens.size()[0] == 1)
        img = to_img(tens.float(), mode='L' if is_grayscale else None)
        if verbose:
            print("min/max for", title, tens.min(), tens.max())
            print("min/max for", title, np.min(np.array(img)),
                  np.max(np.array(img)))
        cbar = axes[i].imshow(img, vmin=0, vmax=255, cmap='PuBu_r')
        axes[i].set_title(title)
        axes[i].set_yticks([])
    # colorbar
    fig.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.02, 0.8])
    fig.colorbar(cbar, cax=cax)

    if not is_subplot:
        show_and_save_plot(fig=fig, save_as=save_as, show_fig=show_fig)

    return fig


def visualize_most_interesting_samples(df: pd.DataFrame, conf: dict,
                                       num_samples: int = 10, best: bool = False, verbose: bool = False,
                                       one_figure: bool = False, fig_scale: float = 3,
                                       save_as: str = None, **iterator_args) -> pd.DataFrame:
    """Plot the masks for most ``num_samples`` interesting samples side by side.
    "Interesting" here refers to those with the lowest ``mean_interesting`` value
    (mean of formula mask pixel values for pixels in which one of the input part masks exceeds a value).

    :param best: whether to plot the best (True) or the worst samples with respect to ``mean_interesting``
    :param one_figure: whether to plot all into one figure or into single ones
    :param iterator_args: arguments to :py:class:`~hybrid_learning.experimentation.fuzzy_exp.fuzzy_exp_eval.ResultsIterator`
    :param verbose: whether to print information about the plotted samples in a DataFrame
    :return: the sub-DataFrame with information about plotted samples
    """
    # Values for the most interesting examples (worst `mean_interesting` values for formula):
    if best:
        quant_thresh = df.loc[:, ('formula', 'mean_interesting')].quantile(
            1 - num_samples / len(df.index))
        sample_infos = df[df.loc[:,
                                 ('formula', 'mean_interesting')] >= quant_thresh]
    else:
        quant_thresh = df.loc[:, ('formula', 'mean_interesting')].quantile(
            num_samples / len(df.index))
        sample_infos = df[df.loc[:,
                                 ('formula', 'mean_interesting')] <= quant_thresh]
    sample_infos = sample_infos.iloc[:min(len(sample_infos), num_samples)]
    if verbose:
        display('{x} {bestorworst} of {num_all} samples ({num_samples} {bestorworst} ones with (formula, mean_interesting) {gtorlt} {quant_thresh})'
                .format(x=len(sample_infos.index), num_all=len(df.index), num_samples=num_samples, quant_thresh=quant_thresh,
                        bestorworst='best' if best else 'worst', gtorlt='>=' if best else '<='),
                sample_infos)
    masks_trafo = trafos.Resize(img_size=conf['img_size'])
    loaded_imgs: Sequence[Tuple[str, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]] = \
        list(ResultsIterator(**{**iterator_args, **dict(conf=conf, only_filenames=sample_infos.index, verbose=True)}))
    rows = [None]*len(sample_infos.index)
    if one_figure:
        assert len(loaded_imgs) > 0
        num_cols = len(loaded_imgs[0][1][1]) + 1
        num_rows = len(sample_infos.index)
        _, rows = plt.subplots(num_rows, num_cols, figsize=(
            fig_scale * num_cols, fig_scale * num_rows))
    for i, (img_fn, (img_t, masks)) in enumerate(loaded_imgs):
        compare_orig_wt_masks(img_t, masks, masks_trafo=masks_trafo,
                              label="{}, {:.3}".format(
                                  img_fn, sample_infos.loc[img_fn, ('formula', 'mean_interesting')]),
                              axes=rows[i])
        if not one_figure:
            show_and_save_plot()
    if one_figure:
        show_and_save_plot(save_as=save_as,
                           default_fn=f"{'best' if best else 'worst'}_{num_samples}_samples_for_{conf_to_fn(conf)}")
    return sample_infos


def plot_summaries(mask_dfs: Dict[str, pd.DataFrame],
                   save_as: str = None) -> Optional[plt.Figure]:
    """Plot row of bar plots one for each summary mask DataFrame.
    The summary DataFrames should have the format as provided by
    :py:func:`hybrid_learning.experimentation.fuzzy_exp.fuzzy_exp_eval.summarize_exp_stats`.

    :param mask_dfs: dictionary of the form ``{mask_type: mask_stats_dataframe}``
    :param save_as: if not ``None``, save figure under this file path
    """
    if len(mask_dfs) == 0:
        logging.getLogger().warning(
            "plot_summaries encountered empty list of mask data frames! Skipping.")
        return
    fig, axes = plt.subplots(1, len(mask_dfs), figsize=(
        len(mask_dfs)*6, 3), squeeze=False)
    axes = axes[0]

    for i, (mask_type, mask_df) in enumerate(mask_dfs.items()):
        curr_select_key = mask_type if mask_type in mask_df.index.get_level_values(
            0) else "formula"
        plot_df = mask_df.loc[(curr_select_key, slice(
            None)), :].reset_index(level=0, drop=True).T
        yerr = pd.DataFrame([np.zeros(len(plot_df.index)), plot_df['std'], np.zeros(len(plot_df.index))],
                            index=['max', 'mean', 'min'], columns=plot_df.index).T
        plot_df.drop('std', axis=1).plot.bar(rot=60, ax=axes[i], legend=False, yerr=yerr,
                                             title=mask_type)
    axes[-1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    show_and_save_plot(fig=fig, save_as=save_as)
    return fig


def plot_metric(metrics_pd: pd.DataFrame, model_key: str, metrics: str = 'recall',
                verbose: bool = True,
                formulas: tuple = None,
                logic_types: tuple = (
                    'boolean', 'lukasiewicz', 'product', 'goedel'),
                title: str = "{model_key} {metric}",
                formula_name_col: Union[str, Dict[str, str]] = {
                    'formula_attrs': 'formula'},
                save_as: str = None,
                xlim: Tuple[float, float] = (0, 1),
                to_pretty_names: Dict[str, str] = None,
                ax=None,
                ) -> List[pd.DataFrame]:
    """Plot a bar chart for each of the given metrics from entries in metrics_pd.

    :param metrics_pd: DataFrame with all of ``formula_name_col``, ``"model_key"``, ``metrics``, ``logic_type`` as columns
    :param verbose: whether to display some intermediate results
    :param formulas: specifier strings of the formulas to match (simple filter option on ``formula_name_col`` column)
    :param logic_types: specifier strings of the logic types to match (simple filter option on ``"logic_type"`` column)
    :param title: figure title; may contain formatting placeholders ``{model_key}`` and ``{metric}``
    :param formula_name_col: name of the column containing the formula specs to match with ``formulas``
    :param save_as: if given, either directory or file path where to save this plot
    :param xlim: if set, the minimum and maximum values displayed on the x-axis;
        if ``None`` or ``False``, automatically determined by matplotlib
    :param to_pretty_names: dictionary mapping values of ``model_key`` and ``metrics`` to pretty names
    :param ax: an axis into which to plot; by default, a new figure is created
    :return: list with the filtered DataFrames that got plotted
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    if ax is not None and len(metrics) > 1:
        raise ValueError(
            "Cannot plot several metrics into one axis! Metrics: {}".format(metrics))
    to_pretty_names = to_pretty_names or {}
    def pretty(x): return to_pretty_names.get(x, x)
    # Nice ylabel
    if isinstance(formula_name_col, dict):
        assert len(formula_name_col) == 1
        ylabel = list(formula_name_col.values())[0]
        formula_name_col = list(formula_name_col.keys())[0]
    else:
        ylabel = formula_name_col

    # Title
    title = lazy_format(title, model_key=pretty(model_key), metric=', '.join(
        [pretty(metric).replace('_', ' ') for metric in metrics]))

    # Show formulas
    if verbose:
        display("Available formulas:", set([tuple(v.values()) for v in metrics_pd[[
                'formula', 'formula_attrs']].T.to_dict().values()]))
        display("Available gt_formulas:", set([tuple(v.values()) for v in metrics_pd[[
                'gt_formula', 'gt_formula_attrs']].T.to_dict().values()]))

    # Pre-process DataFrame: Subsetting & sorting
    formulas = formulas or sorted(
        set([formula for formula in metrics_pd[formula_name_col]]))
    metrics_pd = metrics_pd[metrics_pd[formula_name_col].isin(formulas)
                            & (metrics_pd.model_key == model_key)
                            & metrics_pd.logic_type.isin(logic_types)] \
        .sort_values(by='logic_type', key=lambda col: [logic_types.index(entry) for entry in col]) \
        .sort_values(by=formula_name_col, key=lambda col: [formulas.index(entry) for entry in col])

    # Display DataFrame
    if verbose:
        with pd.option_context('display.max_columns', 500, 'display.max_colwidth', 1000):
            display(metrics_pd[['formula_attrs', 'formula', 'gt_formula', 'logic_type', *metrics, 'logic_dir']]
                    .sort_values([formula_name_col, 'logic_type'], key=lambda col: [[*formulas, *metrics_pd[formula_name_col]].index(e) if col.name == formula_name_col
                                                                                    else logic_types.index(e) for e in col]) \
                    # .apply(lambda x: x.map(lambda d: os.path.basename(os.path.dirname(d))) if x.name == 'logic_dir' else x) \
                    .apply(lambda x: x.astype(float) if x.name in metrics else x)
                    .style.highlight_between(subset=metrics, left=0, right=0.5)
                    )
    fig, axes = None, [ax]
    if ax is None:
        fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 4*0.125*len(metrics_pd[formula_name_col].unique())),
                                 sharey=True, squeeze=False)
        axes = axes[0]
    plotted_pds = []
    for metric, ax in zip(metrics, axes):
        # Plot DataFrame
        plotted_pd = metrics_pd[[formula_name_col, metric, 'logic_type']] \
            .pivot(index=formula_name_col, columns='logic_type') \
            .T.reset_index(level=0, drop=True).T \
            .astype(float) \
            .sort_index(key=lambda col: [-formulas.index(entry) for entry in col]) \
            .sort_index(key=lambda row: [-logic_types.index(entry) for entry in row], axis=1, level=1)
        if plotted_pd.isnull().sum().sum() > 0:
            logging.getLogger().warning(
                "Found NaN values for metric {}! Please inspect the data more closely:\n{}".format(metric, plotted_pd.isnull()))
        if plotted_pd.size > 0:
            plotted_pds.append(plotted_pd)
            ax = plotted_pd.plot.barh(legend=0, ax=ax)
            if xlim:
                ax.set_xlim(*xlim)
            ax.set_xlabel(pretty(metric).replace('_', ' '))
            #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            logging.getLogger().warning("No data found to plot for model %s.", model_key)
    axes[0].set_ylabel(ylabel.replace('_', ' '))
    if fig and len(axes) > 1:
        fig.suptitle(title, y=0.92)
        fig.subplots_adjust(wspace=0.1)
        # handles, labels = [sum(hl, start=[]) for hl in zip(*[ax.get_legend_handles_labels() for ax in axes])]
        # fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        axes[0].set_title(title)
    axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if fig:
        show_and_save_plot(fig=fig, save_as=save_as,
                           default_fn=f"{title}--{','.join(metrics)}")
    return plotted_pds


def plot_curve(metrics_pd: pd.DataFrame,
               # Axes
               x='recall', y='precision', ax=None,
               # Varying factors and constraints
               variables=None,
               constraints: Dict[str, float] = None,
               model_key: str = 'maskrcnn_box',
               logic_types: Sequence[str] = ('boolean', 'lukasiewicz', 'product', 'goedel'),
               formulas: List[str] = None,
               # Additional plot infos
               add_diagonal: bool = None,
               add_value_hints: bool = True, add_markers: bool = False,
               legend_outer: bool = True,
               # Style options
               title: str = "ROC {model_key} ({variable})", figsize=(10, 10),
               formula_name_col: str = 'formula_attrs',
               to_pretty_names: Dict[str, str] = None,
               verbose: bool = False,
               lims: Tuple[Tuple[float, float], Tuple[float, float]] = (
                   (0, 1.0), (0, 1.0)),
               save_as: str = None):
    """
    Plot a curve of ``y`` against ``x`` varying the value of ``variables[0]`` given ``constraints``.
    For each entry in ``logic_types`` and ``formulas``, and for each value of entries in ``variables[1:]``,
    produce a new line.

    To just get a normal plot, call
    ``plot_curve(x='x', y='y', variables=['x'], add_value_hints=False, add_diagonal=False, ...)``.

    :param x: column name of the x-values
    :param y: column name of the y-values
    :param variables: keys of variables in the DataFrame to vary over;
        different values for the first one are plotted into one curve,
        further variables will create new curves for different values
    :param constraints: mapping of keys of variables in the dataframe to a fixed value; used to filter the DataFrame
    :param model_key: shorthand for ``constraints={'model_key': model_key}``
    :param logic_types: filter the DataFrame for any of these values of the ``"logic_type"`` column
    :param formulas: if given, filter the DataFrame for any of these values of the ``formula_name_col`` column
    :param formula_name_col: see formulas
    :param add_value_hints: whether to add small text fields with the values of ``variable`` at each sample point
    :param add_diagonal: add a diagonal; set to ``False`` to disable; else set to
        ``'start_at_one'`` (default for PR-curves) or ``'start_at_zero'`` (default else)
    :param legend_outer: whether to put the legend of plots outside of the plot
    :param title: figure or axis title; may contain formatting placeholders ``{model_key}``, and ``{variable}`` (for ``variables[0]``)
    :param to_pretty_names: dictionary mapping values of ``model_key`` and ``metrics`` to pretty names
    :param verbose: whether to also prettily display all values of the relevant columns of the DataFrame
    :param lims: tuple of ``((xmin, xmax), (ymin, ymax))`` to set the xlim and ylim of the axes
    :param save_as: if given, either directory or file path where to save this plot
    :return: all plotted rows of the DataFrame
    """
    # Argument defaults
    formulas = [formulas] if isinstance(formulas, str) else formulas
    add_diagonal = add_diagonal if add_diagonal is not None else (
        'start_at_one' if (x == 'recall' and y == 'precision') else 'start_at_zero')
    to_pretty_names = to_pretty_names or {}
    def pretty(x): return to_pretty_names.get(x, x)
    variables = variables or []
    variable = variables[0] if len(variables) else x

    # Value check for constraints
    constraints = constraints or {}
    assert all(k not in variables for k in constraints), \
        "Overlap between keys to vary and constrained keys:\nvariables: {}\nconstraints: {}".format(
            variables, constraints)
    for col_keys, col_desc in [([x], 'x-axis'), ([y], 'y-axis'), (variables, 'variable'), (constraints, 'constraint')]:
        for col_key in col_keys:
            assert col_key in metrics_pd.columns, "Missing {} column {} from metrics_pd; available columns: {}".format(
                col_desc, col_key, metrics_pd.columns)

    # Title
    title = lazy_format(title, model_key=pretty(model_key), variable=pretty(variable))

    # Subsetting & types
    metrics_pd = metrics_pd[(metrics_pd.model_key == model_key) & (metrics_pd.logic_type.isin(logic_types))]
    # Get formulas that have data for more than one variable value:
    formulas = formulas or sorted(set([formula for formula in metrics_pd[formula_name_col]
                                       if len(set(metrics_pd[metrics_pd[formula_name_col] == formula][variable])) > 1]))  # == set(metrics_pd[variable])]))

    plotted_pd = metrics_pd[metrics_pd[formula_name_col].isin(formulas)] \
        .astype({x: float, y: float, variable: float})
    if plotted_pd.size == 0:
        return plotted_pd

        # Display DataFrame
    if verbose:
        with pd.option_context('display.max_columns', 500, 'display.max_colwidth', 1000):
            display(metrics_pd[['formula_attrs', 'formula', 'gt_formula', 'logic_type', x, y, *variables, 'logic_dir']] \
                    # .sort_values(['formula', 'logic_type'], key = lambda col: [formulas.index(e) if col.name=='formula' and e in formulas else logic_types.index(e) for e in col]) \
                    # .apply(lambda x: x.map(lambda d: os.path.basename(os.path.dirname(d))) if x.name == 'logic_dir' else x) \
                    .apply(lambda col: col.astype(float) if col.name in (x, y, *variables) else col)
                    .style.highlight_between(subset=[x, y], left=0, right=0.5)
                    )

    line_specs: Set[Tuple[str, str, Tuple[Tuple[str, float], ...]]] = set(
        plotted_pd.apply(lambda row: (row[formula_name_col], row['logic_type'], tuple((variable, row[variable]) for variable in variables[1:])),
                         axis=1)
    )

    # Colors
    cmap = plt.get_cmap("tab10")
    markers = [2, 3, '|', 'x', '+', 'v', '^', '>', '<',
               "3", "4"] if add_markers else ['']*len(line_specs)
    if cmap.N < len(line_specs):
        logging.getLogger().warning("Got %d combinations of (logic_type, formula) but only have %d colors, so some lines will get the same color!", len(line_specs), cmap.N)
    if len(markers) < len(line_specs):
        logging.getLogger().warning("Got %d combinations of (logic_type, formula) but only have %d marker types, so some lines will get the same marker!",
                                    len(line_specs), len(markers))
    for i, (formula, logic_type, variable_vals) in enumerate(sorted(line_specs, key=lambda lf: (lf[0], logic_types.index(lf[1]), lf[2]))):
        variable_vals: Dict[str, float] = dict(variable_vals)
        curr_plotted_pd = constrain_pd(plotted_pd, {formula_name_col: formula,
                                                    'logic_type': logic_type,
                                                    **variable_vals, **constraints})
        # Plot the data
        ax = curr_plotted_pd.plot(x=x, y=y,
                                  ax=ax, figsize=figsize if ax is None else None,
                                  label=', '.join([pretty(s) for s in [logic_type, *[f"{pretty(k)}={v}" for k, v in {**variable_vals, **constraints}.items()], formula]
                                                   if pretty(s)]),
                                  color=cmap(i), marker=markers[min(len(markers)-1, i)])
        # Add diagonal
        diag_points = None if not add_diagonal else (
            [[0, 1], [1, 0]] if add_diagonal == 'start_at_one' else [[0, 1], [0, 1]])
        if diag_points:
            ax.plot(*diag_points, '--', color='gray', linewidth=1)

        # Set format and meta-data
        ax.set_xlim(*lims[0]), ax.set_ylim(*lims[1])
        ax.set_xlabel(pretty(x).replace('_', ' ')), ax.set_ylabel(
            pretty(y).replace('_', ' '))
        ax.set_title(title)
        if legend_outer:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # Add value hints
        if add_value_hints:
            for _, row in curr_plotted_pd.iterrows():
                ax.annotate(f'{row[variable]:.5f}', xy=(row[x], row[y]),
                            fontsize=7, color=cmap(i),
                            xytext=(0, 10), textcoords='offset points',
                            rotation=90, va='bottom', ha='left',
                            #arrowprops = {'width': .5, 'headwidth': .5, 'headlength': .5, 'shrink':0.05}
                            )
    show_and_save_plot(
        save_as=save_as, default_fn=f"{title}--{y}_vs_{x}_over_{variable}{'_with_' if len(constraints) else ''}{'-'.join([f'{k}{v}' for k, v in constraints.items()])}")
    return plotted_pd


def visualize_random_samples(
        experiment_root: str, model_key: str, split: str, logic_type: str,
        formula_dirs: list = None,
        changed_constants=None,  # dict(mask_thresh=0.1),
        num_samples: int = 15, only_filenames: Iterable = None,
        skip_missing: Union[bool, str] = 'warn',
        save_as: str = None,
        additional_formula_mods: Dict[str, Callable[[
            fuzzy_logic.Merge, fuzzy_logic.Merge], fuzzy_logic.Merge]] = None,  # dict(
        #recalced=(lambda formula_obj, _: formula_obj),
    #recalced_gt=(lambda _, gt_formula_obj: gt_formula_obj),
    #non_binary=(lambda formula_obj, _: formula_obj.in_keys[0]),
    #threshed_concepts=(lambda formula_obj, _: formula_obj.in_keys[0].in_keys[1]),
    #unthreshed_concepts=(lambda formula_obj, _: formula_obj.in_keys[0].in_keys[1].in_keys[1]),
    #any_concept_formula = (lambda formula_obj, _: formula_obj.in_keys[0].in_keys[0]),
    # )
        verbose: bool = False,
) -> List[Dict[str, Any]]:
    """For the first ``num_samples`` samples visualize the cached concept and recalculated formula output masks.
    The formula objects given by ``additional_formula_mods`` are recalculated based on the given
    ``changed_constants``, and displayed together with the concept outputs.
    The original formula objects and settings are loaded from the sacred experiment logs,
    the concept outputs are loaded from cache (defined in experiment settings).
    For the experiment logs, the following folder structure is assumed:
    ``experiment_root/model_key/split/<formula_dir>/<logic_type>/logs/``

    :param additional_formula_mods: see :py:class:`~hybrid_learning.experimentation.fuzzy_exp.fuzzy_exp_eval.ResultsIterator`
    :param changed_constants: see :py:class:`~hybrid_learning.experimentation.fuzzy_exp.fuzzy_exp_eval.ResultsIterator`
    :param num_samples: number of samples to plot
    :param skip_missing: see :py:func:`~hybrid_learning.experimentation.fuzzy_exp.fuzzy_exp_eval.to_logic_dirs`
    :param save_as: see :py:func:`~hybrid_learning.experimentation.exp_eval_common.show_and_save_plot`
    """
    num_samples = len(only_filenames) if only_filenames is not None else num_samples
    additional_formula_mods = additional_formula_mods or {}
    logic_dirs = to_logic_dirs(experiment_root=experiment_root, model_key=model_key, split=split,
                               formula_dirs=formula_dirs, logic_type=logic_type, skip_missing=skip_missing)
    # Collect infos & plot visualizations
    for logic_dir in logic_dirs:
        # GET FORMULA AND ITERATOR
        conf = get_exp_conf(os.path.join(logic_dir, 'logs'), conf_validation_vals=dict(
            model_key=model_key, split=split))
        iterator = ResultsIterator(conf, raise_on_missing=not skip_missing, only_filenames=only_filenames,
                                   additional_formula_mods=additional_formula_mods,
                                   changed_constants=changed_constants,
                                   recalc=True, verbose=verbose)
        masks_trafo = trafos.Resize(img_size=conf['img_size'])

        # PLOT AND COLLECT HISTOGRAM (on the first X available samples)
        all_vals = []
        skipped_img_fns = []
        found_something_to_plot = False
        fig, rows = None, None
        for i, (img_fn, (img_t, masks_t)) in enumerate(iterator):
            if i >= num_samples:
                break
            if any(f is None for f in (img_t, *masks_t.values())):
                num_samples += 1
                skipped_img_fns.append(img_fn)
                continue
            found_something_to_plot = True
            if fig is None:
                num_rows, num_cols = num_samples, 1+len(masks_t)
                fig, rows = plt.subplots(num_rows, num_cols, figsize=(
                    3*num_cols, 3*num_rows), squeeze=False)
            compare_orig_wt_masks(img_t, masks_t, masks_trafo=masks_trafo,
                                  label=img_fn,
                                  axes=rows[i])

        if found_something_to_plot:
            if len(skipped_img_fns) > 0 and skip_missing == 'warn':
                logging.getLogger().warning(
                    "Skipped descriptors due to missing files: %s", skipped_img_fns)
            show_and_save_plot(fig=fig, save_as=save_as,
                               default_fn=f"random_samples_with_{'_'.join([f'{k}{v}' for k, v in (changed_constants or {}).items()])}_for_{conf_to_fn(conf)}")
        else:
            logging.getLogger().warning("Did not find any descriptor to plot due to missing files.")

        # HISTOGRAM
        plt.hist(np.array(all_vals).reshape(-1), bins=1000,
                 range=[0, 0.005])  # of interest for low threshold


def conf_to_fn(conf: dict) -> str:
    """Turn a sacred experiment config to a unique file name."""
    formula_dir = calc_helpers.formula_spec_to_dir(conf["formula_spec"], conf["fuzzy_logic_key"],
                                      constants=conf["constants"], predicate_setts=conf["predicate_setts"])
    return '_'.join([formula_dir, conf["fuzzy_logic_key"]])