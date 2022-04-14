#  Copyright (c) 2022 Continental Automotive GmbH
"""Common utility functions for modifying experiment data."""

import functools
import itertools
import json
import operator
import os
from datetime import datetime
from typing import Sequence, Dict, Any, Callable, List, Union, Tuple, Set

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from hybrid_learning.datasets.transforms.common import lazy_format


def f_beta(precision: float, recall: float, beta: float = 1):
    """Calculate F1_beta score (cf. https://en.wikipedia.org/wiki/F-score).
    The beta factor makes recall beta-times as important as precision."""
    if pd.isna(precision) or pd.isna(recall):
        return np.nan
    assert isinstance(precision, float) and isinstance(recall, float), \
        "Encountered precision (type {}, {}) or recall (type {}, {}) or wrong type".format(
            type(precision), precision, type(recall), recall)
    assert precision >= 0 and recall >= 0, "Encountered negative precision ({}) or recall ({})".format(
        precision, recall)
    dividend = (1 + (beta**2)) * (precision*recall)
    divisor = (((beta**2)*precision) + recall)
    return dividend/divisor if divisor > 0 else 0


def auc(x: Sequence[float], y: Sequence[float], add_boundaries: bool = True, skip_na: bool = True) -> float:
    """Calculate the area under curve for the given x and y values.

    :param add_boundaries: whether to add points (0,0) and (0,1) (for ROC curve AUC calculation)
    :param skip_na: whether to simply prune NaN values; if ``False``, ``NaN`` is returned
        if any provided value is ``NaN``; if ``True``, ``NaN`` is returned only in case no
        area can be calculated.
    :return: the AUC value
    """
    assert len(x) == len(y)
    # Prune NaN values
    if skip_na:
        x, y = zip(*[(vx, vy) for vx, vy in zip(x, y) if not pd.isna(vx) and not pd.isna(vy)])
    # Make sure x values are ascending
    x, y = zip(*sorted(zip(x, y), key=lambda x_y: x_y[0]))
    if add_boundaries:
        x, y = [0., *x, 1.], [0., *y, 1.]
    if len(x) == 0:
        return pd.NA
    total_area = 0.
    for i in range(len(x)-1):
        # add next trapezoid
        total_area += 0.5 * (y[i] + y[i+1]) * (x[i+1] - x[i])
    return total_area


def update_dict(d: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Return a dict that is a recursive update of ``d`` with values in ``update``."""
    updated: Dict[str, Any] = dict(d)
    for key, val in update.items():
        if key in updated and isinstance(updated[key], dict):
            updated[key] = update_dict(updated[key], val)
        else:
            updated[key] = val
    return updated


def flatten_cols(df: pd.DataFrame, columns: Sequence[str], keep_columns: Sequence[str] = tuple()) -> pd.DataFrame:
    """If items in an dataframe are iterables, flatten these.
    Flattening here means:

    >>> df = pd.DataFrame({'a': {'idx1': (1,2), 'idx2': (3,)},
    ...                    'b': {'idx1': ('x', 'y'), 'idx2': ('z',)},
    ...                    'keep': {'idx1': 'k1', 'idx2': 'k2'}})
    >>> flatten_cols(df, columns=['a', 'b'], keep_columns=['keep'])
       orig_idx  a  b  keep
    0         0  1  x    k1
    1         0  2  y    k1
    2         1  3  z    k2

    :param df: the dataframe to create a modified version of
    :param columns: the columns to include into the new frame with values flattened
    :param keep_columns: spread entries from these columns from a row to the rows
        newly created from that row
    :return: a new dataframe ``new_df`` with columns ``['orig_idx', *columns, *keep_columns]``
        where for each index ``idx``, ``col`` in ``columns``, ``kcol`` in ``keep_columns``:
        ``df.loc[idx, col][i] == new_df[new_df['orig_idx']==idx][col].iloc[i]``
        and ``new_df.loc[idx, kcol] == df.loc[new_df.loc[idx, 'orig_idx'], kcol]``
    """
    columns, keep_columns = list(columns), list(keep_columns)
    # ensure entries in one row have the same length
    assert (df[columns].applymap(len).std(1, ddof=0) == 0).all()
    return pd.DataFrame([
        [orig_idx, *vals, *vals_to_keep]
        for orig_idx, (vals_to_flatten, vals_to_keep) in enumerate(zip(df[columns].values, df[keep_columns].values))
        for vals in zip(*vals_to_flatten)],
        columns=['orig_idx', *columns, *keep_columns]
    )


def cache_at(save_as: str, common_settings: Dict[str, Any] = None):
    """Return decorator that will cache the returned list of dicts in the JSON file ``save_as``.
    If the file exists, the results are loaded from the file instead of
    executing the decorated function, ensuring the same ``common_settings`` had been used.

    :raises: An ``AssertionError`` is raised if the ``common_settings`` do not coincide.

    """
    common_settings = common_settings or {}

    def cache_at_decorator(fun: Callable[..., List[Dict[str, Any]]]):
        def cached_fun(*args, to_pandas: bool = True, **kwargs
                       ) -> Union[Dict[str, Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame]:
            """Cached function.

            :param to_pandas: whether the result obtained from the decorated function or cache
                should be post-processed to become a pandas DataFrame.
            :return: the result of the decorated function (potentially loaded from a JSON cache file)
            """
            if os.path.exists(save_as):
                with open(save_as, 'r') as f:
                    results_raw: Dict[str, Dict[str, Any]] = json.load(f)
                cached_common_settings = results_raw.pop('info', {})
                if cached_common_settings != common_settings:
                    cached_common_settings = cached_common_settings or {}
                    differing_keys = [k for k in [*cached_common_settings.keys(), *common_settings.keys()]
                                      if k not in set(cached_common_settings.keys()).intersection(common_settings.keys())]
                    differing_values = [k for k, v in cached_common_settings.items()
                                        if k in common_settings and common_settings[k] != v]
                    raise AssertionError(("Common settings cached at {} do not match assumed common settings:"
                                          "\nCached: {}\nAssumed: {}\nDiffering keys: {}\nMatching keys with different values: {}"
                                          ).format(save_as, json.dumps(cached_common_settings), json.dumps(common_settings), differing_keys, differing_values))
                if to_pandas:
                    return pd.DataFrame(results_raw).T
                return results_raw
            else:
                outs: List[Dict[str, Any]] = fun(*args, **kwargs)

                try:
                    to_save = {'info': common_settings, **{i: outs[i] for i in range(len(outs))}}
                    os.makedirs(os.path.dirname(save_as), exist_ok=True)
                    with open(save_as, 'w') as f:
                        json.dump(to_save, f)
                except Exception as e:
                    print(f"WARNING: Caching results at {save_as} failed with the following error:\n{e}")

                if to_pandas:
                    try:
                        # Make common dataframe
                        common_settings_pd: pd.Series = pd.Series(common_settings)
                        results: pd.DataFrame = pd.DataFrame(outs)
                        results.loc[:, common_settings_pd.index] = pd.concat([common_settings_pd]*results.index.size, axis=1).T
                        return results
                    except Exception as e:
                        print(f"WARNING: Conversion to pandas.DataFrame failed with the following error (returning dict representation):\n{e}")
                return outs
        return cached_fun
    return cache_at_decorator


def show_and_save_plot(fig: plt.Figure = None, save_as: str = None,
                       show_fig: bool = True, default_fn: str = None,
                       allowed_endings: Sequence[str] = ('.png', '.pdf', '.svg')):
    """Save a plot with proper default timestamped filename.

    :param fig: the figure to plot, defaults to ``plt.gcf()``
    :param save_as: the file path or directory path under which to save the figure;
        if save_as is a directory, the default_fn is used to create the file name;
        may contain the formatting key ``{default}`` which is then replaced by
        ``default_fn`` (with file ending!), and ``{ts}`` which is replaced by a timestamp
    :param show_fig: whether to call ``show()`` on the figure
    :param default_fn: the default file name (with .pdf as default file ending)
    :param allowed_endings: if ``save_as`` does not end in one of ``allowed_endings``,
        it is assumed to be the file name and ``".pdf"`` is appended
    """
    fig = fig or plt.gcf()
    if show_fig:
        fig.show()

    if not save_as:
        return

    # Format with {default} and {ts}
    if default_fn and not any(default_fn.endswith(ending) for ending in allowed_endings):
        default_fn = f"{default_fn}.pdf"
    if "{default}" in save_as and not default_fn:
        raise ValueError(
            "save_as ({}) contains formatting string for default_fn, which is not given".format(save_as))
    save_as = lazy_format(save_as, default=default_fn,
                          ts=datetime.now().strftime('%Y-%m-%d_%H%M%S%f'))

    # Default filename if save_as is dir
    if os.path.isdir(save_as) or not any(save_as.endswith(ending) for ending in allowed_endings):
        if not default_fn:
            raise FileExistsError(
                "save_as {} is an existing directory and no default_fn is given".format(save_as))
        default_fn = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S%f')}_{default_fn}"
        save_as = os.path.join(save_as, default_fn)
    if os.path.dirname(save_as):
        os.makedirs(os.path.dirname(save_as), exist_ok=True)

    # Actual saving
    fig.savefig(save_as, bbox_inches='tight')
    if save_as.endswith('.svg') or save_as.endswith('.pdf'):
        fig.savefig(save_as.replace('.svg', '.png').replace(
            '.pdf', '.png'), bbox_inches='tight')


def do_for(func: Callable, allow_looping=('experiment_root', 'model_key', 'split', 'logic_type'), verbose_looping: bool = True,
           **kwargs) -> List[Tuple[Dict[str, Any], Any]]:
    """Apply ``func`` to all combinations of elements in the ``allow_looping`` list of ``kwargs``-keys.
    If a value for an ``allow_looping`` key is a list or tuple, it will be iterated over.
    Nested lists get flattened.

    :return: a list of tuples ``(used_kwargs, func_return)``
    """
    ret = []
    for key in sorted(set(allow_looping).intersection(set(kwargs.keys())), key=lambda k: allow_looping.index(k)):
        values = kwargs[key]
        if isinstance(values, (list, tuple)):
            for curr_value in values:
                if verbose_looping:
                    print(f"------- EVALUATION FOR: ", ", ".join([f'{k}={curr_value if k==key else v}' for k, v in kwargs.items()
                                                                 if k == key or not (k in allow_looping and isinstance(v, (list, tuple)))]))
                curr_kwargs = {**kwargs, key: curr_value}
                ret += do_for(func, allow_looping=allow_looping,
                              verbose_looping=verbose_looping, **curr_kwargs)
            return ret
    return [(kwargs, func(**kwargs))]


def to_fig_for(func: Callable, allow_looping=('experiment_root', 'model_key', 'split', 'logic_type'),
               fig_args: Dict[str, Any] = None, shared_legend_in: Tuple[int, int] = False,
               save_overall_fig_as: str = None,
               **kwargs) -> Tuple[plt.Figure, List[Tuple[Dict[str, Any], Any]]]:
    """Apply ``func`` to all combinations of elements in the ``allow_looping`` list of ``kwargs``-keys and gather results into image.
    If a value for an ``allow_looping`` key is a list or tuple, it will be iterated over.
    Nested lists get flattened.
    At each call, the ``func`` is supplied with an ``ax`` argument
    holding the axis of the overall figure the function should use.
    Columns are indexed by values of the first key of ``allow_looping``,
    rows by combinations of all other looping values.

    :param func: the function to loop over
    :param allow_looping: the keys of arguments to loop over in case they are lists or tuples;
        each column corresponds to a value of the first key
    :param fig_args: fed as keyword arguments to the figure initialization
    :param shared_legend_in: legends from all axes in the subplots are removed except for the one
        determined by the ``(row, col)`` coordinate in ``shared_legend_in`` (if given)
    :param save_overall_fig_as: file path where to save the final figure
    :return: a tuple of the figure and a list of tuples ``(used_kwargs, func_return)``
    """
    fig_args: Dict[str, Any] = dict(fig_args or {})
    figscale: float = fig_args.pop('figscale', 3)
    ret = []
    looping_keys: Set[str] = set(allow_looping).intersection(set(kwargs.keys()))
    looping_keys: List[str] = sorted([k for k in looping_keys if isinstance(kwargs[k], (list, tuple))],
                                     key=lambda k: allow_looping.index(k))
    total_axes: Dict[str, int] = {
        k: len(v) for k, v in kwargs.items() if k in looping_keys}
    nrows: int = max(
        1, sum(l for k, l in total_axes.items() if k in looping_keys[1:]))
    ncols: int = max(1, total_axes[looping_keys[0]])

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False,
                             **{'figsize': (figscale*ncols, figscale*nrows), **fig_args},)
    # fig.tight_layout()
    col_values = [{looping_keys[0]: v} for v in kwargs[looping_keys[0]]]
    row_values = [dict(zip(looping_keys[1:], vals)) for vals in itertools.product(
        *[kwargs[k] for k in looping_keys[1:]])]
    for i, row_value in enumerate(row_values):
        for j, col_value in enumerate(col_values):
            curr_kwargs = {**kwargs, **col_value,
                           **row_value, 'ax': axes[i, j]}
            ret.append((curr_kwargs, func(**curr_kwargs)))

    if shared_legend_in:
        for i, j in itertools.product(range(nrows), range(ncols)):
            if (i, j) != shared_legend_in:
                legend = axes[i, j].get_legend()
                if legend:
                    legend.remove()
    if save_overall_fig_as is not None:
        os.makedirs(os.path.dirname(save_overall_fig_as), exist_ok=True)
        fig.savefig(save_overall_fig_as, bbox_inches='tight')
    return fig, ret


def constrain_pd(df: pd.DataFrame, constraints: Dict[str, Any]):
    """Given a dict of ``{col_name: constraint_value}`` return the dataframe row subset fulfilling all constraints."""
    for k in constraints.keys():
        assert k in df.columns, "Missing column {} from dataframe with columns {}".format(k, df.columns)
    cdf = df[(functools.reduce(operator.and_, [df[k] == v for k, v in constraints.items()], [True]*len(df.index)))]
    if isinstance(cdf, pd.Series):
        cdf = cdf.to_frame()
    return cdf
