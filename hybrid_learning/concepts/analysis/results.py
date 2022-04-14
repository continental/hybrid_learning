"""Handles for processing and storing concept analysis results.
These are used as exchange format for results within the
:py:class`~hybrid_learning.concepts.analysis.concept_detection.ConceptAnalysis`
class.
"""

#  Copyright (c) 2022 Continental Automotive GmbH
import abc
import os
from typing import Tuple, Dict, ItemsView, List, Union, Iterable, Sequence

import numpy as np
import pandas as pd

from hybrid_learning.concepts.models.embeddings import ConceptEmbedding


class ResultsHandle(abc.ABC):
    """Base class for dictionary form result handles."""

    @abc.abstractmethod
    def save(self, folder: str):
        """Save the results under the root given by ``folder``."""
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def load(cls, folder: str) -> 'ResultsHandle':
        """Load results and return handle for them."""
        raise NotImplementedError()

    @abc.abstractmethod
    def to_pandas(self) -> pd.DataFrame:
        """Return a pandas object representation of the held results."""
        raise NotImplementedError()

    def __repr__(self) -> str:
        return self.to_pandas().to_string()

    @staticmethod
    def emb_info_to_pandas(embs: Union[ConceptEmbedding,
                                       Iterable[ConceptEmbedding]],
                           stats: pd.Series = None,
                           std_dev: Tuple[np.ndarray, float, float] = None
                           ) -> pd.Series:
        """Quick info about an embedding and its stats (and standard dev)
        as :py:class:`pandas.Series`."""
        emb_info = {}
        stats_info = stats if stats is not None else {}
        embs = [embs] if isinstance(embs, ConceptEmbedding) else embs
        for i, emb in enumerate(embs):
            if emb.normal_vec is not None:
                emb_info.update(
                    {f"normal vec len_{i}": np.linalg.norm(emb.normal_vec),
                     f"scaling factor_{i}": float(emb.scaling_factor or 0)})
            if emb.bias is not None:
                emb_info.update(
                    {f"support factor_{i}": float(emb.support_factor)})
        std_info = {"std dev normal vec (len)": np.linalg.norm(std_dev[0]),
                    "std dev support factor": std_dev[1],
                    "std dev scaling factor": std_dev[2]} \
            if std_dev is not None else {}
        return pd.Series({**stats_info, **emb_info, **std_info})

    @classmethod
    def emb_info_to_string(cls,
                           embs: Union[ConceptEmbedding,
                                       Iterable[ConceptEmbedding]],
                           stats: pd.Series = None,
                           std_dev: Tuple[
                               np.ndarray, float, float] = None) -> str:
        """Printable quick info about the given embedding with stats
        (and standard deviation)."""
        info: pd.Series = cls.emb_info_to_pandas(embs, stats,
                                                 std_dev=std_dev)
        # Formatting
        float_format: str = "{: < 14.6f}"
        exp_format: str = "{: < 14.6e}"
        for idx in [i for i in info.index if "std" in i]:
            info[idx] = exp_format.format(info[idx])
        return info.to_string(float_format=float_format.format)


class AnalysisResult(ResultsHandle):
    """Handle for saving, loading and inspection of analysis results.
    The results are saved in :py:attr:`results`. See there for the format."""

    def __init__(self,
                 results: Dict[str, Dict[int, Tuple[Sequence[ConceptEmbedding],
                                                    pd.Series]]]):
        self.results: Dict[str, Dict[int, Tuple[Sequence[ConceptEmbedding],
                                                pd.Series]]] \
            = results
        """The dict storage of the managed results.
        Format: ``{layer_id: {run: ([embedding1, embedding2, ...],
        results_series)}}``."""

    def items(self) -> ItemsView[str, 'AnalysisResult']:
        """Emulate an items view that yields an analysis result per layer ID."""
        return {layer_id: AnalysisResult({layer_id: self.results[layer_id]})
                for layer_id in self.results.keys()}.items()

    def result_for(self, layer_id: str) -> 'AnalysisResult':
        """Return the results for a single layer."""
        return AnalysisResult({layer_id: self.results[layer_id]})

    def save(self, folder_path: str):
        """Save analysis results.
        The format is one retrievable by :py:meth:`load`.
        The results are saved in the following files within ``folder_path``

        - ``<layer> <run> <i>.pt``: torch PT file with ith embedding resulting
          from ``<run>`` on ``<layer>``; can be loaded to an embedding using
          :py:meth:`hybrid_learning.concepts.models.embeddings.ConceptEmbedding.load`
        - ``stats.csv``: CSV file holding a :py:class:`pandas.DataFrame` with
          each rows holding an embedding statistics;
          additional columns are ``'layer'``, ``'run'``, and
          ``'embedding_{i}'``, where the ``'embedding_{i}'`` column holds the
          path to the ith PT-saved embedding corresponding of the row
          relative to the location of ``stats.csv``

        .. note::
            Also the .npz legacy format is accepted and determined from the
            file ending.

        :param folder_path: the root folder to save files under;
            must not yet exist
        """
        info = self.to_pandas()
        info.index.names = ['layer', 'run']
        for layer, run in info.index:
            for i, emb in enumerate(self.results[layer][run][0]):
                emb_fn = f"{layer} {run} {i}.pt"
                # Save and note in the info frame:
                emb.save(os.path.join(folder_path, emb_fn))
                info.loc[(layer, run), f'embedding_{i}'] = emb_fn
        info.reset_index(inplace=True)
        info.to_csv(os.path.join(folder_path, "stats.csv"))

    @classmethod
    def load(cls, folder_path: str) -> 'AnalysisResult':
        """Load analysis results previously saved.
        The saving format is assumed to be that of :py:meth:`save`."""
        if not os.path.isdir(folder_path):
            raise ValueError("Folder {} does not exist!".format(folder_path))
        info: pd.DataFrame = pd.read_csv(os.path.join(folder_path, "stats.csv"))
        if all([col not in info.columns for col in ("layer", "run")]):
            info.rename(columns={'Unnamed: 0': "layer", 'Unnamed: 1': "run"},
                        inplace=True)
        assert all([col in info.columns for col in ("layer", "run")])
        emb_cols: List[str] = sorted([col for col in info.columns
                                      if 'embedding' in col])
        assert len(emb_cols) > 0
        info.set_index(['layer', 'run'], inplace=True)
        layers = info.index.get_level_values('layer').unique()
        runs = info.index.get_level_values('run').unique()
        analysis_results = {layer: {run: None for run in runs}
                            for layer in layers}
        for layer in layers:
            for run in runs:
                row: pd.Series = info.loc[(layer, run)]
                embs: List[ConceptEmbedding] = [ConceptEmbedding.load(
                    os.path.join(folder_path, row[emb_col]))
                    for emb_col in emb_cols]
                stat = row[row.index.difference(emb_cols)].to_dict()
                analysis_results[layer][run] = (embs, stat)
        return cls(analysis_results)

    def to_pandas(self):
        """Provide :py:class:`pandas.DataFrame` multi-indexed by layer and
        run w/ info for each run.
        The information for each run is the one obtained by
        :py:meth:`~ResultsHandle.emb_info_to_pandas`.

        :returns: a :py:class:`pandas.DataFrame` with run result information
            multi-indexed by ``(layer, run)``
        """
        return pd.DataFrame({(layer_id, run): self.emb_info_to_pandas(embs,
                                                                      stats)
                             for layer_id, runs in self.results.items()
                             for run, (embs, stats) in runs.items()
                             }).transpose()


class BestEmbeddingResult(ResultsHandle):
    """Handle for results on layer-wise reduction or analysis results to
    best embeddings.
    The handle can save and load results, as well as provide different
    representations (see :py:meth:`to_pandas`).
    The results are saved in :py:attr:`results`. See there for the format.
    """

    def __init__(self,
                 results: Dict[str, Tuple[ConceptEmbedding,
                                          Tuple[np.ndarray, float, float],
                                          pd.Series]]):
        self.results: Dict[str, Tuple[ConceptEmbedding,
                                      Tuple[np.ndarray, float, float],
                                      pd.Series]] = results
        """The actual results dictionary of the form ``{layer_id: info_tuple}``
        where the ``info_tuple`` holds:

        - the best concept embedding of the layer,
        - the standard deviation results,
        - the metric results when evaluated on its concept
        """

    def to_pandas(self) -> pd.DataFrame:
        """Provide :py:class:`pandas.DataFrame` indexed by layer ID wt/ info
        about embeddings."""
        return pd.DataFrame({
            layer_id: self.emb_info_to_pandas(emb, stats, std)
            for layer_id, (emb, std, stats) in self.results.items()
        }).transpose()

    def save(self, folder_path: str):
        r"""Save results of embedding reduction.
        The following is saved:

        - embeddings as
          ``folder_path/layer_id\ best.pt``
        - merged stats and standard deviation info as
          ``folder_path/best_emb_stats.csv``
        """
        info = self.to_pandas()
        info.index.names = ['layer']
        info['embedding'] = None
        for layer in info.index:
            emb: ConceptEmbedding = self.results[layer][0]
            emb_fn = "{} best.pt".format(layer)
            # Save and note in the info frame:
            emb.save(os.path.join(folder_path, emb_fn))
            info.loc[layer, 'embedding'] = emb_fn
        info.reset_index(inplace=True)
        info.to_csv(os.path.join(folder_path, "best_emb_stats.csv"))

    @classmethod
    def load(cls, folder_path: str) -> 'BestEmbeddingResult':
        """Load previously saved results for best embeddings.
        Note that the standard deviation information cannot be fully
        retrieved, as the standard deviation vector is replaced by its
        length during saving."""
        if not os.path.isdir(folder_path):
            raise ValueError("Folder {} does not exist!".format(folder_path))
        info: pd.DataFrame = \
            pd.read_csv(os.path.join(folder_path, "best_emb_stats.csv"))
        if "layer" not in info.columns:
            if 'index' in info.columns:
                info.rename(columns={'index': "layer"}, inplace=True)
            elif 'Unnamed: 0' in info.columns:
                info.rename(columns={'Unnamed: 0': "layer"}, inplace=True)
        assert all([col in info.columns
                    for col in ("layer", "embedding")])
        info.set_index('layer', inplace=True)

        layers = info.index.unique()
        results = {layer: None for layer in layers}
        for layer in layers:
            row: pd.Series = info.loc[layer]
            emb = ConceptEmbedding.load(
                os.path.join(folder_path, row['embedding']))
            stats_std = row[row.index.difference(['embedding'])]
            std = row[[col for col in stats_std.index if col.startswith('std')]]
            stats = stats_std[stats_std.index.difference(std.index)]
            results[layer] = (emb, std, stats)
        return cls(results)
