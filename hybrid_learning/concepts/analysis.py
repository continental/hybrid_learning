"""Concept analysis functionality.

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
#  Copyright (c) 2020 Continental Automotive GmbH

import enum
import logging
import os
from typing import Tuple, Dict, Any, Sequence, Callable, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from hybrid_learning.datasets import data_visualization as datavis
from . import visualization as vis
from .concepts import ConceptTypes, Concept, SegmentationConcept2D
from .embeddings import ConceptEmbedding
# For type hints:
from .models import ConceptDetectionModel2D, ConceptDetection2DTrainTestHandle

LOGGER = logging.getLogger(__name__)


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


class ConceptAnalysis:
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

    def __init__(self,
                 concept: Concept,
                 model: torch.nn.Module,
                 layer_infos: Union[Dict[str, Dict[str, Any]],
                                    Sequence[str]] = None,
                 cross_val_runs: int = 1,
                 num_val_splits: int = 5,
                 emb_reduction: EmbeddingReduction = EmbeddingReduction.DEFAULT,
                 show_train_progress_bars: bool = True,
                 concept_model_args: Dict[str, Any] = None,
                 train_val_args: Dict[str, Any] = None,
                 ):
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
        :param show_train_progress_bars: whether to show the training
            progress bars of the models
        :param concept_model_args: dict with arguments for the concept model
            initialization
        :param train_val_args: any further arguments to initialize the concept
            model handle
        """
        if not concept.type == ConceptTypes.SEGMENTATION:
            raise NotImplementedError(
                ("Analysis only available for segmentation concepts,"
                 "but concept was of type {}").format(concept.type))
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
        """The number of validation splits per cross-validation run."""
        self.emb_reduction: EmbeddingReduction = emb_reduction
        """Aggregation function to reduce a list of embeddings from several
        runs to one."""
        self.show_train_progress_bars: bool = show_train_progress_bars
        """Whether to show the training progress bars of the models"""
        self.train_val_args: Dict[str, Any] = train_val_args \
            if train_val_args is not None else {}
        """Any training and evaluation arguments for the concept model
        initialization."""
        self.concept_model_args: Dict[str, Any] = concept_model_args \
            if concept_model_args is not None else {}
        """Any arguments for initializing a new concept model."""

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
            show_train_progress_bars=self.show_train_progress_bars,
            train_val_args=self.train_val_args,
            concept_model_args=self.concept_model_args
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
        analysis_results = analysis_results or self.analysis()
        best_embs_stds_stats: Dict[
            str, Tuple[ConceptEmbedding, Tuple, pd.Series]] = {}
        for layer_id, results_per_run in analysis_results.items():
            best_embs_stds_stats[layer_id] = \
                self.embedding_reduction(results_per_run)
        best_layer_id = self.best_layer_from_stats(best_embs_stds_stats)
        LOGGER.info("Concept %s final layer: %s", self.concept.name,
                    best_layer_id)
        best_embedding, _, _ = best_embs_stds_stats[best_layer_id]
        return best_embedding

    def analysis(self) -> Dict[str,
                               Dict[int, Tuple[ConceptEmbedding, pd.Series]]]:
        """Conduct a concept embedding analysis.

        For each layer in :py:attr:`layer_infos`:

        - train :py:attr:`cross_val_runs` x :py:attr:`num_val_splits`
          concept models,
        - collect their evaluation results,
        - convert them to embeddings.

        :return: a dictionary of
            ``{layer_id: {run: (embedding,
            pandas.Series with {pre_: metric_val}}}``
        """
        results_per_layer: Dict[
            str, Dict[int, Tuple[ConceptEmbedding, pd.Series]]] = {}
        for layer_id in self.layer_infos:
            results_per_run: Dict[int, Tuple[ConceptEmbedding, pd.Series]] = \
                self.analysis_for_layer(layer_id)
            results_per_layer[layer_id] = results_per_run
        return results_per_layer

    @classmethod
    def analysis_results_to_pandas(cls, analysis_results):
        """Provide :py:class:`pandas.DataFrame` multi-indexed by layer and
        run w/ info for each run.
        The information for each run is the one obtained by
        :py:meth:`emb_info_to_pandas`.

        :param analysis_results: analysis results in the for as produced by
            :py:meth:`analysis`
        :returns: a :py:class:`pandas.DataFrame` with run result information
            multi-indexed by ``(layer, run)``
        """
        return pd.DataFrame({(layer_id, run): cls.emb_info_to_pandas(emb, stats)
                             for layer_id, runs in analysis_results.items()
                             for run, (emb, stats) in runs.items()
                             }).transpose()

    @classmethod
    def best_emb_infos_to_pandas(cls,
                                 results: Dict[str, Tuple[
                                     ConceptEmbedding,
                                     Tuple[np.ndarray, float, float],
                                     pd.Series]]) -> pd.DataFrame:
        """Provide :py:class:`pandas.DataFrame` indexed by layer ID wt/ info
        about embeddings.
        The format of results must be a dictionary indexed by the layer ID
        and with values as provided by :py:meth:`embedding_reduction`
        """
        return pd.DataFrame({layer_id: cls.emb_info_to_pandas(emb, stats, var)
                             for layer_id, (emb, var, stats) in results.items()
                             }).transpose()

    @classmethod
    def save_best_emb_results(
            cls,
            results: Dict[str, Tuple[ConceptEmbedding,
                                     Tuple[np.ndarray, float, float],
                                     pd.Series]],
            folder_path: str):
        """Save results of embedding reduction.
        The format of results must be a dict with layer IDs as keys and
        values as provided by :py:meth:`embedding_reduction`.
        """
        info = cls.best_emb_infos_to_pandas(results)
        info['embedding'] = None
        for layer in info.index:
            emb: ConceptEmbedding = results[layer][0]
            emb_fn = "{} best.npz".format(layer)
            # Save and note in the info frame:
            emb.save(os.path.join(folder_path, emb_fn))
            info.loc[layer, 'embedding'] = emb_fn
        info.to_csv(os.path.join(folder_path, "best_emb_stats.csv"))

    @classmethod
    def save_analysis_results(cls,
                              results: Dict[str, Dict[
                                  int, Tuple[ConceptEmbedding, pd.Series]]],
                              folder_path: str):
        """Save analysis results.
        The format is one retrievable by :py:meth:`load_analysis_results`.
        The results are saved in the following files within ``folder_path``

        - ``<layer> <run>.npz``: npz file with embedding resulting from
          ``<run>`` on ``<layer>``; can be loaded to an embedding using
          :py:meth:`hybrid_learning.concepts.embeddings.ConceptEmbedding.load`
        - ``stats.csv``: CSV file holding a :py:class:`pandas.DataFrame` with
          each rows holding an embedding statistics;
          additional columns are ``'layer'``, ``'run'``, and ``'embedding'``,
          where the ``'embedding'`` column holds the path to the npz-saved
          embedding corresponding of the row relative to the location of
          ``stats.csv``

        :param results: results dictionary in the format returned by
            :py:meth:`analysis`
        :param folder_path: the root folder to save files under;
            must not yet exist
        """
        info = cls.analysis_results_to_pandas(results)
        info['embedding'] = None
        for layer, run in info.index:
            emb: ConceptEmbedding = results[layer][run][0]
            emb_fn = "{} {}.npz".format(layer, run)
            # Save and note in the info frame:
            emb.save(os.path.join(folder_path, emb_fn))
            info.loc[(layer, run), 'embedding'] = emb_fn
        info.to_csv(os.path.join(folder_path, "stats.csv"))

    @staticmethod
    def load_analysis_results(folder_path: str
                              ) -> Dict[str, Dict[int, Tuple[ConceptEmbedding,
                                                             pd.Series]]]:
        """Load analysis results previously saved.
        The saving format is assumed to be that of
        :py:meth:`save_analysis_results`."""
        if not os.path.isdir(folder_path):
            raise ValueError("Folder {} does not exist!".format(folder_path))
        stats_frame = pd.read_csv(os.path.join(folder_path, "stats.csv"))
        assert all([col in stats_frame.columns
                    for col in ("layer", "run", "embedding")])
        stats_frame.set_index(['layer', 'run'])
        layers = stats_frame.index.get_level_values('layer').unique()
        runs = stats_frame.index.get_level_values('run').unique()
        analysis_results = {layer: {run: None for run in runs}
                            for layer in layers}
        for layer in layers:
            for run in runs:
                row = stats_frame.loc[(layer, run)]
                emb = ConceptEmbedding.load(
                    os.path.join(folder_path, row['embedding']))
                stat = row.drop('embedding', axis=1)
                analysis_results[layer][run] = (emb, stat)
        return analysis_results

    def analysis_for_layer(self, layer_id: str
                           ) -> Dict[int, Tuple[ConceptEmbedding, pd.Series]]:
        """Get a concept embedding of the given concept in the given layer.

        :param layer_id: ID of the layer to find embedding in; key in
            :py:attr:`layer_infos`
        :return: a tuple of the best found embedding, the standard deviation,
            and its performance
        """
        c_model = self.concept_model_for_layer(layer_id)
        c_handle: ConceptDetection2DTrainTestHandle = \
            self.concept_model_handle(c_model)
        if 'lr' in self.layer_infos[layer_id]:
            c_handle.optimizer.lr = self.layer_infos[layer_id]['lr']
        stats_per_run = {}
        for cross_val_run in range(self.cross_val_runs):
            states, _, _ = zip(*c_handle.cross_validate(
                num_splits=self.num_val_splits,
                run_info_templ=("{}, cv {}/{}, ".format(
                    layer_id, cross_val_run + 1, self.cross_val_runs) +
                                "run {run}/{runs}"),
                show_progress_bars=self.show_train_progress_bars))
            for split, state_dict in enumerate(states):
                c_model.load_state_dict(state_dict)
                embedding = c_model.to_embedding()
                metrics: pd.Series = self.evaluate_embedding(embedding)
                # storing & logging
                run = split + cross_val_run * self.num_val_splits
                stats_per_run[run] = (embedding, metrics)
                context = "Concept {}, layer {}, run {}".format(
                    self.concept.name, layer_id, run)
                LOGGER.info("%s:\n%s", context,
                            self.emb_info_to_string(embedding, metrics))

        return stats_per_run

    def concept_model_handle(self,
                             c_model: ConceptDetectionModel2D = None,
                             emb: ConceptEmbedding = None,
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
        return ConceptDetection2DTrainTestHandle(c_model, **self.train_val_args)

    def concept_model_for_layer(self, layer_id):
        """Return a concept model for the given layer ID.

        :param layer_id: ID of the layer the concept model should be attached
            to; key in :py:attr:`layer_infos`
        :returns: concept model for :py:attr:`concept` attached to given
            layer in :py:attr:`model`
        """
        c_model: ConceptDetectionModel2D = ConceptDetectionModel2D(
            concept=SegmentationConcept2D.new(self.concept),
            model=self.model, layer_id=layer_id,
            **{'kernel_size': self.layer_infos[layer_id].get('kernel_size',
                                                             None),
               **self.concept_model_args}

        )
        return c_model

    @staticmethod
    def concept_model_from_embedding(embedding: ConceptEmbedding
                                     ) -> ConceptDetectionModel2D:
        """Get concept model from embedding for training and eval."""
        return ConceptDetectionModel2D.from_embedding(embedding)

    @staticmethod
    def emb_info_to_string(
            emb: ConceptEmbedding, stats: pd.Series = None,
            std_dev: Tuple[np.ndarray, float, float] = None) -> str:
        """Printable quick info about the given embedding with stats
        (and standard deviation)."""
        info: pd.Series = ConceptAnalysis.emb_info_to_pandas(emb, stats,
                                                             std_dev=std_dev)
        # Formatting
        float_format: str = "{: < 14.6f}"
        exp_format: str = "{: < 14.6e}"
        for idx in [i for i in info.index if "std" in i]:
            info[idx] = exp_format.format(info[idx])
        return info.to_string(float_format=float_format.format)

    @staticmethod
    def emb_info_to_pandas(emb: ConceptEmbedding, stats: pd.Series = None,
                           std_dev: Tuple[np.ndarray, float, float] = None
                           ) -> pd.Series:
        """Quick info about embedding with stats (and standard dev)
        as :py:class:`pandas.Series`."""
        stats_info = stats if stats is not None else {}
        emb_info = {"normal vec len": np.linalg.norm(emb.normal_vec),
                    "support factor": float(emb.support_factor),
                    "scaling factor": float(emb.scaling_factor)}
        std_info = {"std dev normal vec (len)": np.linalg.norm(std_dev[0]),
                    "std dev support factor": std_dev[1],
                    "std dev scaling factor": std_dev[2]} \
            if std_dev is not None else {}
        return pd.Series({**stats_info, **emb_info, **std_info})

    def evaluate_embedding(self, embedding: ConceptEmbedding):
        """Evaluate the embedding on its concept test data."""
        # Value check:
        if not embedding.concept.type == ConceptTypes.SEGMENTATION:
            raise NotImplementedError(
                ("Routine currently only available for segmentation concepts,"
                 "but concept was of type {}").format(embedding.concept.type))

        # Evaluation:
        with torch.no_grad():
            eval_model = self.concept_model_from_embedding(embedding)
            stats: pd.Series = self.concept_model_handle(eval_model).evaluate()
        return stats

    def embedding_reduction(
            self,
            results_per_run: Dict[int, Tuple[ConceptEmbedding, pd.Series]]
    ) -> Tuple[ConceptEmbedding, Tuple[np.ndarray, float, float], pd.Series]:
        """Aggregate the embeddings collected in ``results_per_run``
        to a best one.
        This is a wrapper with standard deviation and stats collection and
        logging around a call to :py:func:`emb_reduction`.

        :param results_per_run: dictionary indexed by different runs to
            obtain a concept embedding in the same setup
            (layer, concept, etc.); values are tuples of:

            - result embedding
            - metrics results on the concept test set for that embedding

        :return: a tuple of

            - an aggregated ("mean") embedding for the concept and the layer,
            - the standard deviation values of the normal vectors,
            - the stats for the chosen "mean" embedding
        """
        if len(results_per_run) == 0:
            raise ValueError("Empty results dict")
        layer_id: str = \
            results_per_run[list(results_per_run.keys())[0]][0].layer_id
        embeddings = [e for e, _ in results_per_run.values()]
        best_embedding = self.emb_reduction(embeddings)

        # Variance and stats collection:
        std_dev: Tuple[np.ndarray, float, float] = \
            ConceptEmbedding.std_deviation(embeddings)
        stats: pd.Series = self.evaluate_embedding(best_embedding)

        # Some logging:
        LOGGER.info("Concept %s, layer %s:\n%s",
                    self.concept.name, layer_id,
                    self.emb_info_to_string(best_embedding, stats,
                                            std_dev=std_dev))

        return best_embedding, std_dev, stats

    @staticmethod
    def best_layer_from_stats(
            results_per_layer: Dict[
                str, Tuple[ConceptEmbedding, Tuple, pd.Series]]) -> str:
        """From the embedding quality results per layer, select the best layer.
        For segmentation concepts, select by set IoU.

        :param results_per_layer: tuple of

            - the best concept embedding of the layer,
            - the standard deviation results,
            - the metric results when evaluated on its concept

        :return: layer ID with best stats
        """
        # DataFrame with layer-wise metric results
        # (col: layer_id, idx: metric_name)
        test_set_iou_key = ConceptDetection2DTrainTestHandle.test_("set_iou")
        stats = pd.DataFrame({l_id: info[-1]
                              for l_id, info in results_per_layer.items()})
        if test_set_iou_key not in stats.index:
            raise KeyError(
                ("KPI key {} not in stats keys {}; Wrong concept type used?"
                 " (currently only segmentation concepts allowed)"
                 ).format(test_set_iou_key, stats.index))

        best_layer_id = stats.loc[test_set_iou_key].idxmax()
        return str(best_layer_id)

    def train_data_infos(self) -> pd.DataFrame:
        """Provide a DataFrame with some information on how each layer."""
        layer_infos = {}
        for layer_id in self.layer_infos:
            c_model = self.concept_model_for_layer(layer_id)
            c_handle = self.concept_model_handle(c_model)
            layer_infos[layer_id] = {
                'kernel_size': c_model.kernel_size,
                'prop_neg_px': datavis.neg_pixel_prop(c_handle.data.train)}
        return pd.DataFrame(layer_infos).transpose()

    def best_embedding_with_logging(
            self,
            concept_exp_root: str,
            logger: logging.Logger = None,
            file_logging_formatter: logging.Formatter = None,
            log_file: str = 'log.txt',
            img_fp_templ: Optional[str] = "{}.png"
    ) -> ConceptEmbedding:
        # TODO: properly separate saving & loading from analysis
        """Conduct an analysis, collect mean and best embeddings,
        and save and log all results.

        .. rubric:: Saved results

        - the embedding of each layer and run as .npz file;
          for format see
          :py:meth:`hybrid_learning.concepts.embeddings.ConceptEmbedding.save`;
          load with
          :py:meth:`hybrid_learning.concepts.embeddings.ConceptEmbedding.load`
        - the aggregated (best) embedding for each layer as .npz file
          (see above)
        - the final best embedding amongst all layers as .npz file
          (chosen from above best embeddings)
        - statistics of the runs for each layer incl. evaluation results and
          infos on final embedding obtained by each run;
          for format see :py:meth:`save_analysis_results`;
          load with :py:meth:`ConceptAnalysis.load_analysis_results`
        - statistics for the aggregated (best) embeddings;
          for format see :py:meth:`ConceptAnalysis.save_best_emb_results`;

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
        :return: the found best embedding for that part
        """
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
                        self.train_data_infos())
            if save_imgs:
                datavis.visualize_mask_transforms(
                    {layer_id: self.concept_model_handle(
                        layer_id=layer_id).data.train
                     for layer_id in self.layer_infos},
                    save_as=save_as("vis_train_data_transforms"))
            # endregion

            # Analysis:
            analysis_results = self.analysis()
            self.save_analysis_results(analysis_results, concept_exp_root)
            logger.info("Embedding results per run:\n%s",
                        self.analysis_results_to_pandas(analysis_results))

            # Best embedding selection:
            best_embs_results: Dict[str,
                                    Tuple[ConceptEmbedding, Tuple, pd.Series]] \
                = {layer_id: self.embedding_reduction(results_per_run)
                   for layer_id, results_per_run in analysis_results.items()}
            self.save_best_emb_results(best_embs_results, concept_exp_root)
            best_emb_infos = self.best_emb_infos_to_pandas(best_embs_results)
            logger.info("Best embeddings per layer:\n%s", best_emb_infos)

            # The very best embedding:
            # Save it twice to find it more easily and store in best_embs
            best_layer_id = self.best_layer_from_stats(best_embs_results)
            best_layer_embs: List[ConceptEmbedding] = \
                [e for e, stats in analysis_results[best_layer_id].values()]
            best_embedding: ConceptEmbedding = \
                best_embs_results[best_layer_id][0]
            best_embedding.save(os.path.join(concept_exp_root, "best.npz"))
            logger.info("Best embedding:\n%s",
                        best_emb_infos.loc[best_layer_id])

            # pair-wise cosines with last row and column the best_embedding:
            pairwise_cos = vis.pairwise_cosines(
                embs=best_layer_embs + [best_embedding],
                keys=list(range(len(best_layer_embs))) + ['best_emb'])
            pairwise_cos.to_csv(
                os.path.join(concept_exp_root, 'pairwise_cosines.csv'))
            logger.info(
                "Mean cosine dist of best_embedding to other embeddings:\n%s",
                pairwise_cos.iloc[:-1, -1].mean())
            logger.info(
                'Pair-wise cosine dist between normal vectors of runs in '
                'best layer:\n%s',
                pairwise_cos)

        # visualizations:
        if save_imgs:
            vis.visualize_concept_model(
                self.concept_model_handle(emb=best_embedding),
                save_as=save_as("vis_best_embedding"))
            vis.visualize_concept_models(
                {**{"best": self.concept_model_handle(emb=best_embedding)},
                 **{"emb {}".format(i): self.concept_model_handle(emb=e)
                    for i, e in enumerate(best_layer_embs)}},
                save_as=save_as("vis_best_layer_embeddings"))
            vis.visualize_concept_models(
                {layer_id: self.concept_model_handle(emb=e)
                 for layer_id, (e, _, _) in best_embs_results.items()},
                save_as=save_as("vis_best_embeddings"))

        # Close part specific logging
        logger.removeHandler(part_log_file_handler)
        logger.setLevel(orig_logging_level)

        return best_embedding
