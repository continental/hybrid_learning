#  Copyright (c) 2022 Continental Automotive GmbH
"""Test functions and fixtures for concept analysis."""

# pylint: disable=no-self-use
# pylint: disable=redefined-outer-name
import os
from typing import Dict, Any, Tuple, List

import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch

from hybrid_learning.concepts.train_eval.kpis import batch_kpis, aggregating_kpis
from hybrid_learning.concepts.analysis import ConceptAnalysis, AnalysisResult, \
    BestEmbeddingResult
from hybrid_learning.concepts.concepts import Concept
from hybrid_learning.concepts.models import ConceptDetection2DTrainTestHandle, ConceptEmbedding
# pylint: disable=unused-import
# noinspection PyUnresolvedReferences
from .common_fixtures import concept, main_model, input_size


# pylint: enable=unused-import

@pytest.fixture
def layer_infos() -> Dict[str, Dict[str, Any]]:
    """Yield the layer infos for concept analysis on the given fixture
    main_model."""
    return {'features.5': {},
            'features.9': {}}


@pytest.fixture
def analyser(concept: Concept, layer_infos: Dict[str, Dict[str, Any]],
             main_model: torch.nn.Module) -> ConceptAnalysis:
    """Return a basic analyser object in which to test methods.
    The analyser is run on a simplistic test/train set.
    """
    analyser = ConceptAnalysis(
        concept=concept, model=main_model, layer_infos=layer_infos,
        cross_val_runs=2, num_val_splits=2,
        train_val_args=dict(max_epochs=1,
                            metric_fns=dict(mean_iou=batch_kpis.IoU(),
                                            set_iou=aggregating_kpis.SetIoU(),
                                            acc=aggregating_kpis.Accuracy())),
        data_args=dict(cache_in_memory=True),
    )
    assert list(analyser.layer_infos.keys()) == list(layer_infos.keys())
    assert 'loss_fn' in analyser.train_val_args
    return analyser


class TestConceptAnalyzer:
    """Test functions for a concept analyser.
    See analyser() for the analyser object used."""

    def test_single_run(self, analyser: ConceptAnalysis):
        """Test whether the analysis can be called with just one run per
        layer."""
        # Analyse 1 layer in 1 run:
        layer = list(analyser.layer_infos.keys())[0]
        analyser.layer_infos = {layer: analyser.layer_infos[layer]}
        analyser.cross_val_runs = 1
        analyser.num_val_splits = 1
        results = analyser.analysis()
        assert list(results.results[layer].keys()) == [0]

    def test_best_embedding(self, analyser: ConceptAnalysis):
        """Test a run of a ConceptAnalysis."""

        embedding: ConceptEmbedding = analyser.best_embedding()

        # Correct references set within embedding?
        assert embedding.concept.name == analyser.concept.name
        assert embedding.concept.train_data == analyser.concept.train_data
        assert embedding.concept.test_data == analyser.concept.test_data
        assert embedding.main_model is analyser.model

    def test_laplace(self, analyser: ConceptAnalysis):
        """Test a run with use_laplace=True of the concept model."""
        analyser.concept_model_args.update(use_laplace=True)

    def test_concept_analysis(self, analyser: ConceptAnalysis):
        """Test a run of a concept analysis."""
        results = analyser.analysis()

        # All layers considered?
        assert sorted(results.results.keys()) == \
               sorted(analyser.layer_infos.keys())
        for layer, layer_results in results.items():
            # Value format
            assert isinstance(layer_results, AnalysisResult), \
                ("Result value at {} not an AnalysisResult but {}"
                 .format(layer, type(layer_results)))
            assert isinstance(layer_results.results, Dict), \
                ("AnalysisResult holding value at {} not a dict but {}"
                 .format(layer, type(layer_results)))
            for run, run_results in layer_results.results[layer].items():
                context = ("At layer {}, run {}, result value {}"
                           .format(layer, run, run_results))
                assert len(run_results) == 2, "{} not of len 2 but {}".format(
                    context, len(run_results))
                # Value entry types
                assert isinstance(run_results[0][0], ConceptEmbedding), \
                    ("{} entry 0 not a ConceptEmbedding but {}"
                     .format(context, type(run_results[0])))
                assert isinstance(run_results[1], pd.Series), \
                    ("{} entry 0 not a Series but {}"
                     .format(context, type(run_results[1])))
                # Stats format
                test_set_iou_key = \
                    ConceptDetection2DTrainTestHandle.test_("set_iou")
                assert test_set_iou_key in list(run_results[1].index), \
                    "{} entry 1 index does not have key {} (keys: {})".format(
                        context, test_set_iou_key, run_results[1].index)

    def test_analysis_wt_saving(self, analyser: ConceptAnalysis, tmp_path: str):
        """Test a run of a concept analysis with results saving."""
        matplotlib.use("Agg")  # prevent from using uninstalled tkinter

        best_embedding = analyser.best_embedding_with_logging(tmp_path)
        assert isinstance(best_embedding, ConceptEmbedding)
        assert os.path.isfile(os.path.join(tmp_path, 'log.txt'))
        assert os.path.isfile(os.path.join(tmp_path, "vis_best_embedding.png"))
        assert os.path.isfile(
            os.path.join(tmp_path, "vis_best_layer_embeddings.png"))
        assert os.path.isfile(os.path.join(tmp_path, "vis_best_embeddings.png"))

    def test_aggregate_embedding(self, analyser: ConceptAnalysis):
        """Control format of aggregate_embedding output."""
        layer_id: str = list(analyser.layer_infos.keys())[0]
        layer_results: AnalysisResult = analyser.analysis_for_layer(layer_id)
        emb, std_dev, stats = \
            analyser.embedding_reduction(layer_results).results[layer_id]

        # Evaluation results reproducible?
        new_stats: pd.Series = analyser.evaluate_embedding(emb)
        assert np.allclose(stats, new_stats.loc[stats.index]), \
            (("For layer {} the best embedding {} evaluation results are not "
              "reproducible:\nembedding_reduction:\n{}\nevaluate_embedding:\n{}"
              ).format(layer_id, emb, stats, new_stats))

        # Standard deviation information sensible and of correct format?
        assert isinstance(std_dev, tuple)
        assert len(std_dev) == 3
        assert isinstance(std_dev[0], np.ndarray) and np.all(std_dev[0] >= 0)
        assert isinstance(std_dev[1], float) and std_dev[1] >= 0
        assert isinstance(std_dev[2], float) and std_dev[2] >= 0

        # stats format
        assert isinstance(stats, pd.Series)
        assert len(stats) >= 1
        test_set_iou_key = ConceptDetection2DTrainTestHandle.test_("set_iou")
        assert test_set_iou_key in list(stats.index), \
            "Index does not have key {} (keys: {})".format(test_set_iou_key,
                                                           stats.index)


class TestAnalysisResult:
    """Test the functionality of the analysis results handle."""

    @pytest.fixture
    def dummy_result(self) -> AnalysisResult:
        """A default analysis result."""
        emb: ConceptEmbedding = ConceptEmbedding(
            state_dict=dict(normal_vec=np.array([1, 0]),
                            bias=np.array(1)),
            normal_vec_name="normal_vec", bias_name="bias",
            kernel_size=(1,))
        res: pd.Series = pd.Series({'test_set_iou': 0.5})
        return AnalysisResult({'dummy_layer': {0: ([emb], res)}})

    def test_to_pandas(self, dummy_result: AnalysisResult):
        """Test the to_pandas functionality."""
        res_pd: pd.DataFrame = dummy_result.to_pandas()

        assert isinstance(res_pd, pd.DataFrame)
        assert isinstance(res_pd.index, pd.MultiIndex)
        assert res_pd.index.get_level_values(0) == ['dummy_layer']
        assert res_pd.index.get_level_values(1) == [0]
        assert 'test_set_iou' in res_pd.columns

    def test_save(self, dummy_result: AnalysisResult, tmp_path):
        """Test the saving functionality."""
        dummy_result.save(str(tmp_path))

        assert os.path.exists(os.path.join(tmp_path, "stats.csv"))
        assert os.path.exists(os.path.join(
            tmp_path, "{} {} {}.pt".format('dummy_layer', 0, 0)))

        stats = pd.read_csv(os.path.join(tmp_path, "stats.csv"))
        assert "layer" in stats.columns
        assert "run" in stats.columns
        assert any("embedding" in col for col in stats.columns)
        assert "test_set_iou" in stats.columns
        assert stats["test_set_iou"].iloc[0] == 0.5

    def test_load(self, dummy_result: AnalysisResult, tmp_path):
        """Test the loading functionality."""
        dummy_result.save(str(tmp_path))

        loaded_res = AnalysisResult.load(str(tmp_path))
        assert list(loaded_res.results.keys()) == ['dummy_layer']
        assert list(loaded_res.results['dummy_layer'].keys()) == [0]
        assert loaded_res.results['dummy_layer'][0][0] == \
               dummy_result.results['dummy_layer'][0][0]
        assert loaded_res.results['dummy_layer'][0][1]['test_set_iou'] == 0.5


class TestBestEmbeddingResult:
    """Test the functionality of the best embedding results handle."""
    STANDARD_COLS: List[str] = [
        'test_set_iou', 'normal vec len', 'support factor', 'scaling factor',
        'std dev normal vec (len)', 'std dev support factor',
        'std dev scaling factor']

    @pytest.fixture
    def dummy_result(self) -> BestEmbeddingResult:
        """A default analysis result."""
        emb: ConceptEmbedding = ConceptEmbedding(
            state_dict=dict(normal_vec=np.array([1, 0]),
                            bias=np.array(1)),
            normal_vec_name="normal_vec", bias_name="bias",
            kernel_size=(1,))
        std: Tuple[np.ndarray, float, float] = (np.array([0.1, 0.2]), 0.3, 0.4)
        res: pd.Series = pd.Series({'test_set_iou': 0.5})
        return BestEmbeddingResult({'dummy_layer': (emb, std, res)})

    def test_to_pandas(self, dummy_result: BestEmbeddingResult):
        """Test the to_pandas functionality."""

        res_pd: pd.DataFrame = dummy_result.to_pandas()

        assert isinstance(res_pd, pd.DataFrame)
        assert isinstance(res_pd.index, pd.Index)
        assert list(res_pd.index) == ['dummy_layer']
        for col_name in self.STANDARD_COLS:
            assert any(col_name in col for col in res_pd.columns)

    def test_save(self, dummy_result: BestEmbeddingResult, tmp_path):
        """Test the saving functionality."""
        dummy_result.save(str(tmp_path))

        assert os.path.exists(os.path.join(tmp_path, "best_emb_stats.csv"))
        assert os.path.exists(os.path.join(
            tmp_path, "{} best.pt".format('dummy_layer')))

        stats = pd.read_csv(os.path.join(tmp_path, "best_emb_stats.csv"))
        assert "layer" in stats.columns
        assert "embedding" in stats.columns
        for col_name in self.STANDARD_COLS:
            assert any(col_name in col for col in stats.columns)
        assert stats["test_set_iou"].iloc[0] == 0.5

    def test_load(self, dummy_result: BestEmbeddingResult, tmp_path):
        """Test the loading functionality."""
        dummy_result.save(str(tmp_path))

        loaded_res: BestEmbeddingResult = \
            BestEmbeddingResult.load(str(tmp_path))
        assert list(loaded_res.results.keys()) == ['dummy_layer']
        assert loaded_res.results['dummy_layer'][0] == \
               dummy_result.results['dummy_layer'][0]
        assert loaded_res.results['dummy_layer'][2]['test_set_iou'] == 0.5
