"""Test functions and fixtures for concept analysis."""
#  Copyright (c) 2020 Continental Automotive GmbH

# pylint: disable=no-self-use
# pylint: disable=redefined-outer-name
import os
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import pytest
import torch

from hybrid_learning.concepts.analysis import ConceptAnalysis
from hybrid_learning.concepts.concepts import Concept
from hybrid_learning.concepts.embeddings import ConceptEmbedding
from hybrid_learning.concepts.models import ConceptDetection2DTrainTestHandle
# pylint: disable=unused-import
# noinspection PyUnresolvedReferences
from .common_fixtures import train_concept, main_model


# pylint: enable=unused-import

@pytest.fixture
def layer_infos() -> Dict[str, Dict[str, Any]]:
    """Yield the layer infos for concept analysis on the given fixture
    main_model."""
    return {'backbone.body.layer3': {},
            'backbone.body.layer4': {}}


@pytest.fixture
def analyser(train_concept: Concept, layer_infos: Dict[str, Dict[str, Any]],
             main_model: torch.nn.Module) -> ConceptAnalysis:
    """Return a basic analyser object in which to test methods.
    The analyser is run on a simplistic test/train set.
    To run it on a larger test set, do the following:

    - ensure, the MS COCO dataset or a larger sub-set thereof is unpacked to
      datasets/coco
    - rename the parameter concept to train_concept
    - comment out the line `train_concept = concept` below.
    """
    analyser = ConceptAnalysis(concept=train_concept, model=main_model,
                               layer_infos=layer_infos, cross_val_runs=2,
                               num_val_splits=2)
    assert list(analyser.layer_infos.keys()) == list(layer_infos.keys())
    return analyser


class TestConceptAnalyzer:
    """Test functions for a concept analyser.
    See analyser() for the analyser object used."""

    def test_best_embedding(self, analyser: ConceptAnalysis):
        """Test a run of a ConceptAnalysis."""

        embedding: ConceptEmbedding = analyser.best_embedding()

        # Correct references set within embedding?
        assert embedding.concept.name == analyser.concept.name
        assert embedding.concept.train_data == analyser.concept.train_data
        assert embedding.concept.test_data == analyser.concept.test_data
        assert embedding.model_stump.wrapped_model is analyser.model

    def test_concept_analysis(self, analyser: ConceptAnalysis):
        """Test a run of a concept analysis."""
        results = analyser.analysis()

        # All layers considered?
        assert list(results.keys()) == list(analyser.layer_infos.keys())
        for layer, layer_results in results.items():
            # Value format
            assert isinstance(layer_results, dict), \
                ("Result value at {} not a dict but {}"
                 .format(layer, type(layer_results)))
            for run, run_results in layer_results.items():
                context = ("At layer {}, run {}, result value {}"
                           .format(layer, run, run_results))
                assert len(run_results) == 2, "{} not of len 2 but {}".format(
                    context, len(run_results))
                # Value entry types
                assert isinstance(run_results[0], ConceptEmbedding), \
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
        layer_results: Dict[int, Tuple] = analyser.analysis_for_layer(layer_id)
        emb, std_dev, stats = analyser.embedding_reduction(layer_results)

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
