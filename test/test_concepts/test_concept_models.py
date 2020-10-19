"""Test classes for concepts.models."""
#  Copyright (c) 2020 Continental Automotive GmbH

# pylint: disable=no-self-use
# pylint: disable=redefined-outer-name

from typing import Dict

import numpy as np
import pytest
import torch
import torchvision as tv

from hybrid_learning.concepts.concepts import SegmentationConcept2D
from hybrid_learning.concepts.embeddings import ConceptEmbedding
from hybrid_learning.concepts.models import ConceptDetectionModel2D, \
    ConceptDetection2DTrainTestHandle
from hybrid_learning.datasets.transforms import same_padding
# pylint: disable=unused-import
# noinspection PyUnresolvedReferences
from .common_fixtures import concept_model, concept, train_concept, main_model


# pylint: enable=unused-import

class TestConceptModel:
    """Tests for concept model functionality."""

    def test_same_padding(self):
        """Test whether the _same_padding function yields correct results."""
        # 1D values:
        assert same_padding((1,)) == (0, 0)
        assert same_padding((2,)) == (0, 1)
        assert same_padding((3,)) == (1, 1)
        assert same_padding((4,)) == (1, 2)

        # >1D values:
        assert same_padding((1, 2, 3, 4, 5)) == (0, 0,
                                                 0, 1,
                                                 1, 1,
                                                 1, 2,
                                                 2, 2)

    def test_state_dict(self, concept_model: ConceptDetectionModel2D):
        """Ensure the state_dict only includes the concept layer."""
        state_dict_keys = list(dict(concept_model.state_dict()).keys())
        for key in state_dict_keys:
            assert key.startswith("concept_layer."), \
                "Invalid state_dict key: {}".format(key)

    def test_init(self, concept: SegmentationConcept2D,
                  main_model: torch.nn.Module):
        """Test whether the determination of in_channels, kernel_size,
        and padding works"""
        layer_idx = 'backbone.body.layer3'
        in_channels = 1024
        kernel_size = (5, 5)

        # Specify neither in_channels, nor kernel_size
        c_model = ConceptDetectionModel2D(concept=concept, model=main_model,
                                          layer_id=layer_idx)
        assert c_model.kernel_size == kernel_size
        assert c_model.in_channels == in_channels

        # Specify wrong in_channels (should be corrected with warning),
        # and no kernel_size
        c_model = ConceptDetectionModel2D(concept=concept, model=main_model,
                                          layer_id=layer_idx,
                                          in_channels=in_channels + 1)
        assert c_model.kernel_size == kernel_size
        assert c_model.in_channels == in_channels

        # Specify wrong in_channels and kernel_size (NO correction of
        # in_channels)
        c_model = ConceptDetectionModel2D(concept=concept, model=main_model,
                                          layer_id=layer_idx,
                                          in_channels=in_channels + 1,
                                          kernel_size=(1, 1))
        assert c_model.kernel_size == (1, 1)
        assert c_model.in_channels == in_channels + 1

    def test_from_embedding(self):
        """Test whether from_embedding works with main_model and concept
        missing."""
        kernel_size = (3, 4)
        weight = np.ones((1, 1, *kernel_size))

        emb: ConceptEmbedding = ConceptEmbedding(normal_vec=weight,
                                                 support_factor=1)
        c_model: ConceptDetectionModel2D = \
            ConceptDetectionModel2D.from_embedding(emb)
        assert c_model.main_model is None
        assert c_model.concept is None
        assert c_model.concept_name is None
        assert c_model.concept_layer.weight.data.numpy() is not weight
        assert np.allclose(c_model.concept_layer.weight.data.numpy(), weight)
        assert list(c_model.kernel_size) == list(kernel_size)

        emb: ConceptEmbedding = ConceptEmbedding(normal_vec=weight,
                                                 support_factor=1,
                                                 concept_name='test_name')
        assert emb.concept_name == 'test_name'
        c_model: ConceptDetectionModel2D = \
            ConceptDetectionModel2D.from_embedding(emb)
        # pylint: disable=protected-access
        assert c_model._concept_name == 'test_name'
        # pylint: enable=protected-access
        assert c_model.concept_name == 'test_name'

    def test_from_to_embedding(self, concept_model: ConceptDetectionModel2D):
        """Test to obtain a concept model from embedding and extract
        embedding again."""
        emb: ConceptEmbedding = ConceptEmbedding(
            concept=concept_model.concept,
            model_stump=concept_model.main_model_stump,
            normal_vec=np.ones(concept_model.concept_layer.weight.size()),
            support_factor=1)
        c_model = concept_model.from_embedding(emb)

        # Were the parameters named correctly?
        assert (list(c_model.state_dict().keys())
                == list(concept_model.state_dict().keys()))

        # Was the data from the embedding correctly copied?
        c_model_weight = c_model.concept_layer.weight.detach().cpu().numpy()
        c_model_bias = c_model.concept_layer.bias.detach().cpu().numpy()
        assert np.allclose(c_model_weight, emb.normal_vec), \
            ("Weight differs:\n  old: {},\n       {}\n  new: {}\n       {}"
             .format(emb.normal_vec.shape, emb.normal_vec, c_model_weight.shape,
                     c_model_weight))
        assert np.allclose(c_model_weight, emb.normal_vec), \
            ("Bias differs:\n  old: {},\n       {}\n  new: {}\n       {}"
             .format(np.array(emb.support_factor).shape, emb.support_factor,
                     c_model_bias.shape, c_model_bias))

    def test_settings(self, concept_model: ConceptDetectionModel2D):
        """Test correct settings retrieval."""
        # Settings can be retrieved successfully?
        settings = concept_model.settings

        # New concept model can be initialized with settings?
        ConceptDetectionModel2D(**settings)

    def test_to_from_embedding(self, concept_model: ConceptDetectionModel2D):
        """Test conversion from and to a ConceptEmbedding."""
        # To
        emb: ConceptEmbedding = concept_model.to_embedding()
        # From
        # noinspection PyTypeChecker
        new_model: ConceptDetectionModel2D = \
            concept_model.from_embedding(emb)

        # Were the parameters named and set correctly?
        state_dict: Dict[str, torch.Tensor] = new_model.state_dict()
        assert (list(state_dict.keys())
                == list(concept_model.state_dict().keys()))
        for param in state_dict.keys():
            assert state_dict[param].allclose(
                concept_model.state_dict()[param]), \
                "Parameter {} differs:\nbefore: {}\nafter: {}".format(
                    param, concept_model.state_dict()[param], state_dict[param])

        # Are the settings equal?
        assert (list(new_model.settings.keys())
                == list(concept_model.settings.keys()))
        for setting, val in new_model.settings.items():
            assert val == concept_model.settings[setting], \
                ("Values of old and new model differ for {}:\nold: {}\nnew: {}"
                 .format(setting, val, concept_model.settings[setting]))

    def test_to_embedding_copy(self, concept_model: ConceptDetectionModel2D):
        """Test that all copied vector data during to_embedding() is deeply
        copied"""
        # Check that two calls do not yield the same numpy arrays!
        emb1 = concept_model.to_embedding()
        emb2 = concept_model.to_embedding()
        # Numpy arrays are different objects:
        assert emb1.normal_vec is not emb2.normal_vec
        assert emb1.support_factor is not emb2.support_factor
        # Concept and model are the same:
        assert emb1.concept is emb2.concept
        assert emb1.model_stump is emb2.model_stump

    def test_reset_parameters(self, concept_model: ConceptDetectionModel2D):
        """Test whether the parameter resetting works."""
        old_weight = np.copy(
            concept_model.concept_layer.weight.data.detach().cpu().numpy())
        old_bias = np.copy(
            concept_model.concept_layer.bias.data.detach().cpu().numpy())
        concept_model.reset_parameters()
        new_weight = np.copy(
            concept_model.concept_layer.weight.data.detach().cpu().numpy())
        new_bias = np.copy(
            concept_model.concept_layer.bias.data.detach().cpu().numpy())
        assert not np.allclose(old_weight, new_weight)
        assert not np.allclose(old_bias, new_bias)


class TestTrainTestHandle:
    """Test training and testing of (concept) models."""

    def test_concept_model_train(self, concept_model: ConceptDetectionModel2D):
        """Basic test whether training works or raises errors."""
        c_handle = ConceptDetection2DTrainTestHandle(concept_model,
                                                     max_epochs=1)
        c_handle.train(show_progress_bars=False)

    def test_init(self, concept):
        """Test anything that could go wrong for training."""
        main_model = tv.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        layer_idx = 'backbone.body.layer3'
        in_channels = 1024

        # Normal init
        c_model = ConceptDetectionModel2D(concept=concept, model=main_model,
                                          layer_id=layer_idx)
        c_model_handle = ConceptDetection2DTrainTestHandle(c_model)

        # Are defaults set?
        assert c_model_handle.loss_fn is not None
        assert c_model_handle.metric_fns is not None
        assert c_model_handle.optimizer is not None

        # Is an error thrown when the concept_layer.in_channels does not fit
        # training data?
        # If in_channels & kernel_size are given, in_channels is not checked!
        wrong_model = ConceptDetectionModel2D(concept=concept, model=main_model,
                                              layer_id=layer_idx,
                                              in_channels=in_channels + 1,
                                              kernel_size=(1, 1))
        with pytest.raises(ValueError):
            ConceptDetection2DTrainTestHandle(wrong_model)
