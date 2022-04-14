"""Test classes for concepts.models."""
#  Copyright (c) 2022 Continental Automotive GmbH

# pylint: disable=no-self-use
# pylint: disable=redefined-outer-name
import os
import random
from typing import Dict, List, Tuple, Any, Callable

import numpy as np
import pytest
import torch

from hybrid_learning.concepts import analysis, kpis
from hybrid_learning.concepts.concepts import SegmentationConcept2D
from hybrid_learning.concepts.models import concept_models as c_models, ConceptEmbedding
from hybrid_learning.concepts.train_eval import callbacks
from hybrid_learning.concepts.train_eval.callbacks import CallbackEvents as CB
from hybrid_learning.datasets import DataTriple
from hybrid_learning.datasets.transforms import same_padding
# pylint: disable=unused-import
# noinspection PyUnresolvedReferences
from .common_fixtures import concept_model, concept, main_model, sample_layer, \
    input_size


# pylint: enable=unused-import

class TestConceptModel:
    """Tests for concept model functionality."""

    CMCLASS = c_models.ConceptDetectionModel2D

    @pytest.fixture
    def concept_model(self, concept: SegmentationConcept2D,
                      main_model: torch.nn.Module,
                      sample_layer: Dict) -> 'CMCLASS':
        """Return a standard concept model for given concept for experiments."""
        concept_model = self.CMCLASS(
            concept=concept, model=main_model,
            layer_id=sample_layer["layer_id"],
            in_channels=sample_layer["out_channels"])
        return concept_model

    def test_pytest_fixture(self, concept_model):
        """Make sure the fixture return has the right type."""
        assert isinstance(concept_model, self.CMCLASS)

    @pytest.mark.parametrize(
        ('inp', 'outp_rear_hang', 'outp_front_hang'), [
            # 1D values:
            ((1,), (0, 0), (0, 0)),
            ((2,), (0, 1), (1, 0)),
            ((3,), (1, 1), (1, 1)),
            ((4,), (1, 2), (2, 1)),
            # >1D values:
            ((1, 2, 3, 4, 5), (0, 0,
                               0, 1,
                               1, 1,
                               1, 2,
                               2, 2), (0, 0,
                                       1, 0,
                                       1, 1,
                                       2, 1,
                                       2, 2))
        ])
    def test_same_padding(self, inp: Tuple[int, ...],
                          outp_front_hang: Tuple[int, ...], outp_rear_hang: Tuple[int, ...]):
        """Test whether the _same_padding function yields correct results."""
        # 1D values:
        assert same_padding(inp) == outp_rear_hang
        assert same_padding(inp, hang_front=True) == outp_front_hang

    def test_state_dict(self, concept_model: c_models.ConceptDetectionModel2D):
        """Ensure the state_dict only includes the concept layer."""
        state_dict_keys = list(dict(concept_model.state_dict()).keys())
        for key in state_dict_keys:
            assert key.startswith("concept_layer"), \
                "Invalid state_dict key: {}".format(key)

    def test_init(self, sample_layer: Dict, concept: SegmentationConcept2D,
                  main_model: torch.nn.Module):
        """Test whether the determination of in_channels, kernel_size,
        and padding works"""
        layer_idx = sample_layer["layer_id"]
        in_channels = sample_layer["out_channels"]
        kernel_size = sample_layer["kernel_size"]

        # Specify neither in_channels, nor kernel_size
        c_model = self.CMCLASS(concept=concept, model=main_model,
                               layer_id=layer_idx)
        assert c_model.kernel_size == kernel_size
        assert c_model.in_channels == in_channels
        assert c_model.apply_sigmoid
        assert isinstance(c_model.activation, torch.nn.Sigmoid)

        # Specify wrong in_channels, and no kernel_size
        # -> auto_in_channels gets overridden by specified ones
        c_model = self.CMCLASS(concept=concept, model=main_model,
                               layer_id=layer_idx,
                               in_channels=in_channels + 1)
        assert c_model.kernel_size == kernel_size
        assert c_model.in_channels == in_channels + 1

        # Specify wrong in_channels and kernel_size (NO correction of
        # in_channels)
        c_model = self.CMCLASS(concept=concept, model=main_model,
                               layer_id=layer_idx,
                               in_channels=in_channels + 1,
                               kernel_size=(1, 1))
        assert c_model.kernel_size == (1, 1)
        assert c_model.in_channels == in_channels + 1

        # Specify apply_sigmoid == False -> No activation
        c_model = self.CMCLASS(in_channels=in_channels,
                               kernel_size=(5, 5),
                               apply_sigmoid=False)
        assert not c_model.apply_sigmoid
        assert c_model.activation is None

    def test_forward(self, concept_model: CMCLASS):
        """Test value checks in forward func."""
        # Normal forward
        in_size = [3, concept_model.in_channels, *concept_model.kernel_size]
        out: torch.Tensor = concept_model(torch.zeros(in_size))
        assert len(out) == concept_model.ensemble_count
        assert list(out[0].size()) == [in_size[0], 1, *in_size[-2:]]

        # Wrong in_channels
        in_size = [3, concept_model.in_channels + 1, *concept_model.kernel_size]
        with pytest.raises(ValueError):
            concept_model(torch.zeros(in_size))

        # No batch dim
        in_size = [3, concept_model.in_channels, *concept_model.kernel_size]
        with pytest.raises(ValueError):
            concept_model(torch.zeros(in_size[1:]))

    def test_from_embedding(self):
        """Test whether from_embedding works with main_model and concept
        missing."""
        kernel_size = (3, 4)
        weight = np.ones((1, 1, *kernel_size))

        emb: ConceptEmbedding = ConceptEmbedding(
            state_dict=dict(weight=weight, bias=np.array(1)),
            normal_vec_name="weight", bias_name="bias",
            kernel_size=(weight.shape[-2:])
        )
        c_model: c_models.ConceptDetectionModel2D = \
            self.CMCLASS.from_embedding(emb)
        assert c_model.main_model is None
        assert c_model.concept is None
        assert c_model.concept_name is None
        assert c_model.concept_layer_0.weight.data.numpy() is not weight
        assert np.allclose(c_model.concept_layer_0.weight.data.numpy(), weight)
        assert list(c_model.kernel_size) == list(kernel_size)

        emb: ConceptEmbedding = ConceptEmbedding(
            state_dict=dict(weight=weight, bias=np.array(1)),
            normal_vec_name="weight", bias_name="bias",
            kernel_size=weight.shape[-2:],
            concept_name='test_name')
        assert emb.concept_name == 'test_name'
        c_model: c_models.ConceptDetectionModel2D = \
            self.CMCLASS.from_embedding(emb)
        # pylint: disable=protected-access
        assert c_model._concept_name == 'test_name'
        # pylint: enable=protected-access
        assert c_model.concept_name == 'test_name'

    def test_from_to_embedding(self, concept_model: CMCLASS):
        """Test to obtain a concept model from embedding and extract
        embedding again."""
        weights: torch.Tensor = concept_model.concept_layer_0.weight \
            if hasattr(concept_model.concept_layer_0, "weight") else \
            concept_model.concept_layer_0.weight_mu
        emb: ConceptEmbedding = ConceptEmbedding(
            state_dict=dict(
                weight=np.ones(weights.size()),
                bias=np.array(1)),
            normal_vec_name="weight", bias_name="bias",
            kernel_size=(weights.size()[-2:]),
            concept=concept_model.concept,
            model_stump=concept_model.main_model_stump, )
        c_model = concept_model.from_embedding(emb)

        # Were the parameters named correctly?
        assert sorted(c_model.state_dict().keys()) \
               == ['concept_layer_0.bias', 'concept_layer_0.weight']

        # Was the data from the embedding correctly copied?
        c_model_weight = c_model.concept_layer_0.weight.detach().cpu().numpy()
        c_model_bias = c_model.concept_layer_0.bias.detach().cpu().numpy()
        assert np.allclose(c_model_weight, emb.normal_vec), \
            ("Weight differs:\n  old: {},\n       {}\n  new: {}\n       {}"
             .format(emb.normal_vec.shape, emb.normal_vec, c_model_weight.shape,
                     c_model_weight))
        assert np.allclose(c_model_weight, emb.normal_vec), \
            ("Bias differs:\n  old: {},\n       {}\n  new: {}\n       {}"
             .format(np.array(emb.support_factor).shape, emb.support_factor,
                     c_model_bias.shape, c_model_bias))

    def test_settings(self, concept_model: CMCLASS):
        """Test correct settings retrieval."""
        # Settings can be retrieved successfully?
        settings = concept_model.settings

        # New concept model can be initialized with settings?
        self.CMCLASS(**settings)

    @pytest.mark.parametrize('seed', (0, 1, 2, 3, 4, 5, 42, 101, 33, 64, 8))
    def test_to_from_embedding(self, concept_model: CMCLASS, seed: int):
        """Test conversion from and to a ConceptEmbedding."""
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        # Preliminary value check:
        old_state_dict = concept_model.state_dict()
        for param in old_state_dict.keys():
            assert not torch.isnan(old_state_dict[param]).any(), \
                ("Parameter {} contains NaN values: {}"
                 .format(param, old_state_dict[param]))

        # To
        emb: List[ConceptEmbedding] = concept_model.to_embedding()
        # From
        # noinspection PyTypeChecker
        new_model: c_models.ConceptDetectionModel2D = \
            concept_model.from_embedding(emb)

        # Were the parameters named and set correctly?
        state_dict: Dict[str, torch.Tensor] = new_model.state_dict()
        assert (list(state_dict.keys())
                == list(concept_model.state_dict().keys()))
        for param in state_dict.keys():
            assert not torch.isnan(state_dict[param]).any(), \
                ("Parameter {} contains NaN values: {}"
                 .format(param, state_dict[param]))
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

    def test_to_embedding_copy(self, concept_model: CMCLASS):
        """Test that all copied vector data during to_embedding() is deeply
        copied"""
        # Check that two calls do not yield the same numpy arrays!
        emb1: ConceptEmbedding = concept_model.to_embedding()[0]
        emb2: ConceptEmbedding = concept_model.to_embedding()[0]
        # Numpy arrays are different objects:
        assert emb1.normal_vec is not emb2.normal_vec
        assert emb1.support_factor is not emb2.support_factor
        # Concept and model are the same:
        assert emb1.concept is emb2.concept
        assert emb1.model_stump is emb2.model_stump

    def test_reset_parameters(self, concept_model: CMCLASS):
        """Test whether the parameter resetting works."""
        old_weight = np.copy(
            concept_model.concept_layer_0.weight.data.detach().cpu().numpy())
        old_bias = np.copy(
            concept_model.concept_layer_0.bias.data.detach().cpu().numpy())
        concept_model.reset_parameters()
        new_weight = np.copy(
            concept_model.concept_layer_0.weight.data.detach().cpu().numpy())
        new_bias = np.copy(
            concept_model.concept_layer_0.bias.data.detach().cpu().numpy())
        assert not np.allclose(old_weight, new_weight)
        assert not np.allclose(old_bias, new_bias)

    @pytest.mark.parametrize('model_setts,loss_fn', [
        ({}, kpis.BalancedBCELoss()),
        (dict(apply_sigmoid=False, use_laplace=True), torch.nn.BCEWithLogitsLoss()),
        (dict(apply_sigmoid=True, use_laplace=True), torch.nn.BCELoss()),
    ])
    def test_concept_model_train(self, concept_model: CMCLASS, model_setts: Dict, loss_fn: Callable):
        """Basic test whether training works or raises errors."""
        c_model = self.CMCLASS(**{**concept_model.settings, **model_setts})
        c_data: DataTriple = \
            analysis.analysis_handle.data_for_concept_model(c_model)
        c_handle = c_models.ConceptDetection2DTrainTestHandle(
            c_model, data=c_data, max_epochs=1, loss_fn=loss_fn)
        c_handle.train()

    @pytest.mark.parametrize('model_setts,loss_fn', [
        ({}, kpis.BalancedBCELoss()),
        (dict(apply_sigmoid=False, use_laplace=True), torch.nn.BCEWithLogitsLoss()),
        (dict(apply_sigmoid=True, use_laplace=True), torch.nn.BCELoss()),
    ])
    def test_concept_model_eval(self, concept_model: CMCLASS, model_setts: Dict, loss_fn: Callable):
        """Basic test whether eval works or raises errors."""
        c_model = self.CMCLASS(**{**concept_model.settings, **model_setts})
        c_data: DataTriple = \
            analysis.analysis_handle.data_for_concept_model(c_model)
        c_handle = c_models.ConceptDetection2DTrainTestHandle(
            c_model, data=c_data, max_epochs=1, loss_fn=loss_fn)
        c_handle.evaluate()


class TestConceptModelLaplace(TestConceptModel):
    """Test a model with use_laplace==True."""

    CMCLASS = c_models.ConceptDetectionModel2D

    @pytest.fixture
    def concept_model(self, concept: SegmentationConcept2D,
                      main_model: torch.nn.Module,
                      sample_layer: Dict) -> 'CMCLASS':
        """Return a standard concept model for given concept for experiments."""
        concept_model = self.CMCLASS(
            concept=concept, model=main_model,
            layer_id=sample_layer["layer_id"],
            in_channels=sample_layer["out_channels"],
            use_laplace=True,
        )
        return concept_model

    def test_buffers(self, concept_model):
        """Check whether correct buffers are initiated if use_laplace is set."""
        for i in range(concept_model.ensemble_count):
            layer: torch.nn.Module = \
                getattr(concept_model, f'concept_layer_{i}')
            assert 'hessian' in dict(layer.named_buffers())
            assert 'var0' in dict(layer.named_buffers())

    @pytest.mark.parametrize('model_setts,loss_fn', [
        ({}, kpis.BalancedBCELoss()),
        (dict(apply_sigmoid=False, use_laplace=True), torch.nn.BCEWithLogitsLoss()),
        (dict(apply_sigmoid=True, use_laplace=True), torch.nn.BCELoss()),
    ])
    def test_concept_model_2nd_train(self, concept_model: CMCLASS, model_setts: Dict, loss_fn: Callable):
        """Basic test whether training works or raises errors."""
        c_model = self.CMCLASS(**{**concept_model.settings, **model_setts})
        c_data: DataTriple = \
            analysis.analysis_handle.data_for_concept_model(c_model)
        c_handle = c_models.ConceptDetection2DTrainTestHandle(
            c_model, data=c_data, max_epochs=1, loss_fn=loss_fn)
        c_handle.second_stage_train()


class TestConceptModelEnsemble(TestConceptModel):
    """Test concept model with ensemble_count > 1."""

    CMCLASS = c_models.ConceptDetectionModel2D

    @pytest.fixture
    def concept_model(self, concept: SegmentationConcept2D,
                      main_model: torch.nn.Module,
                      sample_layer: Dict) -> 'CMCLASS':
        """Return a standard concept model for given concept for experiments."""
        concept_model = self.CMCLASS(
            concept=concept, model=main_model,
            layer_id=sample_layer["layer_id"],
            in_channels=sample_layer["out_channels"],
            ensemble_count=3,
        )
        return concept_model

    def test_ensemble_count(self, concept_model):
        """Is ensemble count correct?"""
        assert concept_model.ensemble_count > 1, \
            "Something is wrong with the test setup"
        for i in range(concept_model.ensemble_count):
            assert f'concept_layer_{i}' in dict(concept_model.named_children())


class TestConceptClassificationModel(TestConceptModel):
    """Test concept classification model."""

    CMCLASS = c_models.ConceptClassificationModel2D

    def test_concept_model_train(self, *args, **kwargs):
        """Test classification model training."""
        print("Skipped currently.")  # TODO

    def test_concept_model_eval(self, *args, **kwargs):
        """Test classification model evaluation."""
        print("Skipped currently.")  # TODO

    def test_init(self, sample_layer: Dict, concept: SegmentationConcept2D,
                  main_model: torch.nn.Module):
        """Test whether the determination of in_channels, kernel_size,
        and padding works."""
        layer_idx = sample_layer["layer_id"]
        in_channels = sample_layer["out_channels"]
        kernel_size = sample_layer["out_size"]

        # Specify neither in_channels, nor kernel_size
        c_model = self.CMCLASS(concept=concept,
                               model=main_model,
                               layer_id=layer_idx)
        assert c_model.kernel_size == kernel_size
        assert c_model.in_channels == in_channels
        assert c_model.apply_sigmoid
        assert not c_model.apply_padding

        # Specify wrong in_channels, and no kernel_size
        # -> auto_in_channels gets overridden by specified ones
        c_model = c_models.ConceptClassificationModel2D(concept=concept,
                                                        model=main_model,
                                                        layer_id=layer_idx,
                                                        in_channels=in_channels + 1)
        assert c_model.kernel_size == kernel_size
        assert c_model.in_channels == in_channels + 1

        # Specify apply_sigmoid == False -> No activation
        c_model = self.CMCLASS(in_channels=in_channels,
                               act_map_size=kernel_size,
                               apply_sigmoid=False)
        assert c_model.kernel_size == kernel_size
        assert c_model.in_channels == in_channels
        assert not c_model.apply_sigmoid
        assert c_model.activation is None
        assert not c_model.apply_padding

    def test_forward(self, sample_layer: Dict, concept: SegmentationConcept2D,
                     main_model: torch.nn.Module):
        """Test a forward run of the classification model."""
        layer_idx = sample_layer["layer_id"]
        in_channels = sample_layer["out_channels"]
        kernel_size = sample_layer["out_size"]

        # Specify neither in_channels, nor kernel_size
        c_model = self.CMCLASS(concept=concept,
                               model=main_model,
                               layer_id=layer_idx)
        # Normal input
        inp_size = [3, in_channels, *kernel_size]
        out: List[torch.Tensor] = c_model(torch.zeros(inp_size))
        assert len(out) == 1
        assert list(out[0].size()) == [inp_size[0]]

        # Wrong in_channels
        in_size = [3, c_model.in_channels + 1, *c_model.kernel_size]
        with pytest.raises(ValueError):
            c_model(torch.zeros(in_size))

        # Wrong act_map_size (too small)
        in_size = [3, c_model.in_channels, 1, 1]
        with pytest.raises(ValueError):
            c_model(torch.zeros(in_size))

        # Wrong act_map_size (too large)
        in_size = [3, c_model.in_channels,
                   c_model.kernel_size[0] + 1, c_model.kernel_size[1]]
        with pytest.raises(ValueError):
            c_model(torch.zeros(in_size))
        in_size = [3, c_model.in_channels,
                   c_model.kernel_size[0], c_model.kernel_size[1] + 1]
        with pytest.raises(ValueError):
            c_model(torch.zeros(in_size))

        # No batch dim
        in_size = [3, in_channels, *kernel_size]
        with pytest.raises(ValueError):
            c_model(torch.zeros(in_size[1:]))


class TestTrainTestHandle:
    """Test training and testing of (concept) models."""

    def test_init(self, concept, sample_layer, main_model: torch.nn.Module):
        """Test anything that could go wrong for training."""
        layer_idx = sample_layer["layer_id"]
        in_channels = sample_layer["out_channels"]

        # Normal init
        c_model = c_models.ConceptDetectionModel2D(concept=concept, model=main_model,
                                                   layer_id=layer_idx)
        c_data: DataTriple = \
            analysis.analysis_handle.data_for_concept_model(c_model)
        c_model_handle = c_models.ConceptDetection2DTrainTestHandle(c_model, c_data)

        # Are defaults set?
        assert c_model_handle.metric_fns is not None
        assert c_model_handle.optimizer is not None

        # Is an error thrown when the concept_layer.in_channels does not fit
        # training data?
        # If in_channels & kernel_size are given, in_channels is not checked!
        wrong_model = c_models.ConceptDetectionModel2D(concept=concept, model=main_model,
                                                       layer_id=layer_idx,
                                                       in_channels=in_channels + 1,
                                                       kernel_size=(1, 1))
        with pytest.raises(ValueError):
            _ = analysis.analysis_handle.data_for_concept_model(wrong_model)

    def test_callbacks(self, concept_model: c_models.ConceptDetectionModel2D):
        """Test whether all callbacks are called."""

        def __get_cb_ev_handle(arg_list: List[Dict[str, Any]]) -> Callable:
            return lambda **x: arg_list.append(x)

        call_args: Dict[CB, List[Dict[str, Any]]] = {ev: [] for ev in CB}
        dummy_dict_cb: Dict[CB, Callable] = {ev: __get_cb_ev_handle(arg_l)
                                             for ev, arg_l in call_args.items()}

        c_data: DataTriple = \
            analysis.analysis_handle.data_for_concept_model(concept_model)
        c_handle = c_models.ConceptDetection2DTrainTestHandle(
            concept_model, data=c_data, max_epochs=2, batch_size=1,
            early_stopping_handle=False,
            callbacks=[dummy_dict_cb], callback_context={'custom': 'custom'},
            loss_fn=kpis.BalancedBCELoss())

        # Other callbacks auto-added
        assert any(isinstance(cb, callbacks.ProgressBarUpdater)
                   for cb in c_handle.callbacks)
        assert any(isinstance(cb, callbacks.LoggingCallback)
                   for cb in c_handle.callbacks)

        c_handle.train()

        # Correct # times called?
        assert len(call_args[CB.BEFORE_EPOCH_TRAIN]) == 2
        assert len(call_args[CB.AFTER_BATCH_TRAIN]) == 2 * len(c_data.train)
        assert len(call_args[CB.AFTER_EPOCH_TRAIN]) == 2
        assert len(call_args[CB.AFTER_EPOCH_EVAL]) == 2
        assert len(call_args[CB.AFTER_EPOCH]) == 2
        assert len(call_args[CB.BETWEEN_EPOCHS]) == 1
        assert len(call_args[CB.AFTER_TRAIN]) == 1

    def test_tensorboard_callback(self, tmp_path: str,
                                  concept_model: c_models.ConceptDetectionModel2D):
        """Test the basic functionality of the tensorboard callback."""
        tb_callback = callbacks.TensorboardLogger(log_dir=tmp_path,
                                                  log_sample_targets=True)
        c_data: DataTriple = \
            analysis.analysis_handle.data_for_concept_model(concept_model)

        c_handle = c_models.ConceptDetection2DTrainTestHandle(
            concept_model, data=c_data, max_epochs=1, loss_fn=kpis.BalancedBCELoss(),
            callbacks=[tb_callback])

        c_handle.train(callback_context={'log_prefix': 'train'})
        assert os.path.exists(os.path.join(tmp_path, 'train'))
        assert len(os.listdir(os.path.join(tmp_path, 'train'))) == 1

        c_handle.evaluate(callback_context={'log_prefix': 'val'})
        assert os.path.exists(os.path.join(tmp_path, 'val'))
        assert len(os.listdir(os.path.join(tmp_path, 'val'))) == 1
