"""Tests for the model extension and slicing classes and functions."""
#  Copyright (c) 2020 Continental Automotive GmbH

import os
# pylint: disable=no-self-use
# pylint: disable=redefined-outer-name
# pylint: disable=no-member
from typing import Dict, List

import PIL.Image
import numpy as np
import pytest
import torch
import torchvision as tv

from hybrid_learning.concepts.models import ConceptDetectionModel2D
from hybrid_learning.concepts.models.model_extension import \
    ActivationMapGrabber, ModelStump, output_size, ModelExtender, \
    dummy_output, output_sizes
from hybrid_learning.datasets.transforms import PadAndResize
# noinspection PyUnresolvedReferences
from .common_fixtures import main_model, concept_model, concept

# Fix random seeds
np.random.seed(0)
torch.manual_seed(0)


class TestActivationMapGrabber:
    """Test wrapping of models for activation map retrieval."""

    BASE_MODEL = tv.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    """The basic model for wrap tests."""
    BASE_LAYERS = tuple('backbone.body.layer{}'.format(i) for i in [1, 2, 3, 4])
    """The layers of interest for wrap tests."""
    EX_IMG_FP = os.path.join("dataset", "coco_test", "images", "train2017",
                             "000000000165.jpg")
    """Path to the example image to use."""
    EX_NN_OUT_SIZES = {'boxes': torch.Size([34, 4]),
                       'labels': torch.Size([34]),
                       'scores': torch.Size([34]),
                       'masks': torch.Size([34, 1, 400, 400])}
    """Sizes of the torch tensors in the NN output for the example image."""

    def test_registering(self):
        """Test all hook registration functions."""
        wnn = ActivationMapGrabber(self.BASE_MODEL, self.BASE_LAYERS)

        # registered_submodules should coincide with keys in hook_handles:
        assert list(wnn.registered_submodules) == list(wnn.hook_handles.keys())

        # correct modules registered in correct order?
        assert list(wnn.registered_submodules) == list(self.BASE_LAYERS)

        # manual unregistering working?
        to_be_newly_registered = self.BASE_LAYERS[0]
        wnn.unregister_submodule(to_be_newly_registered)
        assert list(wnn.registered_submodules) == list(self.BASE_LAYERS)[1:]
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            wnn.register_submodule(None)

        # unregistering of existing module results in error:
        with pytest.raises(KeyError):
            wnn.unregister_submodule("non-registered_sub")

        # manual registering working?
        wnn.register_submodule(to_be_newly_registered)
        assert list(wnn.registered_submodules) == [*list(self.BASE_LAYERS)[1:],
                                                   to_be_newly_registered]

    def test_interm_output_retrieval(self):
        """Is the format and content of the output of a wrapper forward
        as expected?"""

        img_t = PadAndResize(img_size=(400, 400))(
            tv.transforms.ToTensor()(PIL.Image.open(self.EX_IMG_FP))
        ).unsqueeze(0)
        wnn = ActivationMapGrabber(self.BASE_MODEL, self.BASE_LAYERS)
        wnn.eval()
        with torch.no_grad():
            nn_out = wnn.wrapped_model(img_t)
            wnn_out = wnn(img_t)

        # format correct?
        assert isinstance(wnn_out, tuple)
        # submodules outputs:
        assert isinstance(wnn_out[1], dict)
        assert list(wnn_out[1].keys()) == list(self.BASE_LAYERS)
        assert all([isinstance(v, torch.Tensor) for v in wnn_out[1].values()])

        # output correct?
        # first tuple entry: nn output
        assert repr(wnn_out[0]) == repr(nn_out)

        # nn output sizes:
        assert {k: v.size() for k, v in
                list(wnn_out[0][0].items())} == self.EX_NN_OUT_SIZES

        # activation map output size:
        for factor, layer in enumerate(self.BASE_LAYERS):
            wnn_act_map0: PIL.Image.Image = \
                tv.transforms.ToPILImage()(wnn_out[1][layer][0, 0])
            assert wnn_act_map0.size == (400 / (2 ** (factor + 1)),
                                         400 / (2 ** (factor + 1))), \
                "Wrong activation map size for {}".format(layer)


class TestModelStump:
    """Tests of model stump functionality."""
    BASE_MODEL = tv.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    """The basic model for wrap tests."""
    BASE_LAYERS = tuple('backbone.body.layer{}'.format(i) for i in [1, 2, 3, 4])
    """The layers of interest for wrap tests."""

    def test_registering(self):
        """Test all hook registration functions."""
        stump_head = self.BASE_LAYERS[0]
        model_stump = ModelStump(model=self.BASE_MODEL, stump_head=stump_head)

        # registered_submodules should coincide with keys in hook_handles:
        assert (list(model_stump.registered_submodules)
                == list(model_stump.hook_handles.keys()))
        assert stump_head in list(model_stump.registered_submodules)

        # unregistering of existing module results in error:
        with pytest.raises(KeyError):
            model_stump.unregister_submodule("non-registered_sub")

        # manual registering working?
        to_be_newly_registered = self.BASE_LAYERS[1]
        model_stump.register_submodule(to_be_newly_registered)
        assert (list(model_stump.registered_submodules)
                == list(self.BASE_LAYERS)[:2])
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            model_stump.register_submodule(None)

        # manual unregistering working?
        model_stump.unregister_submodule(to_be_newly_registered)
        assert list(model_stump.registered_submodules) == [self.BASE_LAYERS[0]]

        # unregistering the stump_head does unset the stump_head:
        model_stump.unregister_submodule(model_stump.stump_head)
        assert list(model_stump.registered_submodules) == []
        assert model_stump.stump_head is None

        # re-registering a new module sets the stump_head:
        model_stump.register_submodule(stump_head)
        assert list(model_stump.registered_submodules) == [stump_head]
        assert model_stump.stump_head == stump_head

    def test_inter_output_retrieval(self):
        """Is the format and content of the output of a wrapper forward as
        expected?"""
        width = 400/(2**(len(self.BASE_LAYERS)))
        stump_head = self.BASE_LAYERS[-1]
        model_stump = ModelStump(self.BASE_MODEL, stump_head=stump_head).eval()
        img_t = PadAndResize(img_size=(400, 400))(
            tv.transforms.ToTensor()(
                PIL.Image.open(TestActivationMapGrabber.EX_IMG_FP))
        ).unsqueeze(0)
        with torch.no_grad():
            stump_act_map_t: torch.Tensor = model_stump(img_t)

        # format correct? (in this case: a single tensor)
        assert isinstance(stump_act_map_t, torch.Tensor)

        # output format correct?
        # activation map output:
        stump_act_map0: PIL.Image.Image = tv.transforms.ToPILImage()(
            stump_act_map_t[0, 0])
        assert stump_act_map0.size == (width, width)
        # noinspection PyTypeChecker


class TestModelExtender:
    """Test the ModelExtender class."""

    def test_init(self, concept_model: ConceptDetectionModel2D):
        """Test initialization of ModelExtender"""
        layer_idx = concept_model.layer_id
        name = concept_model.concept.name
        extended_model = ModelExtender(
            model=concept_model.main_model,
            extensions={layer_idx: {name: concept_model}})

        assert list(extended_model.extensions.keys()) == [layer_idx]
        assert list(extended_model.extensions[layer_idx].keys()) == [name]
        assert extended_model.extensions[layer_idx][name] == concept_model
        assert list(extended_model.extension_models.keys()) == [name]
        assert extended_model.name_registrations == {layer_idx: [name]}

    def test_forward(self, concept_model: ConceptDetectionModel2D):
        """Test whether a forward of an extended model works fine."""
        layer_idx = concept_model.layer_id
        name = concept_model.concept.name
        extended_model = ModelExtender(
            model=concept_model.main_model,
            extensions={layer_idx: {name: concept_model}})
        extended_model.eval()

        with torch.no_grad():
            img_t, _ = concept_model.concept.test_data[0]
            main_model_out, extensions_out = extended_model(img_t.unsqueeze(0))

        assert (list(extensions_out.keys())
                == list(extended_model.extension_models.keys()))

        # Is the extension output the actual concept model output?
        with torch.no_grad():
            intermediate_out: torch.Tensor = \
                concept_model.main_model_stump.eval()(img_t.unsqueeze(0))
            concept_out: torch.Tensor = concept_model.eval()(intermediate_out)
        assert concept_out.equal(extensions_out[name])

        # Is the main_model_output the actual main model's output?
        with torch.no_grad():
            orig_main_model_out: List[Dict[str, torch.Tensor]] = \
                concept_model.main_model.eval()(img_t.unsqueeze(0))
        assert (list(orig_main_model_out[0].keys())
                == list(main_model_out[0].keys()))
        for k in orig_main_model_out[0]:
            assert orig_main_model_out[0][k].equal(main_model_out[0][k])


def test_dummy_sizes(main_model: torch.nn.Module):
    """Tests for getting dummy outputs of model layers."""
    input_size = (3, 400, 400)
    all_layers: List[str] = [name for name, _ in main_model.named_modules()]
    layer_id = 'backbone.body.layer3'
    assert layer_id in all_layers, ("Setup failure: Make sure, {} is in "
                                    "main_model.named_modules()"
                                    .format(layer_id))

    # Run for all layers:
    outp = dummy_output(main_model, [1, *input_size])
    assert isinstance(outp, dict)
    # All layers caught?
    assert list(outp.keys()) == all_layers

    # Run for a selection of layers:
    outp = dummy_output(main_model, [1, *input_size], layer_ids=[layer_id])
    assert list(outp.keys()) == [layer_id]


def test_output_sizes(main_model: torch.nn.Module):
    """Tests for the output_sizes function."""
    input_size = (3, 400, 400)
    all_layers: List[str] = [name for name, _ in main_model.named_modules()]
    layer_id = 'backbone.body.layer3'
    assert layer_id in all_layers, ("Setup failure: Make sure, {} is in "
                                    "main_model.named_modules()"
                                    .format(layer_id))
    layer_size = (1024, 50, 50)

    # The main model should have a dict as final output, which cannot be
    # parsed to output size:
    with pytest.raises(AttributeError):
        output_sizes(main_model, input_size, ignore_non_tensor_outs=False)
    # It should not fail, if non-tensors are ignored, but yield sizes of all
    # tensor outputs:
    outp = dummy_output(main_model, [1, *input_size])
    outp_s = output_sizes(main_model, input_size)
    assert list(outp_s.keys()) == [ln for ln, o in outp.items()
                                   if isinstance(o, torch.Tensor)]

    # Give without batch dimension
    outp_s = output_sizes(main_model, input_size, layer_ids=[layer_id])
    assert list(outp_s.keys()) == [layer_id]
    assert outp_s[layer_id] == layer_size

    # Give with batch dimension
    outp_s = output_sizes(main_model, [1, *input_size],
                          layer_ids=[layer_id], has_batch_dim=True)
    assert list(outp_s.keys()) == [layer_id]
    assert outp_s[layer_id] == layer_size


def test_output_size(main_model: torch.nn.Module):
    """Tests for the output_size function."""
    input_size = (3, 400, 400)
    all_layers: List[str] = [name for name, _ in main_model.named_modules()]
    layer_id = 'backbone.body.layer3'
    assert layer_id in all_layers, ("Setup failure: Make sure, {} is in "
                                    "main_model.named_modules()"
                                    .format(layer_id))
    model_stump = ModelStump(main_model, layer_id)
    outp: torch.Tensor = model_stump.eval()(torch.zeros(size=[1, *input_size]))
    outp_size_without_batch = outp.size()[1:]

    # The main model should have a dict as final output, which cannot be
    # parsed to output size:
    with pytest.raises(AttributeError):
        output_size(main_model, input_size)

    # It should not fail for an intermediate layer with tensor output:
    # Give without batch dimension
    outp_s = output_size(model_stump, input_size)
    assert outp_s == outp_size_without_batch

    # Give with batch dimension
    outp_s = output_size(model_stump, [1, *input_size], has_batch_dim=True)
    assert outp_s == outp_size_without_batch
