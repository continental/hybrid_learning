"""Tests for the model extension and slicing classes and functions."""
#  Copyright (c) 2022 Continental Automotive GmbH

import os
# pylint: disable=no-self-use
# pylint: disable=redefined-outer-name
# pylint: disable=no-member
from typing import Dict, List, Tuple

import PIL.Image
import numpy as np
import pytest
import torch
import torchvision as tv
from torch.utils.hooks import RemovableHandle

from hybrid_learning.concepts.models import ConceptDetectionModel2D
from hybrid_learning.concepts.models.model_extension import \
    ActivationMapGrabber, ModelStump, output_size, ModelExtender, \
    dummy_output, output_sizes
from hybrid_learning.datasets.transforms import PadAndResize
# pylint: disable=unused-import
# noinspection PyUnresolvedReferences
from .common_fixtures import main_model, concept_model, concept, \
    sample_layer, input_size

# Fix random seeds
np.random.seed(0)
torch.manual_seed(0)


@pytest.fixture
def layer_sizes() -> Dict[str, Tuple[int, int]]:
    """The layers of interest for wrap tests with their output act map sizes."""
    return {
        'features.2': (27, 27),
        'features.5': (13, 13),
        'features.12': (6, 6)
    }


class TestActivationMapGrabber:
    """Test wrapping of models for activation map retrieval."""
    EX_IMG_FP = os.path.join("dataset", "coco_test", "images", "train2017",
                             "000000000165.jpg")
    """Path to the example image to use."""

    def test_registering(self, main_model: torch.nn.Module, layer_sizes):
        """Test all hook registration functions."""
        wnn = ActivationMapGrabber(main_model, list(layer_sizes.keys()))

        # registered_submodules should coincide with keys in hook_handles:
        assert list(wnn.registered_submodules) == list(wnn.hook_handles.keys())

        # correct modules registered in correct order?
        assert list(wnn.registered_submodules) == list(layer_sizes.keys())

        # manual unregistering working?
        to_be_newly_registered = list(layer_sizes.keys())[0]
        wnn.unregister_submodule(to_be_newly_registered)
        left_registered = list(layer_sizes.keys())
        left_registered.remove(to_be_newly_registered)
        assert sorted(wnn.registered_submodules) == sorted(left_registered)

        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            wnn.register_submodule(None)

        # unregistering of existing module results in error:
        with pytest.raises(KeyError):
            wnn.unregister_submodule("non-registered_sub")

        # manual registering working?
        wnn.register_submodule(to_be_newly_registered)
        assert sorted(wnn.registered_submodules) == \
               sorted(layer_sizes.keys())
        assert wnn.registered_submodules[-1] == to_be_newly_registered

    def test_interm_output_retrieval(self, main_model: torch.nn.Module,
                                     input_size: Tuple[int, int, int],
                                     layer_sizes: Dict):
        """Is the format and content of the output of a wrapper forward
        as expected?"""

        img_t = PadAndResize(img_size=input_size[1:])(
            tv.transforms.ToTensor()(PIL.Image.open(self.EX_IMG_FP))
        ).unsqueeze(0)
        wnn = ActivationMapGrabber(main_model, list(layer_sizes.keys()))
        wnn.eval()
        with torch.no_grad():
            nn_out = wnn.wrapped_model(img_t)
            wnn_out = wnn(img_t)

        # format correct?
        assert isinstance(wnn_out, tuple)
        # submodules outputs:
        assert isinstance(wnn_out[1], dict)
        assert list(wnn_out[1].keys()) == list(layer_sizes.keys())
        assert all([isinstance(v, torch.Tensor) for v in wnn_out[1].values()])

        # output correct?
        # first tuple entry: nn output
        assert repr(wnn_out[0]) == repr(nn_out)

        # activation map output size:
        for layer, size in layer_sizes.items():
            wnn_act_map0: PIL.Image.Image = \
                tv.transforms.ToPILImage()(wnn_out[1][layer][0, 0])
            assert wnn_act_map0.size == size, \
                "Wrong activation map size for {}".format(layer)


class TestModelStump:
    """Tests of model stump functionality."""

    def test_registering(self, main_model: torch.nn.Module, layer_sizes: Dict):
        """Test all hook registration functions."""
        stump_head = list(layer_sizes.keys())[0]
        model_stump = ModelStump(model=main_model, stump_head=stump_head)

        # registered_submodules should coincide with keys in hook_handles:
        assert (list(model_stump.registered_submodules)
                == list(model_stump.hook_handles.keys()))
        assert stump_head in list(model_stump.registered_submodules)

        # unregistering of existing module results in error:
        with pytest.raises(KeyError):
            model_stump.unregister_submodule("non-registered_sub")

        # manual registering working?
        to_be_newly_registered = list(layer_sizes.keys())[1]
        model_stump.register_submodule(to_be_newly_registered)
        assert (sorted(model_stump.registered_submodules)
                == [stump_head, to_be_newly_registered])
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            model_stump.register_submodule(None)

        # manual unregistering working?
        model_stump.unregister_submodule(to_be_newly_registered)
        assert sorted(model_stump.registered_submodules) == [stump_head]

        # unregistering the stump_head does unset the stump_head:
        model_stump.unregister_submodule(model_stump.stump_head)
        assert list(model_stump.registered_submodules) == []
        assert model_stump.stump_head is None

        # re-registering a new module sets the stump_head:
        model_stump.register_submodule(stump_head)
        assert list(model_stump.registered_submodules) == [stump_head]
        assert model_stump.stump_head == stump_head

    def test_inter_output_retrieval(self, input_size: Tuple[int, int, int],
                                    main_model: torch.nn.Module,
                                    layer_sizes: Dict):
        """Is the format and content of the output of a wrapper forward as
        expected?"""
        stump_head = list(layer_sizes.keys())[-1]
        model_stump = ModelStump(main_model, stump_head=stump_head).eval()
        img_t = PadAndResize(img_size=input_size[1:])(
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
        assert stump_act_map0.size == layer_sizes[stump_head]
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
            concept_out: List[torch.Tensor] = \
                concept_model.eval()(intermediate_out)
        assert len(concept_out) == 1
        assert concept_out[0].allclose(extensions_out[name][0])

        # Is the main_model_output the actual main model's output?
        with torch.no_grad():
            orig_main_model_out: torch.Tensor = \
                concept_model.main_model.eval()(img_t.unsqueeze(0))
        assert orig_main_model_out.equal(main_model_out)

    def test_return_orig_out(self, concept_model: ConceptDetectionModel2D):
        layer_idx = concept_model.layer_id
        name = concept_model.concept.name
        extended_model = ModelExtender(
            model=concept_model.main_model,
            extensions={layer_idx: {name: concept_model}})

        with torch.no_grad():
            img_t, _ = concept_model.concept.test_data[0]
            out1 = extended_model(img_t.unsqueeze(0))
            assert isinstance(out1, tuple)
            assert isinstance(out1[1], dict)
            assert len(out1) == 2

            extended_model.return_orig_out = False
            out2 = extended_model(img_t.unsqueeze(0))
            assert isinstance(out2, dict)
            assert sorted(out2.keys()) == sorted(out1[1].keys())


def test_dummy_sizes(main_model: torch.nn.Module, sample_layer: Dict):
    """Tests for getting dummy outputs of model layers."""
    input_size = (3, 224, 224)
    all_layers: List[str] = [name for name, _ in main_model.named_modules()]
    layer_id: str = sample_layer["layer_id"]
    assert layer_id in all_layers, ("Setup failure: Make sure, {} is in "
                                    "main_model.named_modules()"
                                    .format(layer_id))

    # Run for a selection of layers:
    outp = dummy_output(main_model, [1, *input_size], [layer_id])
    assert isinstance(outp, dict)


def test_output_sizes(main_model: torch.nn.Module, sample_layer: Dict):
    """Tests for the output_sizes function."""
    input_size = (3, 224, 224)
    all_layers: List[str] = [name for name, _ in main_model.named_modules()]
    layer_id: str = sample_layer["layer_id"]
    assert layer_id in all_layers, ("Setup failure: Make sure, {} is in "
                                    "main_model.named_modules()"
                                    .format(layer_id))
    layer_size = (sample_layer["out_channels"], *sample_layer["out_size"])

    # It should not fail, if non-tensors are ignored, but yield sizes of all
    # tensor outputs:
    outp_s = output_sizes(main_model, input_size, [''])
    assert list(outp_s['']) == [1000]

    # Give without batch dimension
    outp_s = output_sizes(main_model, input_size, layer_ids=[layer_id])
    assert list(outp_s.keys()) == [layer_id]
    assert outp_s[layer_id] == layer_size

    # Give with batch dimension
    outp_s = output_sizes(main_model, [1, *input_size],
                          layer_ids=[layer_id], has_batch_dim=True)
    assert list(outp_s.keys()) == [layer_id]
    assert outp_s[layer_id] == layer_size


def test_output_size(main_model: torch.nn.Module, sample_layer: Dict):
    """Tests for the output_size function."""
    input_size = (3, 224, 224)
    all_layers: List[str] = [name for name, _ in main_model.named_modules()]
    layer_id = sample_layer["layer_id"]
    assert layer_id in all_layers, ("Setup failure: Make sure, {} is in "
                                    "main_model.named_modules()"
                                    .format(layer_id))
    model_stump = ModelStump(main_model, layer_id)
    outp: torch.Tensor = model_stump.eval()(torch.zeros(size=[1, *input_size]))
    outp_size_without_batch = outp.size()[1:]

    # It should not fail for an intermediate layer with tensor output:
    # Give without batch dimension
    outp_s = output_size(model_stump, input_size)
    assert outp_s == outp_size_without_batch

    # Give with batch dimension
    outp_s = output_size(model_stump, [1, *input_size], has_batch_dim=True)
    assert outp_s == outp_size_without_batch


def test_del(main_model: torch.nn.Module):
    """Test whether deleting the HooksHandle really unregisters the hooks."""
    stump = ModelStump(main_model, stump_head='')
    hook: RemovableHandle = stump.hook_handles['']
    assert hook.id in hook.hooks_dict_ref()
    stump.__del__()
    assert hook.id not in hook.hooks_dict_ref()
