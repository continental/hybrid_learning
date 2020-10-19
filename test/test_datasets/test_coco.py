"""Test classes for coco dataset handles."""
#  Copyright (c) 2020 Continental Automotive GmbH

# Pytest does usually not use the "self" when grouping tests in classes
# pylint: disable=no-self-use

# During testing, also the protected methods should get tested:
# pylint: disable=protected-access

# Pylint seems to think the code libraries are external:
# pylint: disable=wrong-import-order

# Pylint cannot properly cope with numpy and torch members, so call with
# --extension-pkg-whitelist=torch

import os
from time import sleep
from typing import Callable, Tuple, Dict, Any

import pytest
import torch
import torchvision as tv
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.models.detection import maskrcnn_resnet50_fpn

from hybrid_learning.concepts.models.model_extension import ModelStump
from hybrid_learning.datasets import ActivationDatasetWrapper
from hybrid_learning.datasets.custom.coco import KeypointsDataset, \
    ConceptDataset, BodyParts, COCODataset


def default_coco_spec() -> Dict[str, Any]:
    """Default path and ``body_part`` args for COCO datasets."""
    data_root = COCODataset.DATASET_ROOT_TEMPL.format(split="train") \
        .replace("coco", "coco_test").replace(".." + os.sep, "")
    return dict(dataset_root=data_root,
                body_parts=(BodyParts.FACE,))


class TestCOCOKeypointsSubset:
    """Test class for coco keypoints dataset handle: Simple subset with just
    a few images."""

    @staticmethod
    @pytest.fixture
    def coco():
        """Common preparation routine: Obtain coco dataset handle."""
        # Dataset initialization:
        spec = default_coco_spec()
        coco = KeypointsDataset(dataset_root=spec['dataset_root'])
        return coco

    def test_repr(self, coco: KeypointsDataset):
        """Does init and printing work?"""
        str(coco)

    def test_len(self, coco: KeypointsDataset):
        """Sanity checks on content format (lengths)."""

        # There should be something in there:
        assert len(coco) > 0

        # And it should be images & annotations:
        content = coco[0]
        assert len(content) == 2

    def test_coco_content_images(self, tmpdir, coco: KeypointsDataset):
        """Is the content of coco a proper image and the second proper coco
        annotation?"""
        content = coco[0]
        img_t, anns = content

        # Check images:
        assert isinstance(img_t, torch.Tensor)
        img = tv.transforms.ToPILImage()(img_t)

        test_fp = os.path.join(tmpdir, "test.png")
        plt.imshow(img)
        coco.coco.showAnns(anns)
        plt.savefig(test_fp)
        assert os.path.exists(test_fp) and os.path.isfile(test_fp), \
            "Image from dataset could not be successfully saved!"


class TestCOCOConceptData:
    """Test class for coco keypoints dataset handle:
    Simple subset with just a few images."""

    @staticmethod
    def cleanup(coco: ConceptDataset) -> None:
        """Cleanup after testing."""
        for mask_fn in os.listdir(coco.masks_root):
            os.remove(os.path.join(coco.masks_root, mask_fn))
        os.rmdir(coco.masks_root)
        assert not os.path.exists(coco.masks_root), \
            ("Cleanup failed: could not remove mask directory {}"
             ).format(coco.masks_root)

    @staticmethod
    @pytest.fixture
    def coco():
        """Common preparation routine: Obtain coco dataset handle."""
        # Dataset initialization:
        coco = ConceptDataset(**default_coco_spec())
        yield coco
        # Cleanup
        TestCOCOConceptData.cleanup(coco)

    def test_masks_root(self, coco: ConceptDataset):
        """Whether the masks root is set and created correctly."""
        assert coco.masks_root == coco._default_masks_root
        assert (os.path.basename(coco.masks_root)
                == "train2017_left_eye-nose-right_eye_rad0.025")
        assert \
            os.path.dirname(os.path.dirname(coco.masks_root)) == \
            os.path.dirname(os.path.dirname(coco.dataset_root)), \
            "Mask and image root root folders do not share the same parent."
        assert os.path.isdir(coco.masks_root)

    def test_repr(self, coco: ConceptDataset):
        """Does init and printing work?"""
        str(coco)

    def test_len(self, coco: ConceptDataset):
        """Sanity checks on content format (lengths)."""

        # There should be something in there:
        assert len(coco) > 0

        # And it should be images & annotations:
        content = coco[0]
        assert len(content) == 2

    def test_coco_content_images(self, coco: ConceptDataset):
        """Is the content of coco a proper image and the second proper coco
        annotation?"""
        content = coco[0]
        img, mask = content

        # Check images:
        assert isinstance(img, torch.Tensor)
        assert isinstance(mask, torch.Tensor)

    def test_mask_exists(self, coco: ConceptDataset):
        """Sanity checks for mask_exists"""
        # Are all generated masks marked as existent?
        coco.generate_masks(show_progress_bar=False)
        for i in range(len(coco.img_ids)):
            assert coco.mask_exists(i), \
                "Mask marked as non-existent: {}".format(coco.mask_filepath(i))

        # And what about others?
        assert len(coco.img_ids) < 10000
        assert not coco.mask_exists(10000)

    def test_generate_masks(self, coco: ConceptDataset):
        """Does mask generation work properly?"""
        i = 0
        coco.generate_masks(show_progress_bar=False)
        assert coco.mask_exists(i)
        orig_modification_time = os.path.getmtime(coco.mask_filepath(i))

        # Re-generate without overwriting
        coco.generate_masks()
        assert coco.mask_exists(i)
        assert (orig_modification_time
                == os.path.getmtime(coco.mask_filepath(i))), \
            "Mask was modified even though overwriting was disabled"

        # Re-generate and overwrite
        assert coco.mask_exists(i)
        sleep(0.01)
        coco.generate_masks(force_rebuild=True)
        assert (orig_modification_time
                < os.path.getmtime(coco.mask_filepath(i))), \
            "Mask was not modified even though overwriting was enabled"

    def test_transform(self, coco: ConceptDataset):
        """Does the default transformation fulfill essential properties
        like idempotence?"""
        img_size = (400, 400)
        to_img: Callable[[torch.Tensor], Image.Image] = \
            tv.transforms.ToPILImage()
        non_target_size = (img_size[0] + 10, img_size[1] + 10)
        plain_img = Image.new('RGB', non_target_size)
        plain_mask = Image.new('1', non_target_size)
        i_m: Tuple[torch.Tensor, torch.Tensor] = \
            coco.transforms(plain_img, plain_mask)
        test_img_t: torch.Tensor = i_m[0]
        test_mask_t: torch.Tensor = i_m[1]

        # Transformation functions correctly called?
        assert coco.transforms == coco._default_transforms, \
            "Transform not set to default"

        # Correct format after first transformation?
        assert to_img(test_img_t).size == img_size, \
            "Size transformation wrong for image"
        assert to_img(test_mask_t).size == img_size, \
            "Size transformation wrong for mask"
        assert (test_mask_t <= 1).all() and (test_mask_t >= 0).all(), \
            "Transformation yields wrong mask image mode"

        # Transformation is idempotent?
        i_m: Tuple[torch.Tensor, torch.Tensor] = \
            coco.transforms(to_img(test_img_t), to_img(test_mask_t))
        test2_img_t: torch.Tensor = i_m[0]
        test2_mask_t: torch.Tensor = i_m[1]
        assert to_img(test2_img_t).size == img_size, \
            "Size transformation not idempotent"
        assert to_img(test2_mask_t).size == img_size, \
            "Size transformation not idempotent"
        assert (test2_mask_t <= 1).all() and (test2_mask_t >= 0).all(), \
            "2nd transformation yields wrong mask image mode"

    def test_get_mask_filepath(self, coco):
        """Is the mask filepath correctly derived from the image id?"""
        assert coco.mask_filepath(0) == os.path.join(
            coco.masks_root, "{:0>12}.jpg".format(coco.img_ids[0]))


class TestCOCOConceptActivationDataset:
    """Test class for coco keypoints dataset handle:
    Simple subset with just a few images."""

    # Default layer to use for testing
    LAYER_KEY = 'backbone.body.layer4'
    MODEL_HASH = '0x5716707f'

    @staticmethod
    def cleanup(coco: ActivationDatasetWrapper):
        """Cleanup after testing."""
        # masks
        # noinspection PyTypeChecker
        TestCOCOConceptData.cleanup(coco.dataset)

        # # activations
        # for act_fn in os.listdir(coco.activations_root):
        #     os.remove(os.path.join(coco.activations_root, act_fn))
        # os.rmdir(coco.activations_root)
        # assert not os.path.exists(coco.activations_root), \
        #     "Cleanup: could not remove activations directory {}"\
        #     .format(coco.activations_root)

    @staticmethod
    @pytest.fixture
    def mcoco():
        """Common preparation routine:
        Obtain coco dataset handle for mask_r_cnn model"""
        # Dataset initialization:
        model_stump = ModelStump(
            model=maskrcnn_resnet50_fpn(pretrained=True),
            stump_head=TestCOCOConceptActivationDataset.LAYER_KEY)
        dataset: ConceptDataset = ConceptDataset(**default_coco_spec())
        coco = ActivationDatasetWrapper(
            act_map_gen=model_stump,
            dataset=dataset)
        yield coco
        TestCOCOConceptActivationDataset.cleanup(coco)

    @staticmethod
    @pytest.fixture
    def model() -> ModelStump:
        """Common model stump to generate activation maps in eval mode"""
        return ModelStump(
            model=maskrcnn_resnet50_fpn(pretrained=True),
            stump_head=TestCOCOConceptActivationDataset.LAYER_KEY).eval()

    @staticmethod
    @pytest.fixture
    def coco(model: ModelStump):
        """Common preparation routine:
        Obtain coco dataset handle from description."""
        dataset = ConceptDataset(**default_coco_spec())
        coco: ActivationDatasetWrapper = ActivationDatasetWrapper(
            dataset=dataset,
            layer_key=TestCOCOConceptActivationDataset.LAYER_KEY,
            model_description=TestCOCOConceptActivationDataset.MODEL_HASH,
            act_map_gen=model)
        yield coco
        TestCOCOConceptActivationDataset.cleanup(coco)

    def test_init_options(self):
        """Test whether init successfully checks outputs."""
        # Some preliminary setup
        coco_dataset: ConceptDataset = ConceptDataset(**default_coco_spec())
        dataset_root: str = coco_dataset.dataset_root
        wrong_desc = 'wrong_hash'
        wrong_desc_dir = os.path.join(
            os.path.dirname(os.path.dirname(dataset_root)),
            ActivationDatasetWrapper._ACT_MAPS_ROOT_ROOT,
            "{}_{}-{}".format(os.path.basename(dataset_root),
                              wrong_desc,
                              self.LAYER_KEY))
        # Remove the activation map directory
        if os.path.exists(wrong_desc_dir):
            for fname in os.listdir(wrong_desc_dir):
                os.remove(fname)
            os.rmdir(wrong_desc_dir)

        # Either model or model_description must be given
        with pytest.raises(ValueError):
            ActivationDatasetWrapper(layer_key=self.LAYER_KEY,
                                     dataset=coco_dataset)

        # Description with related directory but with img missing from
        # directory content
        with pytest.raises(FileNotFoundError):
            ActivationDatasetWrapper(layer_key=self.LAYER_KEY,
                                     model_description=wrong_desc,
                                     dataset=coco_dataset,
                                     lazy_generation=False)
        os.rmdir(wrong_desc_dir)

        # Description without related directory
        with pytest.raises(FileNotFoundError):
            ActivationDatasetWrapper(layer_key=self.LAYER_KEY,
                                     model_description=wrong_desc,
                                     dataset=coco_dataset)
        os.rmdir(wrong_desc_dir)

        assert not os.path.exists(wrong_desc_dir), \
            ("Cleanup failure: temporary directory {} still exists."
             .format(wrong_desc_dir))

    def test_activations_root(self, coco: ActivationDatasetWrapper):
        """Whether the activations root is set and created correctly."""
        assert coco.act_maps_root == coco._default_activations_root
        assert os.path.basename(coco.act_maps_root) == "train2017_{}-{}".format(
            TestCOCOConceptActivationDataset.MODEL_HASH,
            TestCOCOConceptActivationDataset.LAYER_KEY
        )
        assert \
            os.path.dirname(os.path.dirname(coco.act_maps_root)) == \
            os.path.dirname(os.path.dirname(coco.dataset_root)), \
            "Mask and image root root folders do not share the same parent."
        assert os.path.isdir(coco.act_maps_root)

    def test_generate_act_map(self, model: ModelStump):
        """Test whether generate_act_map actually generates valid act maps"""
        img_t = tv.transforms.ToTensor()(Image.new('RGB', (400, 300)))
        act_map = ActivationDatasetWrapper.generate_act_map(model, img_t)

        assert isinstance(act_map, torch.Tensor)
        assert act_map.size() == torch.Size([2048, 25, 34])

    def test_len(self, coco: ConceptDataset):
        """Sanity checks on content format (lengths)."""

        # There should be something in there:
        assert len(coco) > 0

        # And it should be images & annotations:
        content = coco[0]
        assert len(content) == 2

    def test_coco_content_images(self, coco: ConceptDataset):
        """Is the content of coco a proper image and the second proper
        coco annotation?"""
        content = coco[0]
        act_map, mask = content

        # Check images:
        assert isinstance(act_map, torch.Tensor)
        assert len(act_map.size()) == 3  # channels, height, width; no batch
        assert len(mask.size()) == 3  # channels==1, height, width; no batch
        assert isinstance(mask, torch.Tensor)

    def test_generate_act_maps(self, mcoco: ActivationDatasetWrapper):
        """Test functionality of generate_act_map"""
        i = 0
        assert mcoco.act_map_exists(i)
        act_fp = mcoco.act_map_filepath(i)

        # Default generation -> All activation maps should exist
        mcoco.generate_act_maps()
        assert len(mcoco) > 0
        for i in range(len(mcoco)):
            file_path = mcoco.act_map_filepath(i)
            assert os.path.exists(file_path), \
                "No act map generated for path {}".format(file_path)

        # Overwrite if stated:
        orig_modification_time = os.path.getmtime(act_fp)
        mcoco.generate_act_maps(force_rebuild=True)
        assert os.path.getmtime(act_fp) > orig_modification_time, \
            "Activation map not overwritten (fp: {})".format(act_fp)

        # No overwrite if not stated:
        orig_modification_time = os.path.getmtime(act_fp)
        # noinspection PyArgumentEqualDefault
        mcoco.generate_act_maps(force_rebuild=False)
        assert os.path.getmtime(act_fp) == orig_modification_time, \
            "Activation map overwritten (fp: {})".format(act_fp)
