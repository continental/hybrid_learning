"""Test classes for coco dataset handles."""
#  Copyright (c) 2022 Continental Automotive GmbH

# Pytest does usually not use the "self" when grouping tests in classes
# pylint: disable=no-self-use

# During testing, also the protected methods should get tested:
# pylint: disable=protected-access

# Pylint seems to think the code libraries are external:
# pylint: disable=wrong-import-order

# Pylint cannot properly cope with numpy and torch members, so call with
# --extension-pkg-whitelist=torch

# pylint: disable=consider-using-enumerate

import os
import time
from time import sleep
from typing import Callable, Tuple, Dict, Any

import numpy as np
import pytest
import torch
import torchvision as tv
from PIL import Image
from matplotlib import pyplot as plt

from hybrid_learning.concepts.models.model_extension import ModelStump
from hybrid_learning.datasets import ActivationDatasetWrapper, caching, \
    transforms, trafos
from hybrid_learning.datasets.custom.coco import KeypointsDataset, \
    ConceptDataset, BodyParts, COCOSegToSegMask, COCOBoxToSegMask
# noinspection PyProtectedMember
from hybrid_learning.datasets.custom.coco.keypoints_processing import \
    person_has_rel_size, _padding_and_scale_for
from hybrid_learning.datasets.base import add_gaussian_peak


def default_coco_spec() -> Dict[str, Any]:
    """Default path and ``body_part`` args for COCO datasets."""
    data_root = ConceptDataset.DATASET_ROOT_TEMPL.format(split="train") \
        .replace("coco", "coco_test").replace(".." + os.sep, "")
    return dict(dataset_root=os.path.abspath(data_root),
                body_parts=(BodyParts.FACE,),
                img_size=(400, 400))


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

    def test_subset(self, coco: KeypointsDataset):
        """Test basic subset functionality."""
        # No effect:
        old_ids = tuple(coco.img_ann_ids)
        len_coco = len(coco)
        assert list(old_ids) == coco.subset(license_ids=None).img_ann_ids

        # Shuffling does not change length:
        assert len(coco.subset(shuffle=True, license_ids=None)) == len_coco
        # Trivial size condition does not change length:
        assert len(coco.subset(condition=person_has_rel_size)) == len_coco

        # Unsatisfiable size condition yields empty subset:
        assert len(coco.subset(condition=(
            lambda i_m, a_m: person_has_rel_size(
                i_m, a_m, min_rel_height=20)))) == 0

        # Some other settings:
        coco.subset(body_parts=['left_eye'])

    def test_transforms_caching(self, tmp_path: str):
        """Test whether caching works with different cache configurations."""
        spec = default_coco_spec()
        print()

        # Only RAM caching
        cache = caching.DictCache()
        coco = KeypointsDataset(dataset_root=spec['dataset_root'],
                                transforms_cache=cache)
        for epoch in range(3):
            start_time = time.time()
            for i in range(len(coco)):
                _ = coco[i]
            end_time = time.time()
            print("RAM cache -------- epoch {}: {}s".format(
                epoch, end_time - start_time))
        cache.clear()

        # Only file cache
        img_pt_cache: caching.PTCache = caching.PTCache(
            cache_root=os.path.join(tmp_path, "img"))
        ann_pt_cache: caching.PTCache = caching.PTCache(
            cache_root=os.path.join(tmp_path, "ann"))
        cache = caching.CacheTuple(img_pt_cache, ann_pt_cache)
        coco = KeypointsDataset(dataset_root=spec['dataset_root'],
                                transforms_cache=cache)
        for epoch in range(3):
            start_time = time.time()
            for i in range(len(coco)):
                _ = coco[i]
            end_time = time.time()
            print("File cache ------- epoch {}: {}s".format(
                epoch, end_time - start_time))
        cache.clear()

        # Rapid caching
        img_pt_cache: caching.PTCache = caching.PTCache(
            cache_root=os.path.join(tmp_path, "img"))
        ann_pt_cache: caching.PTCache = caching.PTCache(
            cache_root=os.path.join(tmp_path, "ann"))
        cache = caching.CacheCascade(
            caching.DictCache(),
            caching.CacheTuple(img_pt_cache, ann_pt_cache),
        )
        coco = KeypointsDataset(dataset_root=spec['dataset_root'],
                                transforms_cache=cache)
        for epoch in range(3):
            start_time = time.time()
            for i in range(len(coco)):
                _ = coco[i]
            end_time = time.time()
            print("RAM & file cache - epoch {}: {}s".format(
                epoch, end_time - start_time))
        assert sorted(cache.descriptors()) == \
               sorted(coco.descriptor(i) for i in range(len(coco)))
        for i in range(len(coco)):
            desc = coco.descriptor(i)
            img, ann = coco[i]
            cached_img, cached_ann = cache.load(desc)
            assert cached_img.equal(img), "Unequal img for {}".format(desc)
            assert cached_ann == ann, "Unequal ann for {}".format(desc)
        cache.clear()

        # No cache
        coco = KeypointsDataset(dataset_root=spec['dataset_root'])
        for epoch in range(3):
            start_time = time.time()
            for i in range(len(coco)):
                _ = coco[i]
            end_time = time.time()
            print("No cache --------- epoch {}: {}s".format(
                epoch, end_time - start_time))

    def test_to_seg_mask(self, coco: KeypointsDataset):
        """Test the COCOSegToSegMask trafo."""
        coco.transforms = (trafos.OnInput(trafos.ToTensor())
                           + trafos.OnTarget(COCOSegToSegMask(coco_handle=coco.coco))
                           + trafos.OnBothSides(trafos.PadAndResize((300, 300))))
        for i in range(len(coco)):
            img, segmask = coco[i]
            assert isinstance(img, torch.Tensor)
            assert isinstance(segmask, torch.Tensor)
            assert list(segmask.size()) == [1, 300, 300]
            assert (segmask > 0).sum() > 0
            assert segmask.min() >= 0 and segmask.max() <= 1

    def test_box_to_seg_mask(self, coco: KeypointsDataset):
        """Test the COCOSegToSegMask trafo."""
        coco.transforms = (trafos.OnInput(trafos.ToTensor())
                           + trafos.OnTarget(COCOBoxToSegMask(coco_handle=coco.coco))
                           + trafos.OnBothSides(trafos.PadAndResize((300, 300))))
        for i in range(len(coco)):
            img, segmask = coco[i]
            assert isinstance(img, torch.Tensor)
            assert isinstance(segmask, torch.Tensor)
            assert list(segmask.size()) == [1, 300, 300]
            assert (segmask > 0).sum() > 0
            assert segmask.min() >= 0 and segmask.max() <= 1


class TestCOCOConceptData:
    """Test class for coco keypoints dataset handle:
    Simple subset with just a few images."""

    @staticmethod
    def cleanup(coco: ConceptDataset) -> None:
        """Cleanup after testing."""
        if os.path.isdir(coco.masks_root):
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

    def test_mask_size(self):
        spec: Dict[str, Any] = default_coco_spec()
        dataset = ConceptDataset(**{**spec, **dict(mask_size=(50, 100),
                                                   img_size=(300, 400))})
        img_t, mask_t = dataset[0]
        assert list(img_t.size()[-2:]) == [300, 400]
        assert list(mask_t.size()[-2:]) == [50, 100]

    def test_masks_root(self, coco: ConceptDataset):
        """Whether the masks root is set and created correctly."""

        # With instance
        assert coco.masks_root == coco.default_masks_root(
            body_parts=coco.body_parts,
            pt_radius=coco.pt_radius,
            dataset_root=coco.dataset_root)
        assert (os.path.basename(coco.masks_root)
                == "train2017_left_eye-nose-right_eye_rad0.025")
        assert os.path.dirname(os.path.dirname(coco.masks_root)) == \
               os.path.dirname(os.path.dirname(coco.dataset_root)), (
            "Mask and image root root folders do not share the same parent.")
        assert os.path.isdir(coco.masks_root)

        # Without instance
        masks_root_root = os.path.join("some", "masks")
        assert (coco.default_masks_root(
            body_parts=[['kpt1', 'kpt2'], ['kpt3', 'kpt4']],
            masks_root_root=masks_root_root)
                == os.path.join("some", "masks", "kpt1-kpt2-kpt3-kpt4"))
        common_setts = dict(
            dataset_root=(os.path.join("some", "path", "basename")),
            body_parts=[['kpt1', 'kpt2'], ['kpt3', 'kpt4']])
        common_prefix = os.path.join("some", "masks",
                                     "basename_kpt1-kpt2-kpt3-kpt4")
        assert coco.default_masks_root(**common_setts) == common_prefix

        # wt img_size
        assert coco.default_masks_root(**common_setts, img_size=(400, 300)) \
               == common_prefix + "_400x300"
        # wt img_size and mask_size
        assert coco.default_masks_root(**common_setts, img_size=(400, 300),
                                       mask_size=(200, 150)) \
               == common_prefix + "_200x150"

        # wt pt_radius
        common_setts.update(pt_radius=0.15)
        assert coco.default_masks_root(**common_setts) \
               == common_prefix + "_rad0.150"
        assert coco.default_masks_root(**common_setts,
                                       person_rel_size_range=(None, None)) \
               == common_prefix + "_rad0.150"
        # wt pt_radius and rel_size
        assert coco.default_masks_root(**common_setts,
                                       person_rel_size_range=(0.5, 1)) \
               == common_prefix + "_rad0.150_relsize0.50to1.00"
        assert coco.default_masks_root(**common_setts,
                                       person_rel_size_range=(0.5, None)) \
               == common_prefix + "_rad0.150_relsize0.50to+inf"
        assert coco.default_masks_root(**common_setts,
                                       person_rel_size_range=(None, 1)) \
               == common_prefix + "_rad0.150_relsize0.00to1.00"

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
        for i in range(len(coco)):
            assert coco.mask_exists(i), \
                "Mask marked as non-existent: {}".format(coco.mask_filepath(i))

        # And what about others?
        assert len(coco) < 10000
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
            coco.masks_root, "{:0>12}.jpg".format(coco.image_meta(0)['id']))


class TestCOCOConceptActivationDataset:
    """Test class for coco keypoints dataset handle:
    Simple subset with just a few images."""

    # Default layer to use for testing
    LAYER_KEY = 'features.5'

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
    def model() -> ModelStump:
        """Common model stump to generate activation maps in eval mode"""
        return ModelStump(
            model=tv.models.alexnet(pretrained=True),
            stump_head=TestCOCOConceptActivationDataset.LAYER_KEY).eval()

    @staticmethod
    @pytest.fixture
    def mcoco(model: ModelStump):
        """Common preparation routine:
        Obtain coco dataset handle for mask_r_cnn model"""
        # Dataset initialization:
        dataset: ConceptDataset = ConceptDataset(**default_coco_spec())
        act_cache_root = os.path.join(
            os.path.dirname(os.path.dirname(dataset.dataset_root)),
            "activations", "{base}_{net}_{layer}".format(
                base=os.path.basename(dataset.dataset_root),
                net=model.wrapped_model.__class__.__name__,
                layer=model.stump_head
            ))
        coco = ActivationDatasetWrapper(dataset=dataset,
                                        act_map_gen=model,
                                        activations_root=act_cache_root)
        yield coco
        TestCOCOConceptActivationDataset.cleanup(coco)

    @staticmethod
    @pytest.fixture
    def coco(model: ModelStump):
        """Common preparation routine:
        Obtain coco dataset handle from description."""
        dataset = ConceptDataset(**default_coco_spec())
        coco: ActivationDatasetWrapper = ActivationDatasetWrapper(
            dataset=dataset, act_map_gen=model)
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
            "activations",
            "{}_{}-{}".format(os.path.basename(dataset_root),
                              wrong_desc,
                              self.LAYER_KEY))
        # Remove the activation map directory
        if os.path.exists(wrong_desc_dir):
            for fname in os.listdir(wrong_desc_dir):
                os.remove(fname)
            os.rmdir(wrong_desc_dir)

        # fill_cache() will raise if generate_act_map is None
        with pytest.raises(ValueError):
            ActivationDatasetWrapper(dataset=coco_dataset,
                                     activations_root=wrong_desc_dir,
                                     ).fill_cache()
        os.rmdir(wrong_desc_dir)

        assert not os.path.exists(wrong_desc_dir), \
            ("Cleanup failure: temporary directory {} still exists."
             .format(wrong_desc_dir))

    def test_generate_act_map(self, model: ModelStump):
        """Test whether the transformation behind generate_act_map generates
        valid act maps."""
        img_t = tv.transforms.ToTensor()(Image.new('RGB', (400, 300)))
        act_map = transforms.ToActMap(model)(img_t)

        assert isinstance(act_map, torch.Tensor)
        assert act_map.size() == torch.Size([192, 17, 24])

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
        _ = mcoco[i]
        assert mcoco.act_map_exists(i)
        act_fp = mcoco.act_map_filepath(i)

        # Default generation -> All activation maps should exist
        mcoco.fill_cache()
        assert len(mcoco) > 0
        for i in range(len(mcoco)):
            file_path = mcoco.act_map_filepath(i)
            assert os.path.exists(file_path), \
                "No act map generated for path {}".format(file_path)

        # Overwrite if stated:
        orig_modification_time = os.path.getmtime(act_fp)
        mcoco.fill_cache(force_rebuild=True)
        assert os.path.getmtime(act_fp) > orig_modification_time, \
            "Activation map not overwritten (fp: {})".format(act_fp)

        # No overwrite if not stated:
        orig_modification_time = os.path.getmtime(act_fp)
        # noinspection PyArgumentEqualDefault
        mcoco.fill_cache(force_rebuild=False)
        assert os.path.getmtime(act_fp) == orig_modification_time, \
            "Activation map overwritten (fp: {})".format(act_fp)


@pytest.mark.parametrize(
    "from_size,to_size,exp_scale,exp_padding", [
        # No-op
        ((1, 1), (1, 1), 1, (0, 0, 0, 0)),
        ((400, 400), (400, 400), 1, (0, 0, 0, 0)),
        # Only scaling
        ((50, 50), (100, 100), 2, (0, 0, 0, 0)),
        # Only padding
        ((100, 100), (200, 100), 1, (0, 0, 50, 50)),
        ((100, 100), (100, 200), 1, (50, 50, 0, 0)),
        # Pad and scale
        ((100, 200), (400, 400), 2, (0, 0, 50, 50)),
        ((200, 100), (400, 400), 2, (50, 50, 0, 0)),
    ])
def test_scale_and_padding_for(from_size: Tuple[int, int],
                               to_size: Tuple[int, int], exp_scale: float,
                               exp_padding: Tuple[int, int, int, int]):
    """Test for helper function _scale_and_padding_for()."""
    # Test samples in the format
    # (from_size, to_size,
    #  scale, (pad_top, pad_left, pad_bottom, pad_right))
    kwargs = {"from_size": from_size, "to_size": to_size}
    padding, scale = _padding_and_scale_for(**kwargs)
    assert scale == exp_scale, "Failed for {}".format(kwargs)
    assert padding == exp_padding, "Failed for {}".format(kwargs)


def test_add_heatmap_peak():
    """Test for helper function add_gaussian_peak."""
    assert add_gaussian_peak(mask_np=np.zeros((1, 1)), centroid=(0.5, 0.5),
                             binary_radius=10) == np.array(1).reshape((1, 1))
