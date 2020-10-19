"""Tests for the basic dataset manipulation methods."""
#  Copyright (c) 2020 Continental Automotive GmbH

# Pylint seems to think the code libraries are external:
# pylint: disable=wrong-import-order

import pytest
import torch
from PIL import Image

from hybrid_learning.datasets import cross_validation_splits
from hybrid_learning.datasets import data_visualization as datavis


def test_cross_validation_splits():
    """Test whether cross_validation_splits are calculated correctly."""
    # Sample datasets and settings to split;
    # Format: List of tuples of
    # ({args for cross_validation_splits}, [tuples len(train), len(test)])
    samples = [
        ({"train_val_data": list(range(50)), "num_splits": 2},
         [(25, 25), (25, 25)]),
        ({"train_val_data": list(range(50)), "num_splits": 3},
         [(33, 17), (33, 17), (34, 16)]),
        ({"train_val_data": list(range(50)), "num_splits": 3},
         [(33, 17), (33, 17), (34, 16)]),
    ]

    for args, lens in samples:
        splits = cross_validation_splits(**args)
        result_lens = [(len(train), len(test)) for train, test in splits]
        assert list(lens) == result_lens

    for invalid_num_splits in (-1, 0, 1):
        with pytest.raises(ValueError):
            cross_validation_splits(list(range(50)),
                                    num_splits=invalid_num_splits)


def test_apply_mask():
    """Does mask application fulfill basic properties?"""
    size = (400, 300)

    # If mask and image size do not match, error is raised
    img = Image.new(mode='RGB', size=size)
    mask = Image.new(mode='L', size=(size[0], size[1] + 100))
    with pytest.raises(ValueError):
        datavis.apply_mask(img, mask)

    # Original image and mask are not overwritten
    img = Image.new(mode='RGB', size=size, color='red')
    mask = Image.new(mode='L', size=size, color=100)
    masked_img = datavis.apply_mask(img, mask)
    assert id(masked_img) != id(img) and id(masked_img) != id(mask)

    # If black mask is applied, nothing happens.
    img = Image.new(mode='RGB', size=size, color='red')
    mask = Image.new(mode='L', size=size)
    masked_img = datavis.apply_mask(img, mask)
    assert masked_img == img

    # If alpha == 0, nothing happens
    img = Image.new(mode='RGB', size=size, color='red')
    mask = Image.new(mode='L', size=size, color=100)
    masked_img = datavis.apply_mask(img, mask, alpha=0)
    assert masked_img == img

    # If alpha == 1, image sections are overwritten
    img = Image.new(mode='RGB', size=size,
                    color='red')  # color different from green
    mask = Image.new(mode='L', size=size, color='white')  # mask everything
    masked_img = datavis.apply_mask(img, mask, alpha=1)
    assert masked_img == Image.new(mode='RGB', size=size,
                                   color='green')  # all masked 100%


def test_tensor_from_img():
    """Test basic properties of tensor to monochrome image conversion."""
    width: int = 400
    height: int = 300
    # pylint: disable=no-member
    img_t: torch.Tensor = torch.ones((height, width))
    # pylint: enable=no-member
    img: Image.Image = datavis.to_monochrome_img(img_t)

    # Correct size
    assert img.size == (width, height)

    # Correct mode
    assert img.mode == 'L'

    # Correct color
    assert img == Image.new(mode='L', size=(width, height), color='white')

# Model hashing
# -------------
# from torchvision.models.alexnet import alexnet
# from code.datasets.base import model_parameter_hash
# def test_model_parameter_hash(tmpdir):
#     """Test basic properties of model hashing."""
#     model = alexnet(pretrained=True)
#     model_description = '0xa4d71c890adc03bed2ab88af5c8d9afa'
#     m_fname = os.path.join(tmpdir, 'm.pkl')
#
#     # Test some exemplary full length hash
#     assert model_parameter_hash(model) == model_description
#
#     # Test truncation option
#     hash_len = 8
#     assert (model_parameter_hash(model, hash_len=hash_len) ==
#             model_description[:hash_len + 2])
#
#     # Two identical models loaded from different files should yield same hash
#     hashes = []
#     for _ in range(2):
#         if os.path.isfile(m_fname):
#             os.remove(m_fname)
#         torch.save(model, m_fname)
#         mod = torch.load(m_fname)
#         hashes.append(model_parameter_hash(mod))
#     assert len(hashes) == 2
#     assert hashes[0] == hashes[1], \
#         ("Identical models did not yield the same hash: "
#          "Got {}, {}").format(*hashes)
#
#     # Two models loaded from the same file should yield the same result
#     hashes = []
#     if os.path.isfile(m_fname):
#         os.remove(m_fname)
#     torch.save(model, m_fname)
#     for _ in range(2):
#         hashes.append(model_parameter_hash(torch.load(m_fname)))
#     assert len(hashes) == 2
#     assert hashes[0] == hashes[1], \
#         ("Models loaded from the same file did not yield the same hash: "
#          "Got {}, {}").format(*hashes)
#
#     # The hash does not change if saved and loaded again
#     assert hashes[0] == model_description, \
#         ("Hash changed when saved and reloaded: "
#          "From {} to {}").format(model_description, hashes[0])
