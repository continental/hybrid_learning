"""Helper functions for visualization and analysis of (image) datasets."""

#  Copyright (c) 2020 Continental Automotive GmbH

import textwrap
from typing import Sequence, List, Dict, Iterable

import PIL.Image
import PIL.ImageEnhance
import numpy as np
import torch
import torchvision as tv
from matplotlib import pyplot as plt
from torch.utils.data import Subset, RandomSampler

from .activations_handle import ActivationDatasetWrapper
from .base import BaseDataset


def to_img(tens: torch.Tensor) -> PIL.Image.Image:
    """Transform a (cuda) tensor to a :py:class:`PIL.Image.Image`."""
    trafo = tv.transforms.ToPILImage()
    return trafo(tens.cpu())


def mean_proportion_pos_px(masks: Sequence[torch.Tensor]):
    """From the given samples, calculate the mean of the proportion of
    positive pixels per mask.
    Assuming mask to be binary.
    """
    prop_per_mask: List[torch.Tensor] = [mask.sum() / mask.numel()
                                         for mask in masks]
    # pylint: disable=no-member
    # pylint: disable=not-callable
    prop: float = float(torch.sum(torch.tensor(prop_per_mask)) / len(masks))
    # pylint: enable=no-member
    # pylint: enable=not-callable
    return prop


def visualize_mask_transforms(
        titled_datasets: Dict[str, ActivationDatasetWrapper],
        save_as: str = None,
        max_num_samples: int = 3) -> None:
    """Visualize difference between original and the transformed masks of
    datasets.
    Visualization results will be shown using
    :py:func:`matplotlib.pyplot.imshow` functionality and are optionally
    saved to a file.

    :param titled_datasets: mapping of ``title`` to ``dataset``
        to be visualized;
        datasets must have the attribute/property ``dataset`` holding the
        original dataset in same indexing order;
        the data contained in each dataset must be a sequence yielding tuples of
        ``(any, transformed mask)``;
        ``data.dataset`` must be a sequence yielding tuples of
        ``(original image, original mask)``;
    :param save_as: file path to save the image under using
        :py:func:`matplotlib.pyplot.savefig`; not saved if ``None``
    :param max_num_samples: the maximum number of samples to show
        for each ``dataset``
    """
    fig = plt.figure(figsize=(3 * max_num_samples * 2,
                              3 * len(titled_datasets)))
    axes = fig.subplots(len(titled_datasets), max_num_samples * 2,
                        sharey='row', squeeze=False)
    fig.tight_layout(h_pad=2)
    for i, (row_title, data) in enumerate(titled_datasets.items()):
        axes[i, 0].set_ylabel(row_title)
        # Collect sample pics
        pics = []
        for ax_idx in range(min(max_num_samples, len(data))):
            _, mask_t = data[ax_idx]
            # Train data is a Subset instance
            assert not isinstance(data, Subset), \
                "'Subset' requires index adaption"
            img_t, orig_mask_t = data.dataset[ax_idx]
            img, orig_mask = to_img(img_t), to_img(orig_mask_t)
            mask = to_img(mask_t).resize(img.size, resample=PIL.Image.BOX)
            applied_masks = apply_mask(apply_mask(img, mask),
                                       orig_mask, alpha=0.4, color='yellow')
            pics += [("Masked img {}; green=new, yellow=orig".format(ax_idx),
                      applied_masks),
                     ("Mask {}".format(ax_idx),
                      mask)]

        for ax_idx, (title, pic) in enumerate(pics):
            axes[i, ax_idx].set_title('\n'.join(textwrap.wrap(title, 30)))
            axes[i, ax_idx].imshow(pic)

    if save_as is not None:
        plt.savefig(save_as, transparent=True)


def visualize_segmentation_data(dataset: BaseDataset, save_as: str = None,
                                max_num_samples: int = 5,
                                skip_none: bool = True, skip_empty: bool = True,
                                shuffle: bool = False):
    """Visualize a dataset yielding tuples of the form ``(input, target_mask)``.

    Both ``input`` and ``target_mask`` must be images as
    :py:class:`torch.Tensor` of the same width and height.

    :param dataset: the dataset to visualize
    :param save_as: file path to save the image under using
        :py:func:`matplotlib.pyplot.savefig`; not saved if ``None``
    :param max_num_samples: the maximum number of samples to show for each
        ``dataset``
    :param skip_none: whether to ignore when a None value in input
        or target is encountered
    :param skip_empty: whether to ignore samples with all-black target masks
    :param shuffle: whether to randomly select samples from the dataset
    """
    num_samples = min(len(dataset), max_num_samples)
    fig, axes = plt.subplots(2, num_samples,
                             figsize=(4 * num_samples, 4 * 2), dpi=100,
                             sharey='row', squeeze=False)
    fig.tight_layout(h_pad=2)

    # row titles
    axes[0, 0].set_ylabel("input")
    axes[1, 0].set_ylabel("target")

    # plot samples
    sampler: Iterable = RandomSampler(dataset) \
        if shuffle else range(len(dataset))
    num_selected_samples: int = 0
    for i in sampler:
        img_t, mask_t = dataset[i]
        # possibly skip sample
        if skip_none and (img_t is None or mask_t is None):
            continue
        if skip_empty and mask_t.sum() == 0:
            continue

        img: PIL.Image.Image = to_img(img_t)
        inverted_mask: PIL.Image.Image = \
            to_img(1 - mask_t).resize(img.size, resample=PIL.Image.BOX)
        applied_mask: PIL.Image.Image = \
            apply_mask(img, inverted_mask, alpha=1, color='black')

        axes[0][num_selected_samples].imshow(img)
        axes[1][num_selected_samples].imshow(applied_mask)

        num_selected_samples += 1
        if num_selected_samples == num_samples:
            break

    if save_as is not None:
        plt.savefig(save_as, transparent=True)


def visualize_classification_data(dataset: BaseDataset, save_as: str = None,
                                  max_num_samples: int = 5,
                                  skip_none: bool = True,
                                  shuffle: bool = False):
    """Visualize a dataset yielding tuples of the form
    ``(input, target_class_identifier)``.
    The ``input`` must be an image as :py:class:`torch.Tensor`.

    :param dataset: the dataset to visualize
    :param save_as: file path to save the image under using
        :py:func:`matplotlib.pyplot.savefig`; not saved if ``None``
    :param max_num_samples: the maximum number of samples to show
        for each ``dataset``
    :param skip_none: whether to ignore samples where a None value in input
        or target is encountered
    :param shuffle: whether to randomly select samples from the dataset
    """
    num_samples = min(len(dataset), max_num_samples)
    fig, axes = plt.subplots(1, num_samples,
                             figsize=(4 * num_samples, 4), dpi=100,
                             sharey='row')
    fig.tight_layout(h_pad=2)

    # plot samples
    sampler: Iterable = RandomSampler(dataset) \
        if shuffle else range(len(dataset))
    num_selected_samples: int = 0
    for i in sampler:
        img_t, ann = dataset[i]
        # possibly skip sample
        if skip_none and (img_t is None or ann is None):
            continue

        axes[num_selected_samples].imshow(to_img(img_t))
        axes[num_selected_samples].set_title(
            '\n'.join(textwrap.wrap(str(ann), 45)))

        num_selected_samples += 1
        if num_selected_samples == num_samples:
            break

    if save_as is not None:
        plt.savefig(save_as, transparent=True)


def neg_pixel_prop(data, max_num_samples: int = 10) -> float:
    """Collect the mean proportion of negative pixels in the binary
    segmentation mask data.
    The proportion is estimated from the first ``max_num_samples`` samples
    in the dataset. Set the value to ``None`` to take into account all
    samples in the dataset.

    :param data: a :py:class:`typing.Sequence` that yields tuples of
        input (arbitrary) and
        masks (as :py:class:`torch.Tensor`).
    :param max_num_samples: the maximum number of samples to take into
        account for the estimation
    :return: the proportion of negative pixels in all (resp. the first
        ``num_samples``) binary segmentation masks contained in the given data
    """
    num_samples: int = len(data) if max_num_samples is None else \
        min(len(data), max_num_samples)
    return 1 - mean_proportion_pos_px([data[i][1] for i in range(num_samples)])


def apply_mask(img: PIL.Image.Image, mask: PIL.Image.Image,
               color: str = 'green', alpha: float = 0.8) -> PIL.Image.Image:
    """Apply monochrome (possibly non-binary) mask to image of same size
    with alpha value.
    The positive parts of the mask are colored in color and added to the image.

    :param img: image
    :param mask: mask (mode ``'L'`` or ``'1'``)
    :param color: color to use for masked regions as argument to Image.new()
    :param alpha: alpha value in ``[0,1]`` to darken mask:

        :0: black,
        :<1: darken,
        :1: original

    :return: ``RGB`` :py:class:`PIL.Image.Image` with mask applied
    """
    if alpha < 0:
        raise ValueError(("Alpha value {} is invalid for darkening"
                          "(must be in [0,1])".format(alpha)))
    if alpha > 1:
        raise ValueError(("Alpha value {} is invalid for darkening (must be in "
                          "[0,1]): Would brighten black areas").format(alpha))

    if mask.size != img.size:
        raise ValueError("Mask size {} and image size {} do not match."
                         .format(mask.size, img.size))

    # The background for mixing
    base_background: PIL.Image.Image = PIL.Image.new(mode='RGB', size=mask.size,
                                                     color=color)

    # Make sure mask has mode 'L':
    mask_l = mask.convert('L')
    # Apply alpha
    enh: PIL.ImageEnhance.Brightness = PIL.ImageEnhance.Brightness(mask_l)
    mask_l = enh.enhance(alpha)

    applied_mask: PIL.Image.Image = PIL.Image.composite(base_background, img,
                                                        mask_l)
    return applied_mask


def to_monochrome_img(img_t: torch.Tensor) -> PIL.Image.Image:
    """:py:class:`torch.Tensor` to monochrome :py:class:`PIL.Image.Image` in
    ``'L'`` (=8-bit) mode.

    :param img_t: monochrome image as tensor of size ``[height, width]``
    :return: resized and darkened image of ``mode='L'``
    """
    img_t_denormalized = (img_t * 255).int()
    img = tv.transforms.ToPILImage()(img_t_denormalized).convert('L')
    return img


def compare_masks(mask1: PIL.Image.Image, mask2: PIL.Image.Image,
                  color1: str = 'blue', color2: str = 'red') -> PIL.Image.Image:
    """Merge several monochrome masks in different colors into the same image.

    :return: image with bright part of each mask in corresponding color
    """
    # region Quick sizes check
    for idx, mask in enumerate([mask1, mask2]):
        if not mask.size == [mask1, mask2][0].size:
            raise ValueError(("Mask at index {} has size {}, which differs from"
                              " mask at index 0 of size {}"
                              ).format(idx, mask.size, [mask1, mask2][0].size))
    # endregion

    colors = [color1, color2]
    colored_mask1 = apply_mask(
        PIL.Image.new(size=mask1.size, mode='RGB', color='black'),
        mask1, alpha=1, color=colors[0])
    colored_mask2 = apply_mask(
        PIL.Image.new(size=mask2.size, mode='RGB', color='black'),
        mask2, alpha=1, color=colors[1])
    # noinspection PyTypeChecker
    mask_comparison = PIL.Image.fromarray(
        np.array(colored_mask2) + np.array(colored_mask1),
        mode='RGB')
    return mask_comparison
