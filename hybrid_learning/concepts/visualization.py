"""Functions supporting the visualization of analysis results, datasets,
and models"""
#  Copyright (c) 2020 Continental Automotive GmbH

# torch and numpy produce some pylint issues:
# pylint: disable=no-member
# pylint: disable=not-callable
from typing import Sequence, Dict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt

from hybrid_learning.datasets import data_visualization as datavis
from .embeddings import ConceptEmbedding
from .models import ConceptDetection2DTrainTestHandle as ConceptModelHandle


def visualize_concept_model(handle: ConceptModelHandle,
                            max_num_samples: int = 10,
                            label_templ: str = None,
                            axes: plt.Axes = None,
                            start_idx: int = 0,
                            save_as: str = None):
    """Visualize predictions of a segmentation concept model on test samples.
    Visualization results will be shown using
    :py:func:`matplotlib.pyplot.imshow` functionality and are optionally
    saved to a file.

    :param handle: the train/test handle for the concept model to evaluate
        (holds model and data)
    :param start_idx: relevant if axes is given; the row index to
        start filling axes
    :param axes: optionally, one can specify an :py:class:`matplotlib.axes.Axes`
        object to work on; generated by default
    :param max_num_samples: how many test samples to visualize
    :param label_templ: template that is formatted and printed along with
        each sample; must contain a placeholder for the image ID as ``'{}'``
    :param save_as: .png file path to store the image file in using
        :py:func:`matplotlib.pyplot.savefig`; not saved if ``None``
    """
    if label_templ is not None and '{}' not in label_templ:
        raise ValueError("label_templ {}".format(label_templ) +
                         "did not contain format string {}")
    if not hasattr(handle.data.test, 'load_image'):
        raise AttributeError("Test data of handle does not provide function"
                             " load_image!")
    load_image = getattr(handle.data.test, 'load_image', None)
    if not callable(load_image):
        raise AttributeError("load_image of handle.test_data is not callable!")
    num_samples: int = min(max_num_samples, len(handle.data.test))

    if axes is None:
        fig = plt.figure(figsize=(3 * 4, 3 * num_samples))
        axes = fig.subplots(num_samples, 4, sharey='row', squeeze=False)
        fig.tight_layout(pad=2)
        start_idx = 0  # ensure start index not shifted if axes auto-generated

    for idx in range(num_samples):
        img_t: torch.Tensor = load_image(idx)
        img: Image.Image = datavis.to_img(img_t)
        act_map, mask_t = handle.data.test[idx]
        mask: Image.Image = datavis.to_img(mask_t).resize(img.size,
                                                          resample=Image.BOX)

        # the predicted mask
        pred_mask_t: torch.Tensor = \
            (handle.model(act_map.unsqueeze(0)).squeeze(0)).float()
        pred_mask: Image.Image = \
            datavis.to_img(pred_mask_t).resize(img.size, resample=Image.BOX)
        threshed_pred_mask = datavis.to_img((pred_mask_t > 0.5).float()) \
            .resize(img.size, resample=Image.BOX)
        inverted_pred_mask = datavis.to_img(1 - pred_mask_t) \
            .resize(img.size, resample=Image.BOX)

        # a comparison with the ground truth
        mask_red: Image.Image = datavis.apply_mask(
            Image.new(size=mask.size, mode='RGB', color='black'), mask, alpha=1,
            color='red')
        pred_mask_blue: Image.Image = datavis.apply_mask(
            Image.new(size=mask.size, mode='RGB', color='black'), pred_mask,
            alpha=1, color='blue')
        # noinspection PyTypeChecker
        mask_comparison = Image.fromarray(
            np.array(pred_mask_blue) + np.array(mask_red), mode='RGB')

        pics: Dict[str, Image.Image] = {
            "Original image with\npredicted mask in green":
                datavis.apply_mask(img, pred_mask, alpha=1),
            "Original image with\npred, threshed mask in green":
                datavis.apply_mask(img, threshed_pred_mask, alpha=1),
            "Original image masked\nby predicted mask":
                datavis.apply_mask(img, inverted_pred_mask, alpha=1,
                                   color='black'),
            "Overlay of\nground truth mask (red) &\npredicted mask (blue);"
            "\npink = true positive":
                mask_comparison
        }

        if label_templ is not None:
            axes[idx + start_idx, 0].set_ylabel(label_templ.format(idx))
        for ax_idx, (title, pic) in enumerate(pics.items()):
            axes[idx + start_idx, ax_idx].set_title(title)
            axes[idx + start_idx, ax_idx].imshow(pic)

    if save_as is not None:
        plt.savefig(save_as, transparent=True)


def visualize_concept_models(titled_handles: Dict[str, ConceptModelHandle],
                             save_as: str = None):
    """Visualize the predictions of several concept models in a common image.
    See :py:func:`visualize_concept_model`.

    :param titled_handles: dict of
        ``{title: concept model train/test handle (holding model and data)}``
    :param save_as: file path to the .png file to save using
        :py:func:`matplotlib.pyplot.savefig`; not saved if ``None``
    """
    fig = plt.figure(figsize=(3 * 4, 3 * len(titled_handles)))
    axes = fig.subplots(len(titled_handles), 4, sharey='row', squeeze=False)
    fig.tight_layout(pad=3)
    for i, (title, handle) in enumerate(titled_handles.items()):
        visualize_concept_model(handle, max_num_samples=1,
                                label_templ=str(title) + " (Img {})",
                                axes=axes, start_idx=i)
    if save_as is not None:
        plt.savefig(save_as, transparent=True)


def pairwise_cosines(embs: Sequence[ConceptEmbedding], keys=None):
    """Provide :py:class:`pandas.DataFrame` with pairwise cosines of
    embedding normal vectors."""
    keys = keys if keys is not None else range(len(embs))
    all_normal_vecs = [e.unique().normal_vec for e in embs]
    pairwise_cos = pd.DataFrame(
        [[np.sum(n1 * n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
          for n1 in all_normal_vecs] for n2 in all_normal_vecs],
        index=keys, columns=keys)
    return pairwise_cos