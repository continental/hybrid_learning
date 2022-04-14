"""Config adaptations and named config to do object detection in the
form of heatmap prediction.
On top of the ``config_coco.py`` settings, modifies the ground truth to be
heatmaps with (non-normalized gaussian) peaks at each body part centroid,
and adapts the loss to optimize for little divergence between the ground
truth and predicted heatmap (resp. distribution).
A non-max-suppression on the predicted heatmaps should yield object centers.
"""

#  Copyright (c) 2022 Continental Automotive GmbH
from typing import Dict, Any

import torch

from .config_ca import trafos, models, kpis
from .config_coco import ex, coco, COCODataGetter


# noinspection PyUnusedLocal
@ex.named_config
def heatmaps_config():
    # pylint: disable=unused-variable
    """The default configuration for coco specifics."""
    # The callable to use for data retrieval
    get_data: COCOHeatmapsDataGetter = COCOHeatmapsDataGetter()

    concept_model_setts: Dict[str, Any] = dict(
        apply_sigmoid=False)  # handled via transforms
    mean_prop_background_px: float = 0.95
    training_setts: Dict[str, Any] = dict(
        # Loss function
        loss_fn=kpis.BalancedPenaltyReducedFocalLoss(
            factor_pos_class=mean_prop_background_px),  # torch.nn.KLDivLoss(),
        batch_size=8,
        batch_size_val=64,
        max_epochs=10,
        early_stopping_handle=models.EarlyStoppingHandle(0.0001, patience=2),
        optimizer=models.ResettableOptimizer(
            torch.optim.Adam,
            lr=0.001,
            # The weight decay to apply (L2 regularization)
            weight_decay=0.),
        # Transformation applied to (pred, target) tuple before loss and metrics
        model_output_transform=(
                trafos.SameSize(resize_target=False, interpolation="bilinear")
                # + trafos.OnInput(trafos.Lambda(torch.nn.LogSigmoid()))
                + trafos.OnInput(trafos.Lambda(torch.nn.Sigmoid()))
        ),
        metric_fns=dict(
            # KL-divergence on the gaussian masks
            kl_div=trafos.ReduceTuple(trafo=trafos.OnInput(torch.log),
                                      reduction=torch.nn.KLDivLoss()),
            # IoU values on the thresholded masks
            set_iou=kpis.SetIoU(),
            mean_iou=kpis.IoU(),
            # KL-divergence on the peak-filtered masks
            kl_div_peaks=trafos.ReduceTuple(
                trafo=(trafos.OnInput(torch.log)
                       + trafos.OnBothSides(trafos.BatchPeakDetection((3, 3)))),
                reduction=torch.nn.KLDivLoss()),
        )
    )
    # Transformation applied to the concept model output before visualization:
    visualization_transform: trafos.Transform = \
        trafos.Lambda(torch.nn.Sigmoid())

    part_infos: Dict[str, Dict[str, Any]] = dict(
        arm=dict(subparts=['LEFT_ELBOW', 'RIGHT_ELBOW']),  # detection
        leg=dict(subparts=['LEFT_KNEE', 'RIGHT_KNEE']),  # detection
    )

    # Resize the ground truth masks to the size of the activation map
    # respectively generate the ground truth to be of that size
    resize_gt_to_act_map_size: bool = True


class COCOHeatmapsDataGetter(COCODataGetter):
    """Get coco data train test tuple."""
    DATASET_CLASS = coco.HeatmapDataset
