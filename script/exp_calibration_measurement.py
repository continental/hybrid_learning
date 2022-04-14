#  Copyright (c) 2022 Continental Automotive GmbH
"""Small experiment setup to measure evaluation metrics of
the torchvision Mask R-CNN model on MS COCO.
This is done for semantic segmentations, i.e. the COCO and Mask R-CNN instance
segmentations of one image are reduced to one segmentation mask.
Please call from project root."""

import logging
import os
from typing import Dict, Any, Tuple, List

import torch
import torchvision as tv
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

# noinspection PyUnresolvedReferences
from hybrid_learning.concepts import analysis, models, kpis
from hybrid_learning import datasets
from hybrid_learning.datasets import transforms as trafos
from hybrid_learning.concepts.train_eval import TrainEvalHandle, callbacks as cb
from hybrid_learning.datasets.custom import coco
from hybrid_learning.experimentation.model_registry.custom_model_postproc import MaskRCNNToSegMask, MaskRCNNBoxToSegMask, EfficientDetToSegMask
from hybrid_learning.experimentation.model_registry import register_model_builder, get_model
from sacred.observers import FileStorageObserver
ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds

ex.logger = logging.getLogger()


# noinspection PyUnusedLocal
@ex.config
def exp_config():
    # Mandatory config:
    # Model key; set using named config (and don't forget to register the model!)
    # Also used to create key for the trafo model applied to the main model output
    # to obtain the segmentation masks (as f'{model_key}_trafo')
    model_key: str = None
    assert model_key is not None, "model_key must be set via named config (together with model registration)!"

    # Root of the COCO dataset
    dataset_root: str = os.path.join('dataset', 'coco')
    split: datasets.DatasetSplit = datasets.DatasetSplit.VAL
    # Image size to which to pad and resize
    img_size: list = [400, 400]
    # The logical OR to use to reduce instance segmentations to semantic segmentation;
    # values may be keys of MaskRCNNToSegMask.LOGICAL_OR
    fuzzy_logic_key: str = 'goedel'
    # Whether the Mask R-CNN instance segmentations or the bounding boxes are used:
    use_boxes: bool = False

    # Settings for evaluation (note that no loss is needed)
    eval_setts: dict = dict(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_workers=1,  # The number of asynchronous workers to use; set to 0 to disable
        batch_size_val=64,  # Batch size for evaluation runs
        # Metrics
        metric_fns={'acc': kpis.Accuracy(),
                    'prec@050': kpis.Precision(),
                    'rec@050': kpis.Recall(),
                    'ECE': kpis.ECE(),
                    'MCE': kpis.ECE(aggregation="max", max_prob=False,
                                    class_conditional=True),
                    'MeanCE': kpis.ECE(aggregation="mean", max_prob=False,
                                       class_conditional=True),
                    # 'TACE001': kpis.ECE(
                    #     class_conditional=True, max_prob=False,
                    #     threshold_discard=0.01, adaptive=True),
                    'CC': kpis.CalibrationCurve(),
                    'PR': kpis.PrecisionRecallCurve(),
                    'SetIoUCurve': kpis.SetIoUThresholdCurve(),
                    'set_iou@050': kpis.SetIoU(),
                    'mean_iou@050': kpis.IoU(),
                    },
        show_progress_bars='always',  # show progress bars for evaluation
        callbacks=[],  # Tensorboard, CSV, progress bar, and normal logging are added automatically
        callback_context={'log_prefix': 'maskrcnn-calibration'}  # log_prefix used for logging folder structure
    )

    # Logging settings (disable any of the logging types by setting to None or "")
    experiment_root: str = os.path.join("experiments", "maskrcnn_calibration_test")  # Main log dir
    tensorboard_logdir: str = os.path.join(experiment_root, "tblogs")  # tensorboard log dir
    csv_logdir: str = os.path.join(experiment_root, "metrics")  # log dir for storing metric results as CSV
    sacred_logdir: str = os.path.join(experiment_root, "logs")  # log dir for sacred file observer


@ex.capture
def get_data(img_size: Tuple[int, int], dataset_root: str, use_boxes: bool) -> datasets.DataTriple:
    """Return data triple of COCO keypoint datasets
    with output transformed to semantic segmentation."""
    sets: Dict[str, coco.KeypointsDataset] = {}
    for split, dirname in ((datasets.DatasetSplit.TRAIN_VAL, 'train2017'),
                           (datasets.DatasetSplit.TEST, 'val2017')):
        dataset: coco.KeypointsDataset = coco.KeypointsDataset(
            dataset_root=os.path.join(dataset_root, 'images', dirname),
            split=split)

        anns_to_seg_trafo = coco.COCOBoxToSegMask(coco_handle=dataset.coco) if use_boxes \
            else coco.COCOSegToSegMask(coco_handle=dataset.coco)
        dataset.transforms = (trafos.OnInput(trafos.ToTensor())
                              + trafos.OnTarget(anns_to_seg_trafo)
                              + trafos.OnBothSides(trafos.PadAndResize(img_size)))
        sets[split.name.lower()] = dataset
    return datasets.DataTriple(**sets)


# noinspection PyUnusedLocal
@ex.config_hook
def config_postproc(config: Dict[str, Any],
                    command_name, logger) -> Dict[str, Any]:
    """Add sacred file logger."""
    config_addendum: Dict[str, Any] = {}

    if config["sacred_logdir"]:
        ex.observers.append(FileStorageObserver(config["sacred_logdir"]))

    # Add tensorboard callback
    new_callbacks: List[cb.Callback] = []
    if config["csv_logdir"]:
        new_callbacks.append(cb.CsvLoggingCallback(log_dir=config["csv_logdir"]))
    if config["tensorboard_logdir"]:
        new_callbacks.append(cb.TensorboardLogger(log_dir=config["tensorboard_logdir"],
                                                  log_sample_targets=True))
    config_addendum.setdefault("eval_setts", config.get("eval_setts", {})) \
        .setdefault("callbacks", []).extend(new_callbacks)

    return config_addendum


@ex.named_config
def maskrcnn():
    """Mask R-CNN model config."""
    model_key: str = 'maskrcnn'
    register_model_builder(
        model_key,
        lambda **conf: torch.nn.Sequential(
            # Mask R-CNN
            tv.models.detection.maskrcnn_resnet50_fpn(pretrained=True),
            # Trafo to turn Mask R-CNN output to semantic segmentation masks
            (MaskRCNNBoxToSegMask(fuzzy_logic=conf['fuzzy_logic_key'],
                                  image_size=conf['img_size']) if conf['use_boxes'] else
             MaskRCNNToSegMask(fuzzy_logic=conf['fuzzy_logic_key']))
        ).eval()
    )


@ex.named_config
def tf_efficientdet_d1():
    """Configuration for pretrained EfficientDet model variants
    from https://github.com/rwightman/efficientdet-pytorch.
    Install the effdet package for this:

    .. code:: shell

        # EfficientDet pytorch implementation
        pip install -e "git+https://github.com/rwightman/efficientdet-pytorch.git@75e16c2f#egg=effdet"
    """
    model_key = 'tf_efficientdet_d1'
    img_size: list = list(get_efficientdet_config()['image_size'])  # must be divisible by 2**7
    register_model_builder(
        model_key,
        lambda **conf: torch.nn.Sequential(
            get_efficientdet(**conf),
            EfficientDetToSegMask(image_size=conf["img_size"],
                                  fuzzy_logic=conf["fuzzy_logic_key"]).eval())
    )


@ex.capture
def get_efficientdet(img_size, model_key='tf_efficientdet_d1',
                     effdet_wrapper='predict', **_) -> torch.nn.Module:
    """Obtain a standard EfficientDet model.

    :param img_size: image size in (height, width); each dimension must be divisible by
        ``2**max_level`` (usually 128)
    :param model_key: the EfficientDet variant specifier; for options see
        https://github.com/rwightman/efficientdet-pytorch/blob/75e16c2f/effdet/config/model_config.py#L82
    :param effdet_wrapper: whether to wrap the EfficientDet model
        for prediction (``'prediction'``, add NMS), for training (``'train'``), or not at all (``''``)
    """
    effdet_config = get_efficientdet_config(model_key)
    effdet_config["image_size"] = list(img_size)
    import effdet
    model: torch.nn.Module = effdet.create_model_from_config(
        effdet_config, pretrained=True, bench_task=effdet_wrapper)
    return model


def get_efficientdet_config(model_key='tf_efficientdet_d1') -> dict:
    """Obtain a standard EfficientDet config.

    :param model_key: the efficientdet type; for options see
        https://github.com/rwightman/efficientdet-pytorch/blob/75e16c2f/effdet/config/model_config.py#L82"""
    try:
        import effdet.config
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Could not find module effdet. This is required to run experiments with EfficientDet variants. "
            "Please make sure to install it from github.com/rwightman/efficientdet-pytorch via\n"
            "\tpip install -e \"git+https://github.com/rwightman/efficientdet-pytorch.git@75e16c2f#egg=effdet\"")
    effdet_config = effdet.config.get_efficientdet_config(model_name=model_key)
    return effdet_config


@ex.main
def measure_maskrcnn_calibration(eval_setts, model_key, _config):
    """Run evaluation on test and train/val sets for Mask R-CNN semantic segmentation."""

    # Data
    data: datasets.DataTriple = get_data()

    # Model
    maskrcnn_seg = get_model(model_key=model_key, config=_config, check_layer_infos=False)

    # Evaluation handle
    eval_handle: TrainEvalHandle = TrainEvalHandle(
        model=maskrcnn_seg, data=data, **eval_setts)

    # Evaluation
    eval_handle.evaluate(mode=datasets.DatasetSplit.TEST)
    eval_handle.evaluate(mode=datasets.DatasetSplit.TRAIN_VAL)


if __name__ == "__main__":
    ex.run_commandline()
