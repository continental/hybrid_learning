#  Copyright (c) 2022 Continental Automotive GmbH
"""
Helper function to register standard models suitable for fuzzy logic experiments.
"""

from hybrid_learning.experimentation.model_registry import custom_model_postproc
from .model_registry import register_model_builder
from ..fuzzy_exp.fuzzy_exp_helpers import get_logic
import torchvision as tv
import torch


def get_maskrcnn_box_trafo(**conf):
    return custom_model_postproc.MaskRCNNBoxToSegMask(
        image_size=conf["img_size"],
        fuzzy_logic=(get_logic(conf["fuzzy_logic_key"], conf["predicate_setts"], warn_about_unused_setts=False)
                     if conf["predicate_setts"].get('_', {}).get('reduce_pred_masks', True)
                     else None),
        ).eval()


def get_efficientdet_trafo(**conf):
    return custom_model_postproc.EfficientDetToSegMask(
        image_size=conf["img_size"],
        fuzzy_logic=(get_logic(conf["fuzzy_logic_key"], conf["predicate_setts"], warn_about_unused_setts=False)
                     if conf["predicate_setts"].get('_', {}).get('reduce_pred_masks', True)
                     else None),
        ).eval()


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


# ========================
# FILL MODEL REGISTRY
# ========================

def register_all_model_builders():
    """Register all model builders that may be used during fuzzy logic experiment."""
    register_model_builder(
        'maskrcnn_box',
        lambda **_: tv.models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval())
    register_model_builder(
        f'maskrcnn_box_trafo',
        get_maskrcnn_box_trafo)
    register_model_builder(
        'maskrcnn',
        lambda **_: tv.models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval())
    register_model_builder(
        f'maskrcnn_trafo',
        lambda **conf: custom_model_postproc.MaskRCNNToSegMask(
            fuzzy_logic=(get_logic(conf["fuzzy_logic_key"], conf["predicate_setts"])
                            if conf["predicate_setts"].get('_', {}).get('reduce_pred_masks', True)
                            else None),
        ).eval())
    register_model_builder(
        'tf_efficientdet_d1',
        get_efficientdet)
    register_model_builder(
        'tf_efficientdet_d1_trafo',
        get_efficientdet_trafo)
