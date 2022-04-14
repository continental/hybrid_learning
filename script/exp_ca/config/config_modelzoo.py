#  Copyright (c) 2022 Continental Automotive GmbH
"""Sacred script to conduct concept analysis for different models from
pytorch model zoo on a subset of MS COCO with concept maskings.
The default chosen model is AlexNet.
To choose another model, specify its named configuration via

.. code:: bash
    python script/exp_coco_modelzoo.py with <mynet>

Call

.. code:: bash
    python script/exp_coco_modelzoo.py print_named_configs

to print all available network configurations.

Additionally required python modules are only loaded within the respective
named config. Make sure to follow the instructions in their docstrings
(ar error messages) for installation.
"""

# pylint: disable=no-name-in-module,import-error
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
# pylint: disable=unused-variable,unused-argument,unused-import

from typing import List

import torch
import torchvision as tv

from .config_ca import ex, register_model_builder


def _resnet_block_ids(num_conv_layers: int) -> List[str]:
    """For a standard ResNet size return the list of block IDs."""
    assert num_conv_layers in (50, 101)
    num_layer3_blocks = 6 if num_conv_layers == 50 else 23
    return ["1.0", "1.1", "1.2",
            "2.0", "2.1", "2.2", "2.3",
            *["3.{}".format(i) for i in range(num_layer3_blocks)],
            "4.0", "4.1", "4.2"]


def _resnet_common_layers(num_conv_layers: int) -> List[str]:
    """For a standard ResNet size collect the IDs of all layers of interest."""
    assert num_conv_layers in (50, 101)
    return [
        "relu",  # Conv 1
        "maxpool",  # MaxPool2d 1
        # The interesting layers for each block:
        *sum([["layer{}.bn1".format(i),  # Conv & BN
               "layer{}.bn2".format(i),  # Conv & BN
               "layer{}.relu".format(i),  # Conv&BN&ReL
               "layer{}".format(i),  # final bottleneck output
               ] for i in _resnet_block_ids(num_conv_layers)], []),
        "layer1.0.downsample",  # downsample Conv & BN
        "layer2.0.downsample",  # downsample Conv & BN
        "layer3.0.downsample",  # downsample Conv & BN
        "layer4.0.downsample",  # downsample Conv & BN
    ]


RESNET50_COMMON_LAYERS: List[str] = _resnet_common_layers(50)
"""The IDs of layers of ResNet50 which are of interest for concept analysis."""
RESNET101_COMMON_LAYERS: List[str] = _resnet_common_layers(101)
"""The IDs of layers of ResNet101 which are of interest for concept analysis."""


# noinspection PyUnusedLocal
@ex.named_config
def alexnet():
    """Configuration for pretrained AlexNet model from pytorch model zoo."""
    # pylint: disable=unused-variable
    img_size: list = [224, 224]
    model_key = 'alexnet'
    register_model_builder(model_key, lambda: tv.models.alexnet(pretrained=True, progress=False))
    layer_infos: list = [
        "features.1",  # Conv 1
        "features.2",  # MaxPool2d 1
        "features.4",  # Conv 2
        "features.5",  # MaxPool2d 2
        "features.7",  # Conv 3
        "features.9",  # Conv 4
        "features.11",  # Conv 5
        "features.12",  # MaxPool2d 3
        "avgpool",  # AdaptiveAvgPool2d 1
    ]


# noinspection PyUnusedLocal
@ex.named_config
def vgg16():
    """Configuration for pretrained VGG16 model from pytorch model zoo."""
    # pylint: disable=unused-variable
    img_size: list = [224, 224]
    model_key = 'vgg16'
    register_model_builder(model_key, lambda: tv.models.vgg16(pretrained=True, progress=False))
    layer_infos: list = [
        "features.1",  # Conv 1
        "features.3",  # Conv 2
        "features.4",  # MaxPool2d 1
        "features.6",  # Conv 3
        "features.8",  # Conv 4
        "features.9",  # MaxPool2d 2
        "features.11",  # Conv 5
        "features.13",  # Conv 6
        "features.15",  # Conv 7
        "features.16",  # MaxPool2d 3
        "features.18",  # Conv 8
        "features.20",  # Conv 9
        "features.22",  # Conv 10
        "features.23",  # MaxPool2d 4
        "features.25",  # Conv 11
        "features.27",  # Conv 12
        "features.29",  # Conv 13
        "features.30",  # MaxPool2d 5
        "avgpool",  # AdaptiveAvgPool2d 1
    ]


# noinspection PyUnusedLocal
@ex.named_config
def mobilenetv2():
    """Configuration for pretrained MobileNetV2 model from pytorch model zoo."""
    # pylint: disable=unused-variable
    img_size: list = [400, 400]
    model_key = 'mobilenetv2'
    register_model_builder(model_key, lambda: tv.models.mobilenet_v2(pretrained=True, progress=False))
    layer_infos: list = [
        "features.0",  # ConvBNReLU 1
        "features.1",  # InvertedResidual 1
        "features.1.conv",  # Sequential
        "features.1.conv.0",  # ConvBNReLU
        # InvertedResidual 2-17:
        *["features.{}".format(i) for i in range(2, 18)],
        # Sequential 2-17:
        *["features.{}.conv".format(i) for i in range(2, 18)],
        # ConvBNReLU (2-17).0
        *["features.{}.conv.0".format(i) for i in range(2, 18)],
        # ConvBNReLU (2-17).1
        *["features.{}.conv.1".format(i) for i in range(2, 18)],
        "features.18",  # ConvBNReLU 35
    ]


# noinspection PyUnusedLocal
@ex.named_config
def resnet50():
    """Configuration for pretrained ResNet50 model from pytorch model zoo."""
    # pylint: disable=unused-variable
    img_size: list = [224, 224]
    model_key = 'resnet50'
    register_model_builder(model_key, lambda: tv.models.resnet50(pretrained=True, progress=False))
    layer_infos: list = [
        *RESNET50_COMMON_LAYERS,
        "avgpool",  # AdaptiveAvgPool2d
    ]


# noinspection PyUnusedLocal
@ex.named_config
def mask_rcnn():
    """Configuration for pretrained Mask R-CNN model from pytorch model zoo."""
    # pylint: disable=unused-variable
    img_size: list = [400, 400]
    model_key = 'mask_rcnn'
    register_model_builder(model_key,
                           lambda: tv.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=False))
    layer_infos: list = [
        *["backbone.body.{}".format(layer)
          for layer in RESNET50_COMMON_LAYERS],
        "backbone.fpn",  # FeaturePyramidNetwork 1
        "backbone.fpn.inner_blocks.0",  # Conv2d 54
        "backbone.fpn.inner_blocks.1",  # Conv2d 55
        "backbone.fpn.inner_blocks.2",  # Conv2d 56
        "backbone.fpn.inner_blocks.3",  # Conv2d 57
        "backbone.fpn.layer_blocks.0",  # Conv2d 58
        "backbone.fpn.layer_blocks.1",  # Conv2d 59
        "backbone.fpn.layer_blocks.2",  # Conv2d 60
        "backbone.fpn.layer_blocks.3",  # Conv2d 61
        "backbone.fpn.extra_blocks",  # LastLevelMaxPool 1
    ]


# noinspection PyUnusedLocal
@ex.named_config
def deeplabv3():
    """Configuration for pretrained DeeplabV3 model from pytorch model zoo."""
    # pylint: disable=unused-variable
    img_size: list = [400, 400]
    model_key = 'deeplabv3'
    register_model_builder(model_key,
                           lambda: tv.models.segmentation.deeplabv3_resnet50(pretrained=True, progress=False))
    layer_infos: list = [
        *["backbone.{}".format(layer)
          for layer in RESNET50_COMMON_LAYERS],
        "classifier.0",  # ASPP 1
        ["classifier.0.convs.{}".format(i)  # Conv & BN & ReLU 2
         for i in range(1, 4)],
        "classifier.0.convs.4.0",  # AdaptiveAvgPool2d
        "classifier.0.convs.4.3",  # Conv & BN & ReLU 5 (after AVGPool)
        "classifier.0.project",  # Conv & BN & ReLU 6
        "classifier.2",  # Conv & BN & ReLU 7
    ]


# noinspection PyUnusedLocal
@ex.named_config
def fcnresnet50():
    """Configuration for pretrained FCN ResNet50 model from pytorch model
    zoo."""
    # pylint: disable=unused-variable
    img_size: list = [400, 400]
    model_key = 'fcnresnet50'
    register_model_builder(model_key,
                           lambda: tv.models.segmentation.fcn_resnet50(pretrained=True, progress=False))
    layer_infos: list = [
        *["backbone.{}".format(layer)
          for layer in RESNET50_COMMON_LAYERS],
        "classifier.2",  # Conv & BN & ReLU
    ]


# noinspection PyUnusedLocal
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
    register_model_builder(model_key, get_efficientdet)
    img_size: list = list(get_efficientdet_config()['image_size'])  # must be divisible by 2**7
    layer_infos: list = [("model." + layer)  # if effdet_wrapper else layer
                         for layer in [
                             # The EfficientNet backbone residual blocks
                             *[f'backbone.blocks.{i}' for i in [1, 2, 3, 4, 5, 6]],
                             # The lowest resolution nodes in the BiFPN layers
                             *[f'fpn.cell.{i}.fnode.7' for i in [0, 1, 2, 3]],
                             # The highest resolution nodes in the BiFPN layers
                             *[f'fpn.cell.{i}.fnode.3' for i in [0, 1, 2, 3]],
                         ]]


@ex.capture
def get_efficientdet(img_size, model_key='tf_efficientdet_d1',
                     effdet_wrapper='predict') -> torch.nn.Module:
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


if __name__ == "__main__":
    ex.run_commandline()
