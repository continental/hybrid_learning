"""Custom torch modules for specific torch models."""
#  Copyright (c) 2022 Continental Automotive GmbH

from .mask_rcnn_extensions import MaskRCNNToSegMask, MaskRCNNBoxToSegMask
from .efficientdet_extension import EfficientDetToSegMask
