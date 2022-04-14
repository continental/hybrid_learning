#  Copyright (c) 2022 Continental Automotive GmbH
"""Helper functions and model extensions to register model builders in a registry.
The registry can be used for easier configuration of model choices:
not the complete model has to be specified or stored for reproducibility,
but only the registry key.

Register model builders using ``register_model_builder``, and apply a builder
by registry key using ``get_model``.
Some custom model definitions can be found in
:py:mod:`~hybrid_learning.experimentation.model_registry.custom_model_postproc`.
"""

from .model_registry import *

