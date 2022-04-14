#  Copyright (c) 2022 Continental Automotive GmbH
"""Module to handle model registering.
This is a trick to flexibly exchange models via Sacred named configs
without sacred saving the complete model to the config file.
Use this for model configs if the config shall stay readable,
must not get too large, and your torch model anyways is not about
to change.

Usage:

- register functions returning a model via :py:func:`register_model_builder`
- build a model by key using :py:func:`get_model`
"""

from typing import Dict, Callable, Any, List, Iterable

import torch

_MODEL_GETTER_MAP: Dict[str, Callable[[Dict[str, Any]], torch.nn.Module]] = dict()
"""Model builder registry."""


def register_model_builder(model_key: str, model_getter: Callable[[Dict[str, Any]], torch.nn.Module]):
    """Register a model builder to later be callable via ``get_model``.
    The builder function must accept all entries of an experiment config
    dictionary and return a torch model."""
    if not callable(model_getter):
        raise ValueError("model_getter ({}) must be callable, but was of type {}"
                         .format(model_getter, type(model_getter)))
    _MODEL_GETTER_MAP[model_key] = model_getter


def registered_model_keys() -> List[str]:
    """Return a list of valid model builder keys."""
    return list(_MODEL_GETTER_MAP.keys())


def get_model(model_key: str, config: Dict[str, Any],
              layer_infos: Iterable[str] = None, check_layer_infos: bool = True
              ) -> torch.nn.Module:
    """Return the result of a model builder previously registered via ``register_model``.

    :param model_key: registered key
    :param layer_infos: iterable of layer IDs the model should feature
    :param check_layer_infos: whether to validate ``layer_infos``
    :param config: the config dict to hand over to the model builder
    :raises ValueError: if the built model is no torch model or ``layer_infos``
        is unset while ``check_layer_infos`` is ``True``
    :raises AssertionError: if ``check_layer_infos`` and one of the entries
        in ``layer_infos`` is not a valid layer identifier of the built model.
    :return: the built model
    """
    if model_key not in registered_model_keys():
        raise KeyError(("model_key {} unknown -- was it registered (see register_model)? "
                        "Registered model keys: {}").format(model_key, registered_model_keys()))
    model: torch.nn.Module = _MODEL_GETTER_MAP[model_key](**config)

    # region value checks
    # Correct type?
    if not isinstance(model, torch.nn.Module):
        raise ValueError(("The model builder registered under key {} did not yield a torch.nn.Module but "
                          "an object of type {}:\n{}").format(type(model), model))
    # Are all specified model layers valid?
    if check_layer_infos:
        main_model_modules = list(dict(model.named_modules()).keys())
        if layer_infos is None:
            raise ValueError("layer_infos unset but check_layer_infos is True! "
                             "Choose from the following layer keys:\n{}".format(main_model_modules))
        for layer_id in layer_infos:
            assert layer_id in main_model_modules, \
                ("Layer_id {} not in model. Choose from the following layer keys:\n{}"
                 .format(layer_id, main_model_modules))
    # endregion

    return model
