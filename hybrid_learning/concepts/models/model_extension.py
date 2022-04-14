"""Wrapper classes to slice and extend ``torch.nn.modules``.
The main mechanism used are hooks to obtain layer intermediate output.
The base class to use this mechanism is :py:class:`HooksHandle`.

This is used to

- *extend the model* output by the intermediate output(s)
  (:py:class:`ActivationMapGrabber`),
- use the intermediate output to *attach further modules* to a model
  (:py:class:`ModelExtender`)
- *cut (and extend) a model* at a layer
  (:py:class:`ModelStump`, :py:class:`ExtendedModelStump`)
"""
#  Copyright (c) 2022 Continental Automotive GmbH

import abc
from typing import Iterable, Dict, Optional, List, Sequence, Tuple, Any, \
    Callable, Union

import torch
import torch.nn
import torch.utils.hooks


class _HookFor:
    """Callable that can act as hook for :py:class:`torch.nn.Module` and
    stores intermediate results using given handle."""
    def __init__(self, module_id: str, handle: 'HooksHandle'):
        """Init.

        :param module_id: the module ID under which to store the intermediate
            result; make sure the hook is really registered to this layer!
        :param handle: the handle to use to store the intermediate layer outputs
        """
        self.module_id: str = module_id
        self.handle: HooksHandle = handle

    def __call__(self, _module, _inp, outp: torch.Tensor
                 ) -> None:  # pylint: disable=unused-argument
        """Hook function that saves intermediate output of sub-module."""
        # noinspection PyProtectedMember
        self.handle._intermediate_outs[self.module_id] = outp


class HooksHandle(torch.nn.Module, abc.ABC):
    """Wrapper that registers and unregisters hooks from model that save
    intermediate output. For this, the pytorch hook mechanism is used.
    """

    def __init__(self, model: torch.nn.Module,
                 module_ids: Iterable[str] = None):
        """Init.

        :param model: the model to wrap
        :param module_ids: the IDs of sub-modules to obtain intermediate
            output from;

            .. note::
                If a sub-module is used several times, its output can only
                be captured after the first call.
        """
        module_ids = module_ids or []
        super().__init__()
        self.wrapped_model: torch.nn.Module = model
        """Original model from which intermediate and final output are
        retrieved."""

        self._intermediate_outs: Dict[str, Optional[torch.Tensor]] = \
            {m_id: None for m_id in module_ids}
        """Intermediate storage for outputs of hooked sub-modules."""

        self.hook_handles: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        """Dictionary of hooks; for each sub-module to grab output from,
        a hook is registered.
        On each forward, the hook for a sub-module of ID ``m`` writes the
        intermediate output of the sub-module into ``_intermediate_outs[m]``.
        The dictionary saves for the sub-module ID the hook handle.
        """
        for m_id in module_ids:
            self.register_submodule(m_id)

    @property
    def registered_submodules(self) -> List[str]:
        """List of IDs of the registered sub-modules."""
        return list(self.hook_handles.keys())

    def register_submodule(self, module_id: str) -> None:
        """Register further submodule of to extract intermediate output from.
        The ``module_id`` must be a valid name of a sub-module of
        :py:attr:`wrapped_model`."""
        if module_id in self.hook_handles.keys():
            return
        if module_id is None:
            raise ValueError("Tried to register a module_id which was None")

        sub_module = self.get_module_by_id(module_id)
        self.hook_handles[module_id] = sub_module.register_forward_hook(
            _HookFor(module_id=module_id, handle=self)
        )

    def unregister_submodule(self, module_id: str) -> None:
        """Unregister a submodule for intermediate output retrieval."""
        if module_id not in self.hook_handles.keys():
            raise KeyError("Tried to remove unknown submodule of ID {}"
                           .format(module_id))
        # remove hook from self.hooks
        hook_handle = self.hook_handles.pop(module_id)
        # unregister hook from self.model
        hook_handle.remove()

    def get_module_by_id(self, m_id):
        """Get actual sub-module object within wrapped model by module ID."""
        # Sub-module select may be replaced by the generic technique in Net2Vec
        named_modules = dict(self.wrapped_model.named_modules())
        if m_id not in named_modules.keys():
            raise KeyError(("Tried to register a hook to non-existing "
                            "sub-module {}; available sub-modules: {}"
                            ).format(m_id, list(named_modules.keys())))
        return named_modules[m_id]

    @abc.abstractmethod
    def forward(self, *inps):
        """Pytorch forward method."""
        raise NotImplementedError()

    def __del__(self):
        """Unregister all hooks held by this handle on handle delete."""
        for hook_handle in self.hook_handles.values():
            hook_handle.remove()


class ActivationMapGrabber(HooksHandle):
    r"""Wrapper class to obtain intermediate outputs from models.
    This is done using the hooking mechanism of :py:class:`torch.nn.Module`.

    The wrapper adds to the output of a model the intermediate output of
    specified sub-modules of it. The output of a forward pass then then is as
    tuple of the form
    ``(output_of_wrapped_model, {module_id: intermediate_out_of_sub_module})``.

    The module ID is the specifier with which the sub-module can be selected
    from :py:meth:`torch.nn.Module.named_modules` of the wrapped ``model``.

    The sub-modules can be registered and unregistered.
    The currently registered sub-modules to obtain intermediate output from and
    the corresponding hooks are stored in the dictionary
    :py:attr:`~HooksHandle.hook_handles`.
    """

    def __init__(self, model: torch.nn.Module,
                 module_ids: Iterable[str] = None):
        """Init.

        :param model: the model to wrap
        :param module_ids: the IDs of sub-modules to obtain intermediate output
            from;

            .. note:
                Sub-modules used several times will only be evaluated the first
                time called!
        """
        super().__init__(model=model, module_ids=module_ids)

    def forward(self, *inps: Sequence[torch.Tensor]
                ) -> Tuple[Any, Dict[str, Any]]:
        """Return tuple of outputs of the wrapped model and of the
        sub-modules."""
        model_out = self.wrapped_model(*inps)
        return model_out, self._intermediate_outs

    def stump(self, module_id: str) -> Callable:
        """Provide a :py:class:`ModelStump` (in eval mode) which yields act
        maps of given sub-module."""
        if module_id not in self.registered_submodules:
            raise KeyError("Submodule {} is not registered".format(module_id))
        return ModelStump(self.wrapped_model, stump_head=module_id).eval()


class ModelExtender(ActivationMapGrabber):
    """This class wraps a given model and extends its output.
    The extension are models taking intermediate output of the original model
    at given sub-modules.
    An extension is specified by the information in a dictionary
    ``{<sub-module ID> : {<name>: <model>}}``.

    where the sub-module must be one of the wrapped model, and the ``<model>``
    is the :py:class:`torch.nn.Module` to feed the sub-module output. The
    name must be unique amongst all registered models:
    It is checked when registering new extensions and used as key for the
    extension model outputs.

    Extensions can be registered and unregistered using the corresponding
    methods :py:meth:`register_extension` and :py:meth:`unregister_extension`.

    The information about registered extensions can be accessed via the
    following properties:

    - :py:attr:`extensions`:
      extension models indexed by sub-module ID in the format described above
    - :py:attr:`extension_models`:
      Just a dict-like with registered models by name
    - :py:attr:`name_registrations`:
      Just a dict with registered extension names by sub-module

    The output of a forward run then is a tuple of the main model output
    and a dict ``{<name>: <ext model output>}``.
    If :py:attr:`return_orig_out` is ``False``, only the dict is returned.
    """

    def __init__(self, model: torch.nn.Module,
                 extensions: Dict[str, Dict[str, torch.nn.Module]],
                 return_orig_out: bool = True):
        """Init.

        :param model: the model to extend
        :param extensions: extensions to initially register;
            for format see :py:attr:`register_extensions`
        :param return_orig_out: see :py:attr:`return_orig_out`
        """
        super().__init__(model=model,
                         module_ids=extensions.keys())
        for module_id, exts in extensions.items():
            for ext_name, ext_mod in exts.items():
                if not isinstance(ext_mod, torch.nn.Module):
                    raise ValueError(
                        ("Extension item of name {} to be registered at "
                         "sub-module {} not of type torch.nn.Module, but of "
                         "type {}").format(ext_name, module_id, type(ext_mod)))

        self.return_orig_out: bool = return_orig_out
        """Whether to return a tuple ``(original_output, extension_outputs)`` or
        only the dict ``extension_outputs``."""
        self.extension_models: torch.nn.ModuleDict = torch.nn.ModuleDict()
        """Dictionary of ``extension_models`` modules indexed by the layer they
        are applied to. Do only change via :py:meth:`register_extension` and
        :py:meth:`unregister_extension`, as the indices must be in
        synchronization with registered submodules."""

        self._name_registrations: Dict[str, List[str]] = {}
        """Dictionary mapping main model sub-modules to their registered
        extension model names. The names of the extensions must match those
        used as keys in :py:attr:`extension_models`:
        ``{<sub-module ID>: [<extension name>, ...]}``
        Do only change via :py:meth:`register_extension` and
        :py:meth:`unregister_extension`, as the indices must be in
        synchronization with registered submodules.
        """

        self.register_extensions(extensions)

    @property
    def name_registrations(self) -> Dict[str, List[str]]:
        """Dict mapping main model sub-modules to their registered
        extension model names.
        The names of the extensions must match those used as keys in
        :py:attr:`extension_models`:
        ``{<sub-module ID>: [<extension name>, ...]}``
        """
        return self._name_registrations

    @property
    def extensions(self) -> Dict[str, Dict[str, torch.nn.Module]]:
        """Nested dict holding all extension modules indexed by ID and layer.
        Merged information in :py:attr:`name_registrations` and
        :py:attr:`extension_models`.

        :return: Dict of the form
            ``{<sub-module ID>: {<ext name>: <registered ext model>}}``
            The name is unique amongst all registered extension models over
            all sub-modules
        """
        return {sub_mod: {e_name: self.extension_models[e_name]
                          for e_name in e_names}
                for sub_mod, e_names in self.name_registrations.items()}

    @property
    def extension_names(self) -> List[str]:
        """List of the names of all registered extensions."""
        return list(self.extension_models.keys())

    def register_extension(self, name: str, module_id: str,
                           model: torch.nn.Module) -> None:
        """Register a new extension model as name.
        Updates the hooks needed for acquiring extension output.

        :raise: :py:exc:`ValueError` if there is a name for which already an
            extension is registered.
        """
        if name in self.extension_names:
            raise ValueError(("Tried to overwrite module under existing name: "
                              "{}").format(name))

        # update hooks
        self.register_submodule(module_id=module_id)

        # update corresponding model registration list
        if module_id not in self.name_registrations.keys():
            self._name_registrations[module_id] = []
        self._name_registrations[module_id].append(name)

        # update model list
        self.extension_models.update({name: model})

    def unregister_extension(self, name: str) -> None:
        """Unregister an existing extension by name.
        Updates the hooks and the registration lists."""
        # Model not registered?
        if name not in self.extension_names:
            raise KeyError("Tried to unregister extension of unknown name {}"
                           .format(name))

        # update corresponding model registration list
        module_id: str = [m_id for m_id in self.name_registrations
                          if name in self.name_registrations[m_id]][0]
        self._name_registrations[module_id].pop(
            self._name_registrations[module_id].index(name))

        # if now no extension is registered at a sub-module, remove its
        # registration
        if len(self._name_registrations[module_id]) == 0:
            self._name_registrations.pop(module_id)
            # clean up hooks
            self.unregister_submodule(module_id)

        # update model list
        self.extension_models.pop(name)

    def register_extensions(
            self,
            new_extensions: Dict[str, Dict[str, torch.nn.Module]]) -> None:
        """Register all specified new extensions.

        :param new_extensions: extensions in the format
            ``{module_id: {extension_name: extension_module}}``
        :raise: :py:exc:`ValueError` if there is a name for which already an
            extension is registered.
        """
        for module_id, exts in new_extensions.items():
            for name, model in exts.items():
                self.register_extension(name=name, module_id=module_id,
                                        model=model)

    def forward(self, *inps: Sequence[torch.Tensor]
                ) -> Union[Tuple[Any, Dict[str, Any]], Dict[str, Any]]:
        """Pytorch forward method.

        :return: Tuple of the form
            ``(<main model out>, {<ext name>: <ext out>})``.
        """
        # Get the sub-module intermediate outputs by sub-module ID:
        extended_out = super().forward(*inps)
        main_model_out: Any = extended_out[0]
        intermediate_outs: Dict[str, Any] = extended_out[1]

        # Now feed the intermediate outputs to the correspondingly registered
        # extension models:
        extension_outs: Dict[str, Any] = {}
        for module_id, intermediate_out in intermediate_outs.items():
            for name in self.name_registrations[module_id]:
                extension_outs[name] = self.extension_models[name](
                    intermediate_out)

        if self.return_orig_out:
            return main_model_out, extension_outs
        else:
            return extension_outs


class ModelStump(HooksHandle):
    # pylint: disable=line-too-long
    """Obtain the intermediate output of a sub-module of a complete NN.
    This is a smaller version of the
    :py:class:`~hybrid_learning.concepts.models.model_extension.ActivationMapGrabber`:

    - It only handles the output of one sub-module, its stump head.
    - It does not retrieve the output of the main model.

    In the other points it is the same as
    :py:class:`~hybrid_learning.concepts.models.model_extension.ActivationMapGrabber`.
    """
    # pylint: enable=line-too-long

    def __init__(self, model: torch.nn.Module, stump_head: str):
        """Init.

        :param model: model to obtain intermediate output from.
        :param stump_head: ID of the sub-module from which to obtain
            intermediate output.

            .. note::
                If the sub-module occurs several times, only the first output
                is collected.
        """

        super().__init__(model=model)
        self._stump_head: Optional[str] = None
        self.stump_head = stump_head

        # Most current applications require eval mode, so make this the default:
        self.eval()

    @property
    def stump_head(self) -> str:
        """ID of the sub-module from which the activation maps are retrieved."""
        return self._stump_head

    @stump_head.setter
    def stump_head(self, stump_head: str):
        """Before setting the stump_head, make sure it is registered
        for act map retrieval."""
        if stump_head is not None \
                and stump_head not in self.registered_submodules:
            self.register_submodule(module_id=stump_head)
        self._stump_head = stump_head

    def register_submodule(self, module_id: str) -> None:
        """Register a sub-module hook.
        If :py:attr:`stump_head` is unset, set it to this sub-module."""
        super().register_submodule(module_id=module_id)
        # Set stump to be the new and only registered sub-module if it is not
        # set:
        if self.stump_head is None:
            self.stump_head = module_id

    def unregister_submodule(self, module_id: str) -> None:
        """Unregister a submodule for intermediate output retrieval."""
        super().unregister_submodule(module_id=module_id)
        if module_id == self.stump_head:
            self.stump_head = None

    def forward(self, *inps):
        """Pytorch forward method: Return intermediate output of stump head.
        Provides __call__ functionality."""
        _ = self.wrapped_model(*inps)
        act_map = self._intermediate_outs[self.stump_head]
        return act_map


class ExtendedModelStump(ModelStump):
    """Optionally apply a modification to the model stump output
    in the forward method.
    Use this e.g. to select one of multiple outputs of a module:

    .. code-block:: python

        ExtendedModelStump(model, stump_head, lambda x: x['out_of_interest'])
    """

    def __init__(self, model: torch.nn.Module, stump_head: str,
                 modification: Callable):
        super().__init__(model, stump_head)
        if not callable(modification):
            raise ValueError("modification function for model intermediate "
                             "out must be callable!")
        self.modification: Callable = modification

    def forward(self, *inps):
        """Collect output of stump head & return ``modification(stump_head)``"""
        outp = super().forward(*inps)
        return self.modification(outp)


def dummy_output(model: torch.nn.Module, input_size: Sequence[int],
                 layer_ids: Sequence[str] = None) -> Dict[str, Any]:
    """Select dummy output of model's given or all layers for all-zero
    tensor of ``input_size``.

    :param input_size: input size of one sample to feed in
        (make sure to include batch dimension!)
    :param model: the model to investigate
    :param layer_ids: the layers to investigate;
        defaults to all listed in the model's
        :py:attr:`~torch.nn.Module.named_modules`
    :return: a dict of the outputs for each layer ID
    """
    with torch.set_grad_enabled(False):
        # pylint: disable=no-member
        inp_tensor: torch.Tensor = torch.zeros(size=tuple(input_size))
        # pylint: enable=no-member
        if len(list(model.parameters())) > 0:
            inp_tensor = inp_tensor.to(next(model.parameters()).device)

        layer_ids = layer_ids or [name for name, _ in model.named_modules()]
        grabber = ActivationMapGrabber(model, layer_ids)
        _, outp = grabber.eval()(inp_tensor)
        grabber.__del__()
    return outp


def output_sizes(model: torch.nn.Module, input_size: Sequence[int],
                 layer_ids: Sequence[str] = None,
                 has_batch_dim: bool = False,
                 ignore_non_tensor_outs: bool = True) -> Dict[str, torch.Size]:
    """Obtain the output sizes of the given or all layers for given input size.
    The layer outputs are obtained by feeding the model an all zero Tensor of
    given input size. The output size can only be determined, if the output
    of the layer is a :py:class:`torch.Tensor`.
    Other entries are skipped, if ``ignore_non_tensor_outs=True``, and
    otherwise, an exception is raised. In such cases, consider using
    :py:func:`~hybrid_learning.concepts.models.model_extension.dummy_output`
    directly.

    :param ignore_non_tensor_outs:
    :param input_size: input size of one sample to feed in; it is assumed to
        have no batch dimension, if ``has_batch_dim == False``
    :param model: the model to investigate
    :param layer_ids: the layers to investigate; defaults to all listed in the
        model's :py:attr:`~torch.nn.Module.named_modules`
    :param has_batch_dim: whether the given tensor has batch dimension;
        if not, it is added
    :return: a dict of the sizes for each layer output tensor
        (batch dimension stripped)
    :raises: :py:exc:`AttributeError`, if one of the considered layers does
        not output a tensor (but e.g. a dict) and
        ``ignore_non_tensor_outs == False``; use
        :py:func:`~hybrid_learning.concepts.models.model_extension.dummy_output`
        directly in this case
    """
    layer_ids = layer_ids or [name for name, _ in model.named_modules()]
    if len(input_size) == 0:
        raise ValueError("Empty input_size not supported.")
    if not has_batch_dim:
        input_size = [1, *input_size]

    outps_sizes: Dict[str, torch.Size] = {}
    for layer_id, outp in dummy_output(model, input_size, layer_ids).items():
        if not isinstance(outp, torch.Tensor):
            if not ignore_non_tensor_outs:
                raise AttributeError(
                    ("Output of layer {} was not of type torch.Tensor but {};"
                     "non-tensor types not supported for determining size.\n"
                     "model: {}").format(layer_id, type(outp), str(model)))
        else:
            outp_size: torch.Size = outp.size()
            if len(outp_size) <= 1:
                raise AttributeError(
                    ("Output size of layer {} is 1D ({}) and does not include "
                     "batch dimension").format(layer_id, outp_size))
            outps_sizes[layer_id] = outp.size()[1:]  # strip batch dimension
    return outps_sizes


def output_size(model: torch.nn.Module, input_size: Sequence[int],
                has_batch_dim: bool = False, layer_id: Optional[str] = None,
                ) -> torch.Size:
    """Feed dummy input of input_size to model to determine the output size
    of the layer with ID ``layer_id``.
    If no layer ID is given, the size of the model final output is returned.

    Will raise if the model output is not a single tensor.
    This essentially is a wrapper around
    :py:func:`~hybrid_learning.concepts.models.model_extension.output_sizes`
    that uses the fact that the model also occurs in its
    :py:meth:`torch.nn.Module.named_modules` listing, with ID ``''``.
    So, the model output can also be obtained by registering a hook to ``''``.

    :param model: the model the output of which is to be investigated
    :param input_size: a single, all-zero tensor of that size is fed to
        the model
    :param layer_id: the ID of the layer to inspect;
        if not given, all layers (i.e. ``''``) are inspected
    :param has_batch_dim: whether the given ``input_size`` already features
        a batch dimension; added if not
    """
    layer_id = layer_id or ''
    return output_sizes(model,
                        layer_ids=[layer_id],
                        input_size=input_size,
                        has_batch_dim=has_batch_dim,
                        ignore_non_tensor_outs=False
                        )[layer_id]
