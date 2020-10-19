Extending DNN model outputs
=============================

The module :py:mod:`~hybrid_learning.concepts.models.model_extension` provides
functionality to wrap models and extend their output by attaching
further modules to intermediate layers.
The base class for this is :py:class:`~hybrid_learning.concepts.models.model_extension.HooksHandle`:
It utilizes the
`pytorch hooking mechanism <https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks>`_
to extract intermediate layer outputs *(no code from the cited tutorial is used)*.

Derived classes are:

.. py:currentmodule:: hybrid_learning.concepts.models.model_extension
.. autosummary::
    :nosignatures:

    ~ActivationMapGrabber
    ~ModelExtender
    ~ModelStump
    ~ExtendedModelStump

Some exemplary applications are collected below.

.. contents:: :local:


Extend model output by activation maps
--------------------------------------
This can be achieved by the


Mini-example
............

Consider a simple example network composed of two linear operations
realizing :math:`tnn(x) = 3\cdot(x+2)`:

>>> import torch
>>> class SampleNN(torch.nn.Module):
...     def __init__(self):
...         super(SampleNN, self).__init__()
...         self.l1, self.l2 = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
...     def forward(self, x):
...         return self.l2(self.l1(x))
...
>>> nn: SampleNN = SampleNN()
>>> nn.l1.weight.data, nn.l1.bias.data, nn.l2.weight.data, nn.l2.bias.data \
...     = [torch.Tensor([i]) for i in [[1], 2, [3], 0]]


Now we can wrap the model and retrieve the output of ``l1``.
The output of the wrapper now consists of a tuple of

- the output of the wrapped model and
- a dict with the outputs of the registered sub-modules.

The output given 1 thus captures
:math:`nn(1) = 3(1+2) = 9` and
:math:`l1(1) = 1+2 = 3`:

>>> from hybrid_learning.concepts.models.model_extension import ActivationMapGrabber
>>> wnn = ActivationMapGrabber(nn, ['l1'])
>>> print(wnn(torch.Tensor([[1]])))
(tensor([[9.]]...), {'l1': tensor([[3.]]...)})

>>> from hybrid_learning.concepts.models.model_extension import ActivationMapGrabber
>>> wnn = ActivationMapGrabber(nn, ['l1'])
>>> print(wnn(torch.Tensor([[1]])))
(tensor([[9.]]...), {'l1': tensor([[3.]]...)})

>>> from hybrid_learning.concepts.models.model_extension import ActivationMapGrabber
>>> wnn = ActivationMapGrabber(nn, ['l1'])
>>> print(wnn(torch.Tensor([[1]])))
(tensor([[9.]]...), {'l1': tensor([[3.]]...)})


More complex example
....................

In the following a pre-trained Mask R-CNN model is wrapped to obtain
the outputs of the last two convolutional blocks in the backend.
These are handed over via a list of their keys in the
:py:meth:`~torch.nn.Module.named_modules` dict.
Note that they are registered in the order the IDs are given.

>>> from torchvision.models.detection import maskrcnn_resnet50_fpn
>>> nn = maskrcnn_resnet50_fpn(pretrained=True)
>>> wnn = ActivationMapGrabber(
...     model=nn,
...     module_ids=['backbone.body.layer3', 'backbone.body.layer4']
... )
>>> print(wnn.registered_submodules)
['backbone.body.layer3', 'backbone.body.layer4']

Manual un-/re-/registration of layers looks as follows
(note that newly registered layers are simply appended and a module
cannot be registered twice):

>>> wnn.unregister_submodule('backbone.body.layer3')  # unregister
>>> print(wnn.registered_submodules)
['backbone.body.layer4']
>>> wnn.register_submodule('backbone.body.layer3')    # (re-)register
>>> print(wnn.registered_submodules)
['backbone.body.layer4', 'backbone.body.layer3']
>>> wnn.register_submodule('backbone.body.layer4')    # register existing
>>> print(wnn.registered_submodules)
['backbone.body.layer4', 'backbone.body.layer3']

The output of the wrapped model now looks as follows:

>>> img_t = torch.rand(size=(3, 400,400))  # example
>>> wnn_out = wnn.eval()(img_t.unsqueeze(0))
>>> len(wnn_out)
2
>>> type(wnn_out[0])  # the Mask R-CNN output
<class 'list'>
>>> type(wnn_out[1])  # the layer outputs
<class 'dict'>
>>> for k in sorted(wnn_out[1].keys()):
...     print(k, ":", wnn_out[1][k].size())
backbone.body.layer3 : torch.Size([1, 1024, 50, 50])
backbone.body.layer4 : torch.Size([1, 2048, 25, 25])


Attaching Further Modules
-------------------------

An application of accessing DNN intermediate outputs is to extend
the DNN by attaching further (output) modules.
As an example, we will extend a Mask R-CNN by several fully connected
output neurons at two layers. The main model is:

>>> from torchvision.models.detection import maskrcnn_resnet50_fpn
>>> main_model = maskrcnn_resnet50_fpn(pretrained=True)


Defining the extended (wrapped) mdel
.....................................

To extend the model, hand over the model and a dictionary of extensions.
This dictionary maps layer IDs to named modules, i.e. dicts of
extension modules indexed by unique IDs.
Consider as extensions some simple linear models (mind the different
sizes of different layer outputs):

>>> import torch, numpy
>>> linear_attach = lambda in_size: torch.nn.Sequential(
...     torch.nn.Flatten(),
...     torch.nn.Linear(int(numpy.prod(in_size)), out_features=1)
... )
>>>
>>> from hybrid_learning.concepts.models.model_extension import ModelExtender
>>> extended = ModelExtender(
...     main_model, extensions={
...        'backbone.body.layer3': {'3_1': linear_attach([1024, 50, 50]),
...                                 '3_2': linear_attach([1024, 50, 50])},
...        'backbone.body.layer4': {'4_1': linear_attach([2048, 25, 25])},
...     })
>>> print(extended)
ModelExtender(
  (wrapped_model): MaskRCNN(...)
  (extension_models): ModuleDict(
    (3_1): Sequential(...)
    (3_2): Sequential(...)
    (4_1): Sequential(...)
  )
)

Each extension module ID must be unique amongst all IDs.
The mapping from layers and extensions attached to these layers is the
list of registrations:

>>> extended.name_registrations
{'backbone.body.layer3': ['3_1', '3_2'], 'backbone.body.layer4': ['4_1']}
>>> extended.extensions
{'backbone.body.layer3': {'3_1': Sequential(...), '3_2': Sequential(...)},
 'backbone.body.layer4': {'4_1': Sequential(...)}}


The extended output
...................

When evaluated, the output is a tuple of the Mask R-CNN output and
a dict with the output of each extension indexed by their ID.
This can now normally be backpropagated:

>>> out = extended.eval()(torch.rand((1, 3, 400, 400)))
>>> print(out)
([{'boxes': ...}],
 {'3_1': tensor(...),
  '3_2': tensor(...),
  '4_1': tensor(...)})
>>> _ = out[1]['3_1'].backward(torch.randn(1, 1))


Handling Registrations of Attachments
.....................................

For un-/(re-)registration of modules use
:py:meth:`~hybrid_learning.concepts.models.model_extension.ModelExtender.register_extensions`
and :py:meth:`~hybrid_learning.concepts.models.model_extension.ModelExtender.unregister_extension`:

>>> extended.register_extensions(
...     {'backbone.body.layer4': {'4_2': linear_attach([2048, 25, 25])}})
>>> extended.name_registrations
{'backbone.body.layer3': ['3_1', '3_2'],
 'backbone.body.layer4': ['4_1', '4_2']}
>>> extended.register_extensions(
...     {'backbone.body.layer4': {'4_2': linear_attach([2048, 25, 25])}})
Traceback (most recent call last):
  ...
ValueError: Tried to overwrite module under existing name: 4_2
>>>
>>> extended.unregister_extension('4_1')
>>> extended.name_registrations
{'backbone.body.layer3': ['3_1', '3_2'], 'backbone.body.layer4': ['4_2']}
>>> extended.unregister_extension('4_1')
Traceback (most recent call last):
  ...
KeyError: 'Tried to unregister extension of unknown name 4_1'