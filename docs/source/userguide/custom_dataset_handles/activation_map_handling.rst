Activation Map Storage and Handling
===================================

The following shows exemplary usage of the wrapper
:py:class:`~hybrid_learning.datasets.activations_handle.ActivationDatasetWrapper`.
It can be used to automatically (lazily) retrieve outputs of DNNs, and store them
for repeated use using the pytorch pickling mechanism.
More precisely, the DNN should be a :py:class:`torch.nn.Module` that produces
:py:class:`torch.Tensor` outputs.
If all activation maps are already generated, the model need not be handed over again,
only the model description for identifying the activation maps.

The dataset then will yield tuples of the original input and the model output.

.. contents:: :local:


The following example shows how a COCO
:py:class:`~hybrid_learning.datasets.custom.coco.mask_dataset.ConceptDataset` is wrapped
to produce activation maps of the layer ``features.5`` of an AlexNet model.
It then yields tuples of

- the activation map tensor obtained from a (transformed) COCO image,
- the original concept annotation.


Preparation: Model and original dataset
---------------------------------------

To retrieve the activation maps, a
:py:class:`~hybrid_learning.concepts.models.model_extension.ModelStump`
can be used.

>>> from torchvision.models import alexnet
>>> from hybrid_learning.concepts.models.model_extension import ModelStump
>>> model = ModelStump(model=alexnet(pretrained=True),
...                    stump_head='features.5').eval()

Obtain the dataset to wrap. Note that the
:py:class:`~hybrid_learning.datasets.base.BaseDataset.dataset_root`
of the wrapped dataset will be used for that of the wrapper if no other
is given.
Also, the ground truth of the dataset is of no relevance, only the
input is used.

>>> from hybrid_learning.datasets.custom import coco
>>> import os
>>> root = os.path.join("dataset", "coco_test")
>>> concept_data = coco.ConceptDataset(
...     body_parts=[coco.BodyParts.FACE],
...     dataset_root=os.path.join(root, "images", "train2017"),
...     img_size=(400, 400),
... )


Wrapper init
------------

Now instantiate an
:py:class:`~hybrid_learning.datasets.activations_handle.ActivationDatasetWrapper`
to handle activation map retrieval. To automatically enable activation map
file caching, also hand over a cache root:

>>> from hybrid_learning.datasets import ActivationDatasetWrapper
>>> act_dataset = ActivationDatasetWrapper(
...     dataset=concept_data,
...     act_map_gen=model,
...     activations_root=os.path.join(root, "activations", "train2017_alexnet_features.5")
...  )
>>> act_map, mask_t = act_dataset[0]
>>> img_t = act_dataset.load_image(0)  # access the original input image

The activation maps in this case are usual conv layer outputs, i.e.
one activation map per filter:

>>> list(act_map.size())  # shape: (filters, width, height)
[192, 24, 24]
>>> # Show the activation map of the first filter:
>>> import PIL.Image
>>> from matplotlib import pyplot as plt
>>> import torchvision as tv
>>> a = tv.transforms.ToPILImage()(act_map[0])
>>> plt.imshow(a.resize((224, 224), resample=PIL.Image.BOX))
<matplotlib.image.AxesImage object...>


Force activation map rebuild
----------------------------

By default, activation maps are lazily generated and saved.
To force (re)build of all activation maps after init, one can directly
call :py:meth:`~hybrid_learning.datasets.activations_handle.ActivationDatasetWrapper.fill_cache`
(here we drastically reduce the amount of images before doing this):

>>> _ = act_dataset.dataset.subset(num=1)
>>> act_dataset.fill_cache(force_rebuild=True, leave=False)
ActivationDatasetWrapper(...)