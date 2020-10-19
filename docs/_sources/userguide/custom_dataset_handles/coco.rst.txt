MS COCO
=======

.. contents:: :local:

Simple Keypoints Dataset
------------------------

Create a standard :py:class:`~hybrid_learning.datasets.custom.coco.KeypointsDataset` via

>>> import os
>>> root = os.path.join("dataset", "coco_test")
>>> from hybrid_learning.datasets.custom.coco import KeypointsDataset
>>> keyptsdata = KeypointsDataset(
...     dataset_root=os.path.join(root, "images", "train2017")
... )

Restrict it to commercial licenses,
certain keypoints, maximum number of (random) images, etc. as follows
(note that by default the datasets are always subsetted to
commercial licenses):

>>> from hybrid_learning.datasets.custom.coco import BodyParts
>>> keyptsdata.subset(
...     license_ids=keyptsdata.COMMERCIAL_LICENSE_IDS,  # default
...     body_parts=[BodyParts.FACE],
...     num=10, shuffle=True
... )
KeypointsDataset(...)

Now one can retrieve tuples with the desired properties from the
dataset, which can be visualized via pyplot:

>>> img_t, anns = keyptsdata[0]
>>>
>>> from matplotlib import pyplot as plt
>>> import PIL.Image
>>> import torchvision as tv
>>> plt.imshow(tv.transforms.ToPILImage()(img_t))
<matplotlib.image.AxesImage object...>

Also other tuple transformations can be specified accepting
the input image (:py

>>> import os
>>> root = os.path.join("dataset", "coco_test")
>>> from hybrid_learning.datasets.custom.coco import KeypointsDataset
>>> keyptsdata = KeypointsDataset(
...     dataset_root=os.path.join(root, "images", "train2017")
... )

Restrict it to commercial licenses,
certain keypoints, maximum number of (random) images, etc. as follows
(note that by default the datasets are always subsetted to
commercial licenses):

>>> from hybrid_learning.datasets.custom.coco import BodyParts
>>> keyptsdata.subset(
...     license_ids=keyptsdata.COMMERCIAL_LICENSE_IDS,  # default
...     body_parts=[BodyParts.FACE],
...     num=10, shuffle=True
... )
KeypointsDataset(...)

Now one can retrieve tuples with the desired properties from the
dataset, which can be visualized via pyplot:

>>> img_t, anns = keyptsdata[0]
>>>
>>> from matplotlib import pyplot as plt
>>> import PIL.Image
>>> import torchvision as tv
>>> plt.imshow(tv.transforms.ToPILImage()(img_t))
<matplotlib.image.AxesImage object...>

Also other tuple transformations can be specified accepting
the input image (:py

>>> import os
>>> root = os.path.join("dataset", "coco_test")
>>> from hybrid_learning.datasets.custom.coco import KeypointsDataset
>>> keyptsdata = KeypointsDataset(
...     dataset_root=os.path.join(root, "images", "train2017")
... )

Restrict it to commercial licenses,
certain keypoints, maximum number of (random) images, etc. as follows
(note that by default the datasets are always subsetted to
commercial licenses):

>>> from hybrid_learning.datasets.custom.coco import BodyParts
>>> keyptsdata.subset(
...     license_ids=keyptsdata.COMMERCIAL_LICENSE_IDS,  # default
...     body_parts=[BodyParts.FACE],
...     num=10, shuffle=True
... )
KeypointsDataset(...)

Now one can retrieve tuples with the desired properties from the
dataset, which can be visualized via pyplot:

>>> img_t, anns = keyptsdata[0]
>>>
>>> from matplotlib import pyplot as plt
>>> import PIL.Image
>>> import torchvision as tv
>>> plt.imshow(tv.transforms.ToPILImage()(img_t))
<matplotlib.image.AxesImage object...>

Also other tuple transformations can be specified accepting
the input image (:py

>>> import os
>>> root = os.path.join("dataset", "coco_test")
>>> from hybrid_learning.datasets.custom.coco import KeypointsDataset
>>> keyptsdata = KeypointsDataset(
...     dataset_root=os.path.join(root, "images", "train2017")
... )

Restrict it to commercial licenses,
certain keypoints, maximum number of (random) images, etc. as follows
(note that by default the datasets are always subsetted to
commercial licenses):

>>> from hybrid_learning.datasets.custom.coco import BodyParts
>>> keyptsdata.subset(
...     license_ids=keyptsdata.COMMERCIAL_LICENSE_IDS,  # default
...     body_parts=[BodyParts.FACE],
...     num=10, shuffle=True
... )
KeypointsDataset(...)

Now one can retrieve tuples with the desired properties from the
dataset, which can be visualized via pyplot:

>>> img_t, anns = keyptsdata[0]
>>>
>>> from matplotlib import pyplot as plt
>>> import PIL.Image
>>> import torchvision as tv
>>> plt.imshow(tv.transforms.ToPILImage()(img_t))
<matplotlib.image.AxesImage object...>

Also other tuple transformations can be specified accepting
the input image (:py

>>> import os
>>> root = os.path.join("dataset", "coco_test")
>>> from hybrid_learning.datasets.custom.coco import KeypointsDataset
>>> keyptsdata = KeypointsDataset(
...     dataset_root=os.path.join(root, "images", "train2017")
... )

Restrict it to commercial licenses,
certain keypoints, maximum number of (random) images, etc. as follows
(note that by default the datasets are always subsetted to
commercial licenses):

>>> from hybrid_learning.datasets.custom.coco import BodyParts
>>> keyptsdata.subset(
...     license_ids=keyptsdata.COMMERCIAL_LICENSE_IDS,  # default
...     body_parts=[BodyParts.FACE],
...     num=10, shuffle=True
... )
KeypointsDataset(...)

Now one can retrieve tuples with the desired properties from the
dataset, which can be visualized via pyplot:

>>> img_t, anns = keyptsdata[0]
>>>
>>> from matplotlib import pyplot as plt
>>> import PIL.Image
>>> import torchvision as tv
>>> plt.imshow(tv.transforms.ToPILImage()(img_t))
<matplotlib.image.AxesImage object...>

Also other tuple transformations can be specified accepting
the input image (:py:class:`PIL.Image.Image`) and the ground truth
annotation (``dict``).
E.g. to use ``pycocotools.COCO.showAnns`` for visualizing the
keypoint annotations, one must not change the image size:

>>> # Change the transformation:
>>> import torchvision as tv
>>> to_tens = tv.transforms.ToTensor()
>>> keyptsdata.transforms = lambda i, gt: (to_tens(i), gt)  # identity
>>>
>>> # Select the image and annotations
>>> img_t, anns = keyptsdata[0]
>>>
>>> # Show the image with annotations
>>> plt.imshow(tv.transforms.ToPILImage()(img_t))
<matplotlib.image.AxesImage object...>
>>> keyptsdata.coco.showAnns(anns)


Concept Segmentation Dataset
----------------------------

Given the keypoints, one can estimate the segmentation of
:py:class:`body parts <hybrid_learning.datasets.custom.coco.BodyParts>`.
This is used in :py:class:`~hybrid_learning.datasets.custom.coco.ConceptDataset`
to generate (and store) segmentation masks for a given body part.

Create a standard :py:class:`~hybrid_learning.datasets.custom.coco.ConceptDataset` via

>>> root = os.path.join("dataset", "coco_test")
>>> from hybrid_learning.datasets.custom.coco import ConceptDataset, BodyParts
>>> concept_data = ConceptDataset(
...     body_parts=[BodyParts.FACE],
...     annotations_fp=os.path.join(root, "annotations",
...                                 "person_keypoints_train2017.json"),
...     dataset_root=os.path.join(root, "images", "train2017")
... )
>>> # The masked keypoints occuring in the body parts are:
>>> sorted(concept_data.keypoint_names)
['left_eye', 'nose', 'right_eye']

By default, the dataset is not restricted to only samples containing
the specified body parts. For this (and restriction to the correct
license and length) one can use
:py

>>> root = os.path.join("dataset", "coco_test")
>>> from hybrid_learning.datasets.custom.coco import ConceptDataset, BodyParts
>>> concept_data = ConceptDataset(
...     body_parts=[BodyParts.FACE],
...     annotations_fp=os.path.join(root, "annotations",
...                                 "person_keypoints_train2017.json"),
...     dataset_root=os.path.join(root, "images", "train2017")
... )
>>> # The masked keypoints occuring in the body parts are:
>>> sorted(concept_data.keypoint_names)
['left_eye', 'nose', 'right_eye']

By default, the dataset is not restricted to only samples containing
the specified body parts. For this (and restriction to the correct
license and length) one can use
:py

>>> root = os.path.join("dataset", "coco_test")
>>> from hybrid_learning.datasets.custom.coco import ConceptDataset, BodyParts
>>> concept_data = ConceptDataset(
...     body_parts=[BodyParts.FACE],
...     annotations_fp=os.path.join(root, "annotations",
...                                 "person_keypoints_train2017.json"),
...     dataset_root=os.path.join(root, "images", "train2017")
... )
>>> # The masked keypoints occuring in the body parts are:
>>> sorted(concept_data.keypoint_names)
['left_eye', 'nose', 'right_eye']

By default, the dataset is not restricted to only samples containing
the specified body parts. For this (and restriction to the correct
license and length) one can use
:py:meth:`~hybrid_learning.datasets.custom.coco.COCODataset.subset`:

>>> concept_data.subset(body_parts=[BodyParts.FACE])
ConceptDataset(...)

To have a look at some examples, one can use
:py:meth:`~hybrid_learning.datasets.data_visualization.apply_mask` to visualize the
ground truth masks for the concept created in
:py:meth:`~hybrid_learning.datasets.custom.coco.ConceptDataset.annotations_to_mask`.

>>> img_t, mask_t = concept_data[0]
>>> import torchvision as tv
>>> to_img = tv.transforms.ToPILImage()
>>> from matplotlib import pyplot as plt
>>> from hybrid_learning.datasets import apply_mask
>>> masked_img = apply_mask(to_img(img_t), to_img(mask_t), alpha=0.5)
>>> plt.imshow(masked_img)
<matplotlib.image.AxesImage object...>

The masks can also be re-generated if e.g.
:py

>>> img_t, mask_t = concept_data[0]
>>> import torchvision as tv
>>> to_img = tv.transforms.ToPILImage()
>>> from matplotlib import pyplot as plt
>>> from hybrid_learning.datasets import apply_mask
>>> masked_img = apply_mask(to_img(img_t), to_img(mask_t), alpha=0.5)
>>> plt.imshow(masked_img)
<matplotlib.image.AxesImage object...>

The masks can also be re-generated if e.g.
:py

>>> img_t, mask_t = concept_data[0]
>>> import torchvision as tv
>>> to_img = tv.transforms.ToPILImage()
>>> from matplotlib import pyplot as plt
>>> from hybrid_learning.datasets import apply_mask
>>> masked_img = apply_mask(to_img(img_t), to_img(mask_t), alpha=0.5)
>>> plt.imshow(masked_img)
<matplotlib.image.AxesImage object...>

The masks can also be re-generated if e.g.
:py:attr:`~hybrid_learning.datasets.base.BaseDataset.transforms` or
:py:attr:`~hybrid_learning.datasets.custom.coco.ConceptDataset.masks_root` are changed,
or if ``lazy_mask_generation`` was not set to ``False`` during init.

>>> concept_data.generate_masks(force_rebuild=True, leave=False)

