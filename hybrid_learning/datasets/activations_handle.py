"""Wrapper dealing with generating / saving activation maps from a dataset."""

#  Copyright (c) 2020 Continental Automotive GmbH

import os
from typing import Callable, Tuple, Optional, Any

import torch
import torchvision as tv
from torch.utils.data import Subset
from tqdm import tqdm

from .base import BaseDataset


class ActivationDatasetWrapper(BaseDataset):
    # noinspection PyUnresolvedReferences
    # pylint: disable=line-too-long
    """Wrapper for image datasets that will generate and yield activation maps.
    Behaves like a sequence of tuples of:

    :Activation map:
        Activation map tensor of the given model and the given layer for
        image at the corresponding index in the wrapped dataset

    :Ground Truth:
        Ground truth at the corresponding index in the wrapped dataset

    If the activation map generator is given during init with
    ``lazy_generation`` set to ``True``, the data is lazily generated,
    i.e. any call to
    :py:meth:`~hybrid_learning.datasets.base.BaseDataset.__getitem__`
    (resp. :py:meth:`getitem`) will generate the corresponding activation
    map if it does not yet exist.
    To generate all activation maps, call :py:meth:`generate_act_maps`,
    but be aware that this can be very time consuming depending on
    the generator.
    """
    # pylint: disable=line-too-long

    _HASH_LEN = 8
    """Length of hash for comparing model :py:meth:`torch.nn.Module.state_dict`
    (includes leading ``0x``)"""

    _ACT_MAPS_ROOT_ROOT = "activations"
    """Usual parent directory of all activation folders;
    by default assumed to be sibling to ``images`` and ``masks`` folder,
    see :py:attr:`_default_activations_root`."""

    @property
    def _default_activations_root(self) -> str:
        """Default root folder for storing activation map files.

        :meta public:
        """
        acts_basename = "{img_base}_{model_hash}-{module_id}".format(
            img_base=os.path.basename(self.dataset_root),
            model_hash=self.model_description,
            module_id=self.layer_key
        )
        acts_dirname = os.path.join(
            os.path.dirname(os.path.dirname(self.dataset_root)),
            self._ACT_MAPS_ROOT_ROOT
        )
        return os.path.join(acts_dirname, acts_basename)

    def __init__(self,
                 dataset: BaseDataset,
                 layer_key: str = None,
                 model_description: str = None,
                 act_map_gen: torch.nn.Module = None,
                 act_map_filepath_fn: Callable[[int, BaseDataset], str] = None,
                 force_rebuild: bool = False,
                 activations_root: str = None,
                 lazy_generation: bool = True,
                 **data_args
                 ):
        # pylint: disable=line-too-long
        """Init.

        :param dataset: Dataset to wrap; must be a sequence of tuples of

            ``(image as :py:class:`torch.Tensor`,
            ground truth as :py:class:`torch.Tensor`)``

            the default transformation assumes that the ground truth are masks
            (same sized images)
        :param layer_key: optional description of the layer to obtain
            activation maps from;
        :param act_map_gen: function yielding from an image the activation
            map to save; intended to be a
            :py:class:`~hybrid_learning.concepts.models.model_extension.ModelStump`
        :param model_description: description of the model parameters from which
            activation maps are generated;
            used to provide default for :py:attr:`act_maps_root`;
            defaults to class name of ``act_map_gen``
        :param activations_root: root directory under which to store and find
            the activation maps;
            defaults: :py:attr:`_default_activations_root`
        :param transforms: see ``__init__()`` of
            :py:class:`~hybrid_learning.datasets.base.BaseDataset`
        :param lazy_generation: if the ``act_map_gen`` is given, whether to
            lazily generate missing activation or to initialize them all
            during init
        :param dataset_root: dataset root directory;
            defaults to ``dataset.dataset_root``
        :param split: dataset split identifier; defaults to
            :py:attr:`~hybrid_learning.datasets.base.BaseDataset.split` of
            ``dataset``
        """
        # pylint: enable=line-too-long
        # Default values
        data_args['split'] = \
            data_args.get('split',
                          # dataset.split
                          getattr(dataset, "split", None) or
                          # dataset.dataset.split
                          getattr(getattr(dataset, "dataset", None),
                                  "split", None))
        data_args['dataset_root'] = \
            data_args.get('dataset_root',
                          # dataset.dataset_root
                          getattr(dataset, "dataset_root", None) or
                          # dataset.dataset.dataset_root
                          getattr(getattr(dataset, "dataset", None),
                                  "dataset_root", None))
        # Value checks
        if data_args['dataset_root'] is None:
            raise ValueError("dataset_root is None: "
                             "not given and dataset specifies none")

        super(ActivationDatasetWrapper, self).__init__(**data_args)

        self.dataset: BaseDataset = dataset
        """The dataset to wrap; activation maps are created from input;
        the ground truth is equal to the original one."""

        if layer_key is None and hasattr(act_map_gen, 'stump_head'):
            layer_key = act_map_gen.stump_head
        self.layer_key: Optional[str] = layer_key or 'unknown_layer'
        """Optional description of the layer from which the activation maps
        were retrieved."""

        # Determine the model description
        if act_map_gen is None and model_description is None:
            raise ValueError("Either model or model_description must be given.")
        self.model_description: str = model_description \
            if model_description is not None else act_map_gen.__class__.__name__
        """Description of the model with which activation maps were generated.
        Used to create default activation map root name
        (by creating form of hash)."""

        self.act_maps_root: str = \
            activations_root or self._default_activations_root
        """Root directory under which to store and locate activation maps."""
        os.makedirs(self.act_maps_root, exist_ok=True)

        if act_map_filepath_fn is None and not hasattr(dataset,
                                                       "image_filepath"):
            if not hasattr(dataset, "dataset"):
                raise ValueError(
                    ("Either act_map_filepath_fn must be given or dataset "
                     "(type {}) must provide function image_filepath "
                     "or have a member dataset.dataset that does so"
                     ).format(type(dataset)))
            if hasattr(dataset, "dataset") and not hasattr(dataset.dataset,
                                                           "image_filepath"):
                raise ValueError(
                    ("Either act_map_filepath_fn must be given or dataset "
                     "(type {}) or its member dataset.dataset (type {}) must "
                     "provide a function image_filepath"
                     ).format(type(dataset), type(dataset.dataset)))
        self.act_map_filepath_fn: Callable[[int, BaseDataset], str] = \
            act_map_filepath_fn or self._default_act_map_filepath_fn
        """The function to determine the activation map filepath relative to
        the given activations root at index ``i``.
        The activations root :py:attr:`act_maps_root` is used.
        It must accept the index and the original dataset.

        .. note::
            It should be stable against reindexing of the original dataset
            like shuffling or subsetting!

        The default requires an ``image_filepath`` function in ``dataset`` or
        ``dataset.dataset`` that yields unique image file names. See
        :py:meth:`_default_act_map_filepath_fn`.
        """

        # Now ensure existence of all activation maps if not lazy_generation
        self.act_map_gen: torch.nn.Module = act_map_gen
        """Callable that yields an activation map given a valid input datum.
        Input data is assumed to origin from the original :py:attr:`dataset`.
        Used to generate missing activation maps in :py:meth:`getitem`.
        If ``lazy_generation=False``, all missing activation maps are
        generated on ``__init__`` call. If additionally
        ``force_rebuild=True``, also existing ones are replaced."""
        if self.act_map_gen is not None and not lazy_generation:
            # Generate the activation maps for the specified model and layer:
            self.generate_act_maps(force_rebuild)
        if self.act_map_gen is None:
            # Sanity check: Do all activation maps exist?
            for i in (i for i in range(len(self))
                      if not self.act_map_exists(i)):
                raise FileNotFoundError(
                    "Act map at index {} missing from root directory {} "
                    "(assumed path {})".format(i, self.act_maps_root,
                                               self.act_map_filepath(i)))

    def getitem(self, i: int) -> Tuple[torch.Tensor, Any]:
        """Get activation map and original ground truth for item at index ``i``.

        Used for
        :py:meth:`~hybrid_learning.datasets.base.BaseDataset.__getitem__`.
        If the activation map does not exist and a generator is given in
        :py:attr:`act_map_gen`, generate and save the activation map.

        :return: tuple of the loaded or generated activation map and the
            target of the original dataset for that act map
        """
        # Get activation map (generate and save lazily if possible)
        if self.act_map_gen is not None and not self.act_map_exists(i):
            img_t: torch.Tensor = self.load_image(i)
            self.save_act_map(i, act_map=self.generate_act_map(self.act_map_gen,
                                                               img_t))
        act_map: torch.Tensor = self.load_act_map(i)

        # Get mask
        _, mask = self.dataset[i]
        return act_map, mask

    def __len__(self) -> int:
        """Length determined by the length of the wrapped dataset.
        See :py:attr:`dataset`."""
        return len(self.dataset)

    @staticmethod
    def _resize_to_fit(fit_t: torch.Tensor, img_t: torch.Tensor):
        """Resize image to the size of ``fit_t``.
        ``fit_t`` must be :py:class:`torch.Tensor` with
        ``len(fit_t.size())>=2``.
        This method can be used e.g. for definition of
        :py:attr:`transformations`.

        :param fit_t: :py:class:`torch.Tensor` to extract target size from
            (last two dimensions)
        :param img_t: :py:class:`torch.Tensor` representation of the image to
            resize
        """
        # Resize to activation map size
        act_map_img_size = (fit_t.size()[-2], fit_t.size()[-1])
        resizer: Callable[[torch.Tensor], torch.Tensor] = tv.transforms.Compose(
            [
                tv.transforms.ToPILImage(),
                tv.transforms.Resize(size=act_map_img_size),
                tv.transforms.ToTensor()
            ])
        return resizer(img_t)

    # def _default_transforms(self, inp: torch.Tensor,
    #                         ground_truth: torch.Tensor
    #                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """Default transformation resizing activation maps to input."""
    #     return inp, self._resize_to_fit(fit_t=inp, img_t=ground_truth)

    def act_map_filepath(self, i: int) -> str:
        """Return the path to the activation map file.
        The base directory is :py:attr:`act_maps_root`.
        The basename is:

        - *If :py:attr:`dataset` or its member ``dataset`` features a
          ``image_filepath()`` method:*
          the same basename as the original input image, with .png resp. .jpg
          replaced by .pkl
        - *Else:*
          the zero-padded decimal index number ending with .pkl

        .. warning::
            Make sure the used :py:attr:`act_map_filepath_fn` yields unique IDs
            for all elements of the :py:attr:`dataset`. Otherwise, the wrong
            activation maps may be loaded and duplicates overwritten.

        :param i: index of the image to get activation map for.
        :return: (relative or absolute) path to the activation map for datum
            at index ``i``
        """
        act_fname = self.act_map_filepath_fn(i, self.dataset)
        return os.path.join(self.act_maps_root, act_fname)

    @staticmethod
    def _default_act_map_filepath_fn(i: int, dataset: BaseDataset):
        """Obtain the act map filepath relative to the
        activations root.
        As activations root, :py:attr:`act_maps_root` is used.
        It is assumed that

        - the images are saved as ``unique_id.{png,jpg}``, that
        - either ``dataset`` or ``dataset.dataset`` (e.g. in case of a
          :py:class:`torch.utils.data.Subset` instance) provide a function
          ``image_filepath(i: int)``, and that
        - this ``image_filepath`` function yields
          ``some/path/to/img/unique_id.{png,jpg}`` for the image at position
          ``i``, with ``unique_id`` being stable against shuffling of the
          indices ``i``.

        Then the activation map filepath is given as ``unique_id.pkl``
        (path relative to an activation map root).

        :meta public:
        """
        # Determine the correct image filepath
        # Normal case
        if hasattr(dataset, "image_filepath") \
                and callable(dataset.image_filepath):
            img_fp = dataset.image_filepath(i)
        # Dataset wrapped in Subset -> take care of index permutation
        elif isinstance(dataset, Subset) \
                and hasattr(dataset.dataset, "image_filepath") \
                and callable(dataset.dataset.image_filepath):
            img_fp = dataset.dataset.image_filepath(dataset.indices[i])
        else:
            raise AttributeError(
                ("Could not determine image_filepath function within dataset "
                 "or its dataset member; dataset type: {}"
                 ).format(type(dataset)))
        img_fname: str = os.path.basename(img_fp)
        act_fname: str = (img_fname.replace('.jpg', '.pkl')
                          .replace('.png', '.pkl'))

        # # The following is not stable against shuffled or subsetted datasets!
        # else:
        #     # Length of the maximum index
        #     num_letters = len(str(len(dataset)))
        #     act_fname: str = ("{!s:0>" + str(num_letters) + "}.pkl").format(i)
        return act_fname

    def act_map_exists(self, i: int) -> bool:
        """Check whether the activation map at index ``i`` was already created.

        :param i: index in :py:attr:`dataset` for which to check
            whether an activation map was created.
        """
        act_fp: str = self.act_map_filepath(i)
        return os.path.exists(act_fp) and os.path.isfile(act_fp)

    def load_act_map(self, i: int) -> torch.Tensor:
        """Load act map as :py:class:`torch.Tensor` for given index ``i``.

        :param i: index to load activation map for
        :return: :py:class:`torch.Tensor` representation of activation map
        """
        return torch.load(self.act_map_filepath(i), map_location=torch.device("cpu"))

    def load_image(self, i: int) -> torch.Tensor:
        """Load the image/original input for index ``i``."""
        return self.dataset[i][0]

    def save_act_map(self, i: int, act_map: torch.Tensor) -> None:
        """Save an activation map at index ``i`` via pytorch pickling.

        :param i: index of the image to which the activation map corresponds
        :param act_map: activation map to save (:py:class:`torch.Tensor`)
        """
        act_fp = self.act_map_filepath(i)
        if os.path.exists(act_fp):
            raise FileExistsError(
                ("Tried saving activation map for index {} to file {},"
                 " but file already exists").format(i, act_fp))
        os.makedirs(os.path.dirname(act_fp), exist_ok=True)
        torch.save(act_map, act_fp)

    @staticmethod
    def generate_act_map(act_map_gen: torch.nn.Module,
                         img_t: torch.Tensor) -> torch.Tensor:
        """Generate activation map :py:class:`torch.Tensor` of image.
        To generate it, use ``act_map_gen`` callable.

        :param img_t: image for which to obtain activation map;
            make sure all necessary transformations are applied
        :param act_map_gen: model generating the activation map from the image
        :return: activation map of layer as :py:class:`torch.Tensor`
        """
        # Run wrapper to obtain intermediate outputs
        with torch.no_grad():
            # move input to correct device
            if len(list(act_map_gen.parameters())) > 0:
                device: torch.device = next(act_map_gen.parameters()).device
                img_t = img_t.to(device)

            act_map = act_map_gen.eval()(img_t.unsqueeze(0))
            # Squeeze batch dimension
            act_map = act_map.squeeze(0)
        return act_map

    def generate_act_maps(self, force_rebuild: bool = False,
                          show_progress_bar: bool = True,
                          **kwargs) -> None:
        """Generate activation maps for all images.

        :param force_rebuild: whether to overwrite existing images or not
        :param show_progress_bar: whether to show the progress using
            :py:class:`tqdm.tqdm`
        :param kwargs: further arguments to the progress bar
        """
        act_maps_to_process = [i for i in range(len(self))
                               if force_rebuild or not self.act_map_exists(i)]
        if len(act_maps_to_process) == 0:
            return

        if show_progress_bar:
            act_maps_to_process = tqdm(
                **{**dict(iterable=act_maps_to_process, unit="act_map",
                          desc="Activation maps newly generated: "),
                   **kwargs})
        for i in act_maps_to_process:
            # Get activation map
            img_t: torch.Tensor = self.load_image(i)
            act_map: torch.Tensor = self.generate_act_map(self.act_map_gen,
                                                          img_t)

            # Save
            if self.act_map_exists(i):
                os.remove(self.act_map_filepath(i))
            self.save_act_map(i, act_map=act_map)
