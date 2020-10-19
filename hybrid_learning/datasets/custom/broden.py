"""Handles for Broden-like datasets.
The original Broad and Densely Labeled Dataset (Broden_) was introduced in
[Bau2017]_ as a combination of several existing semantic segmentation and
classification datasets on overlapping image sets.
For more details on Broden and its encoding see :py:class:`BrodenHandle`.

.. note::
    The original Broden dataset is not required for usage of this code.
    Used datasets just must use a format as is used by the Broden dataset.

.. _Broden:
    https://github.com/CSAILVision/NetDissect-Lite/blob/master/script/dlbroden.sh
.. [Bau2017] Bau, David, Bolei Zhou, Aditya Khosla, Aude Oliva, and Antonio
    Torralba. 2017.
    “Network Dissection: Quantifying Interpretability of Deep Visual
    Representations.”
    In Proc. 2017 IEEE Conf. Comput. Vision and Pattern Recognition, 3319–3327.
    Honolulu, HI, USA: IEEE Computer Society.
    https://doi.org/10.1109/CVPR.2017.354.
"""

#  Copyright (c) 2020 Continental Automotive GmbH

import os
from typing import NamedTuple, Optional, Dict, Tuple, List, Sequence, Union, \
    Set, Any, Callable

import PIL.Image
import numpy as np
import pandas as pd
import torch
import torchvision as tv
from tqdm import tqdm

from .. import transforms as trafo
from ..base import BaseDataset


class BrodenLabel(NamedTuple):
    """Information needed to load the annotation of a Broden label."""
    name: str
    """The (unique) name of the label in the annotations."""
    number: int
    """The label ID."""
    category: str
    """The category from which to select samples for the label"""


class BrodenHandle(BaseDataset):
    """Handle to collect a sub-dataset of a dataset following Broden format.

    .. note::
        The original Broden dataset is not required for usage of this handle.
        Used datasets just must use a format as is used by the Broden dataset.
        In the following, the format specifics relevant for the datasets that
        can be handled are explained, using the original Broden Dataset as
        role model. *(No code from the original datasets was used.)*

    **About the Original Broden Dataset**

    The Broden dataset is the broad and densely labeled dataset initially
    prepared for the paper
    `Network Dissection <http://arxiv.org/abs/1704.05796>`_.
    It is a combination of the following datasets:

    - `ADE (scene, object, part)
      <https://groups.csail.mit.edu/vision/datasets/ADE20K/>`_
    - `Pascal-Context (object)
      <https://cs.stanford.edu/~roozbeh/pascal-context/>`_
    - `Pascal-Part (part)
      <http://roozbehm.info/pascal-parts/pascal-parts.html>`_
    - `OpenSurfaces (material)
      <http://opensurfaces.cs.cornell.edu/>`_
    - `DTD (texture)
      <https://www.robots.ox.ac.uk/~vgg/data/dtd/>`_
    - and a generated color dataset, with 11 human selected colors

    The original Broden data features both
    pixel-level semantic segmentation annotations
    (for categories see :py:attr:`SEG_CATS`), and
    image-level classification annotations
    (for categories see :py:attr:`CLS_CATS`).

    The :py:attr:`annotations` attribute stores the raw annotation information
    as :py:class:`pandas.DataFrame` as it is loaded from the index file
    (see :py:attr:`INDEX_CSV_FILE`) within the
    :py:attr:`~hybrid_learning.datasets.base.BaseDataset.dataset_root`.
    For the format of the annotations see :py:attr:`annotations` directly.

    .. note::
        To create sub-sets, one can also provide the annotations information
        on init.

    **Default Output Format**

    The :py:meth:`~hybrid_learning.datasets.base.BaseDataset.getitem` method
    yields tuples of input image and a dictionary ``{label_name: annotation}``
    containing the annotations for all specified labels.
    For the exact output format of the annotations have a look at the
    :py:meth:`getitem` doc.
    By default, for classification, the annotation is ``bool``, and for
    segmentation, it is a :py:class:`numpy.ndarray` binary mask for the
    label. If the label information is missing for the selected item,
    ``None`` is returned instead.
    This output is transformed by
    :py:attr:`~hybrid_learning.datasets.base.BaseDataset.transforms` before
    yielding it as output of
    :py:meth:`~hybrid_learning.datasets.base.BaseDataset.__getitem__`.

    .. note::
        - To collect a single custom label/merged annotations from the Broden
          dataset, refer to the :py:meth:`custom_label` builder.
        - To modify the internal annotations table after init, use
          :py:meth:`prune` or directly modify :py:attr:`annotations`.
    """
    CAT_SEP = ">>"
    """Separator string if the category is specified for a label.
    Then the format is ``"{label}{sep}{category}"``."""

    LABEL_CSV_FILE: str = "label.csv"
    """Path to the file containing meta-information about the labels, relative
    to a dataset root.
    For details on the encoding see :py:meth:`label_info_for`."""

    INDEX_CSV_FILE: str = "index.csv"
    """Path to the file containing the annotation information, relative to a
    dataset root.
    For the encoding see the documentation of this class."""

    IMAGES_ROOT: str = "images"
    """Root directory for annotated image files.
    Relative to the
    :py:attr:`~hybrid_learning.datasets.base.BaseDataset.dataset_root`.
    Annotations can be found in :py:attr:`INDEX_CSV_FILE`.
    """

    SEG_CATS = ('object', 'part', 'color', 'material')
    """Categories that provide segmentation data."""

    CLS_CATS = ('scene', 'texture')
    """Categories that provide classification data."""

    def __init__(self,
                 labels: Sequence[BrodenLabel],
                 dataset_root: str,
                 annotations: pd.DataFrame = None,
                 prune_na: bool = True, prune_na_rule: str = 'all',
                 broden_split: Optional[str] = None,
                 max_num_samples: Optional[int] = None,
                 shuffle: bool = False,
                 **dataset_args):
        """Init.

        For further arguments see the details in :py:meth:`standard_prune`.

        .. warning::
            Currently, no labels with duplicate names are allowed.
            Therefore, a label may only occur for one category.

        :param labels: list of labels to collect for each sample.
        :param dataset_root: the path to the root directory holding the
            annotation files and the images/ directory with the images and
            segmentations
        :param annotations: optional initializer for :py:attr:`annotations`,
            which is by default loaded from :py:const:`INDEX_CSV_FILE`;
            use to create sub-sets
        :param dataset_args: arguments to
            :py:class:`~hybrid_learning.datasets.base.BaseDataset`.
        """
        if annotations is not None and len(annotations) <= 0:
            raise ValueError("Empty annotations!")
        if len(labels) == 0:
            raise ValueError("Empty labels!")

        self._default_transforms = self.datum_to_tens
        """The default transformation will return tensors."""

        super(BrodenHandle, self).__init__(dataset_root=dataset_root,
                                           **dataset_args)

        self.annotations: pd.DataFrame = annotations \
            if annotations is not None \
            else self.load_annotations_table(self.dataset_root)
        """The actual annotation (meta-)information.
        The columns used here are described below.

        .. rubric:: Preliminary Remarks

        - All file-paths are relative to
          :py:attr:`~hybrid_learning.datasets.base.BaseDataset.dataset_root`
          ``/images``.
        - Several files or class labels may be given, separated by semi-colon.
        - A mask for a category is an RGB-image encoding segmentation masks for
          all different labels of that category.
          For the encoding see :py:meth:`process_seg_mask`.
        - An annotation may have labels in different categories
          (i.e. entries in these category columns). If annotation information
          for a category is missing, this column is ``None``.

        .. rubric:: The Columns

        The following columns are used here:

        - *image*: The file-path to the original image file of this annotation
        - *split*: The dataset split for which this annotation was used
          (``train`` or ``val``)
        - category columns:

          - *color*: color mask file-path
          - *object*: object mask file-path (semantic object segmentation)
          - *part*: part mask file-path (same as object masks, only parts
            belong to a super-object)
          - *material*: material mask file-path
          - *scene*: label number of the depicted scene
          - *texture*: texture label numbers
        """
        if len(self) == 0:
            raise RuntimeError("Loaded annotations information is empty!")

        label_infos: pd.DataFrame = pd.read_csv(
            os.path.join(self.dataset_root, self.LABEL_CSV_FILE))
        self.labels: List[BrodenLabel] = \
            [self.parse_label(label_spec, label_infos)
             for label_spec in labels]
        """The labels to load the values for in each line of the Broden
        annotations."""

        # Check for duplicate labels:
        for label in self.labels:
            duplicates: List[BrodenLabel] = [lab for lab in self.labels if
                                             lab.name == label.name]
            if self.labels.count(label) > 1:
                raise ValueError(
                    "Duplicate label names for labels {}".format(duplicates))

        # Prune annotations
        self.standard_prune(max_num_samples=max_num_samples, prune_na=prune_na,
                            prune_na_rule=prune_na_rule,
                            broden_split=broden_split, shuffle=shuffle)

    def standard_prune(self, max_num_samples: Optional[int] = None,
                       prune_na: bool = True, prune_na_rule: str = 'all',
                       broden_split: Optional[str] = None,
                       shuffle: bool = False) -> 'BrodenHandle':
        """Apply the specified standard pruning operations.
        Pruning is applied to the :py:attr:`annotations` table.

        :param prune_na: whether to prune all entries (rows) from the
            :py:attr:`annotations` table in which ``'all'`` or ``'any'`` of
            the covered label categories are ``NaN`` (see also ``prune_rule``)
        :param prune_na_rule: if ``prune_na`` is ``True``, rule by which to
            select candidates for pruning:

            - ``'all'``: all categories occurring in the specified labels must
              be ``NaN``
            - ``'any'``: any must be ``NaN``
        :param broden_split: the original dataset had a fix split into
            training and validation data; choose the corresponding original
            split (see also :py:attr:`annotations`, where the split
            meta-information is stored in)
        :param max_num_samples: the maximum number of samples to select;
            if set to ``None``, no restriction is applied
        :param shuffle: whether to shuffle the dataset (before restricting to
            ``max_num_samples``)
        :return: self
        """
        # region Value checks
        if broden_split is not None and broden_split not in ('train', 'val'):
            raise ValueError(("broden_split must be one of ('train', 'val'), "
                              "but was: {}").format(broden_split))
        if prune_na and prune_na_rule not in ('all', 'any'):
            raise ValueError(("prune_na_rule must be one of ('all', 'any'), "
                              "but was {}").format(prune_na_rule))
        # endregion

        # Prune NaN values
        if prune_na:
            na_selector = \
                self.annotations[{la.category for la in self.labels}].isna()
            if prune_na_rule == 'all':
                na_selector = na_selector.all(axis=1)
            else:
                na_selector = na_selector.any(axis=1)
            self.annotations: pd.DataFrame = self.annotations.loc[~na_selector]

        # Restrict to the selected split
        if broden_split is not None:
            self.annotations = \
                self.annotations.loc[self.annotations['split'] == broden_split]

        # Restrict to the selected number of samples (and shuffle)
        if max_num_samples is None or max_num_samples <= 0 or \
                max_num_samples > len(self.annotations):
            max_num_samples = len(self.annotations)
        if shuffle:
            self.annotations = self.annotations.sample(n=max_num_samples
                                                       ).reset_index(drop=True)
        self.annotations = self.annotations.iloc[:max_num_samples]

        # Final sanity check
        if len(self) == 0:
            raise RuntimeError("Annotations information is now empty after "
                               "standard pruning!")

        return self

    @classmethod
    def load_annotations_table(cls, dataset_root: str,
                               index_file: str = None) -> pd.DataFrame:
        """Load the annotation information from the ``index_file``
        under ``dataset_root``.
        For simplicity of parsing, all category and the ``"image"`` column
        are parsed to string.

        :param dataset_root: the root directory under which to find the
            index file
        :param index_file: the file name / relative path under ``dataset_root``
            of the index CSV file to load the annotations from;
            defaults to :py:attr:`INDEX_CSV_FILE`
        :return: annotations table with correct types of the category columns
        """
        index_file = index_file or cls.INDEX_CSV_FILE
        return pd.read_csv(os.path.join(dataset_root, index_file),
                           dtype={col: str for col in
                                  [*cls.CLS_CATS, *cls.SEG_CATS, "image"]})

    def parse_label(self, label_spec: Union[str, BrodenLabel],
                    label_infos: pd.DataFrame,
                    ) -> BrodenLabel:
        """Given a label specifier, parse it to a :py:class:`BrodenLabel` given
        ``label_infos``.

        :param label_spec: the label specifier to turn into a
            :py:class:`BrodenLabel`
        :param label_infos: the meta-information about all Broden labels;
            contains the information about available labels
        :return: the :py:class:`BrodenLabel` instance with information of
            the ``label_spec``
        """

        # Already in correct format:
        if isinstance(label_spec, BrodenLabel):
            return label_spec

        category: Optional[str] = None
        # region collect category information from label_spec if available
        if self.CAT_SEP not in label_spec:
            label_name: str = label_spec
        elif label_spec.split(self.CAT_SEP) == 2:
            label_name, category = label_spec.split(self.CAT_SEP)
        else:
            raise ValueError(
                ("Wrong label format of label specifier {}: expected exactly 1 "
                 "occurrence of {}").format(label_spec, self.CAT_SEP))
        # endregion

        # select category
        label_info: pd.Series = self.label_info_for(label_name, label_infos)
        categories: Dict[str, int] = self._to_cat_info(label_info['category'])
        category: str = category or list(categories.keys())[0]

        # region validate category
        if category not in categories:
            raise ValueError(("Category {} not available for labels {}; "
                              "choose one of {}"
                              ).format(category, self.labels, categories))
        if category not in [*self.SEG_CATS, *self.CLS_CATS]:
            raise ValueError("Label {} has invalid category {}; allowed: {}"
                             .format(label_spec, category,
                                     [*self.SEG_CATS, *self.CLS_CATS]))
        if category not in self.annotations.columns:
            raise ValueError(("Category {} of label {} not available in "
                              "annotations; found cols: {}"
                              ).format(category, label_spec,
                                       self.annotations.columns))
        # endregion

        return BrodenLabel(name=label_name, number=label_info.number,
                           category=category)

    @staticmethod
    def label_info_for(label_name: str, label_infos: pd.DataFrame) -> pd.Series:
        """Obtain information for label given by name from label information.
        A label may have samples in different categories.

        The output features the following information (compare Broden README):

        :number: the label ID (used for annotation in the segmentation masks)
        :name: the trivial unique name
        :category:
            the categories the labels have samples in, specified as
            semi-colon separated list of entries in
            ``{'color', 'object', 'material', 'part', 'scene', 'texture'}``,
            each entry followed by the total amount of samples for the label
            for that category;
            use :py:meth:`_to_cat_info` to process those
        :frequency: total number of images having that label over all categories
        :coverage: the mean(?) pixels per image
        :syns: synonyms

        :param label_name: the name of the label
        :param label_infos: the meta-information on all Broden labels as can
            by default be loaded from :py:const:`LABEL_CSV_FILE`.
        :returns: :py:class:`pandas.Series` with above fields filled
        :raises: :py:exc:`ValueError` if the label is not unique or cannot
            be found
        """
        label_info = label_infos[label_infos['name'] == label_name]
        if len(label_info) < 1:
            raise ValueError("Label {} not found".format(label_name))
        if len(label_info) > 1:
            raise ValueError("Label {} ambiguous: {} occurrences"
                             .format(label_name, len(label_info)))
        label_info = label_info.iloc[0]
        return label_info

    @staticmethod
    def _to_cat_info(cat_info_str: str):
        """Transform category info str of cat1(freq1);cat2(freq2);... to a dict.

        :meta public:
        """
        cats_freq: List[Tuple[str, ...]] = [tuple(cf.split('(')) for cf in
                                            cat_info_str.split(';')]
        for cat_freq in (cf for cf in cats_freq if not len(cf) == 2):
            raise ValueError(("Unknown format for category: {} (full category"
                              "info: {})").format('('.join(cat_freq),
                                                  cat_info_str))
        return {c: f.rstrip(')') for c, f in cats_freq}

    def __len__(self):
        return len(self.annotations)

    def getitem(self, i: int) -> Tuple[PIL.Image.Image,
                                       Dict[str, Union[bool, np.ndarray]]]:
        """Provide tuple of input image and dictionary with annotations for
        all labels. (See :py:attr:`labels`).
        Used for
        :py:meth:`~hybrid_learning.datasets.base.BaseDataset.__getitem__`.

        The output format is a tuple of
        ``(input_image, {label_name: annotation})``.
        The return type is as follows:
        The input image is an RGB image as :py:class:`~PIL.Image.Image`;
        For the annotations dictionary holds:

        - Each label from :py:attr:`labels` is considered, and the annotation
          for a label is

          - for classification: a ``bool`` value
          - for segmentation: a binary mask as :py:class:`numpy.ndarray`

        - In case the label is not available, its value in the annotations dict
          is ``None``.

        is a tuple of the input :py:class:`~PIL.Image.Image` and the
        annotations dict.

        :return: tuple of input image and annotations dict
        """
        img: PIL.Image.Image = PIL.Image.open(self.image_filepath(i))
        anns: Dict[str, Union[bool, np.ndarray]] = self.load_anns(i)
        return img, anns

    def load_anns(self, i: int) -> Dict[str, Union[bool, np.ndarray]]:
        """Load all annotation information for row ``i``.
        Information is retrieved from :py:attr:`annotations`.
        For details on the output format see :py:meth:`load_ann`."""
        loaded_rgb_masks = {}
        raw_ann_row: pd.Series = self.annotations.iloc[i]
        anns: Dict[str, Union[bool, np.ndarray]] = {
            label.name: self.load_ann(label, raw_ann_row=raw_ann_row,
                                      loaded_rgb_masks=loaded_rgb_masks)
            for label in self.labels
        }
        return anns

    @staticmethod
    def datum_to_tens(img: PIL.Image.Image, anns: Dict[bool, np.ndarray]
                      ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """This transformation will convert an output tuple of image, label dict
        to a tensor. For the input format see :py:meth:`getitem`.
        Any ``None`` entries in the annotations dictionary will remain ``None``.
        """
        img_t = tv.transforms.ToTensor()(img)
        # pylint: disable=no-member
        anns_t = {k: (torch.as_tensor(a, dtype=torch.float)
                      if a is not None else None) for k, a in anns.items()}
        # pylint: enable=no-member
        return img_t, anns_t

    def image_filepath(self, i: int) -> str:
        """Get the path to the image file for row ``i``.
        Information is retrieved from :py:attr:`annotations`."""
        return os.path.join(self.dataset_root, self.IMAGES_ROOT,
                            self.annotations.iloc[i]['image'])

    def load_ann(self, label: BrodenLabel, i: Optional[int] = None,
                 raw_ann_row: pd.Series = None,
                 loaded_rgb_masks: Dict[str, List[PIL.Image.Image]] = None
                 ) -> Optional[Union[bool, np.ndarray]]:
        """Load the annotation information for ``label`` at row ``i``.
        Information is retrieved from :py:attr:`annotations`.
        If the annotation information is missing for the given label category,
        return ``None``.

        .. note::
            If ``loaded_rgb_masks`` is given, this function has the side effect
            of updating this dict with newly loaded masks!
            This is used to speed up loading of several labels from the same
            mask.

        :param label: the label to restrict the annotation to
        :param i: the index of the row in the annotations information
            :py:attr:`annotations` which holds the information for this
            single annotation of interest
        :param raw_ann_row: optionally directly hand over the row of interest
            instead of providing its index (see ``i``)
        :param loaded_rgb_masks: RGB segmentation masks loaded so far
            (for speed-up); gets updated with any newly loaded masks
        :return: One of

            - ``None`` if category information is missing,
            - the binary segmentation mask for the label in case of a
              segmentation category,
            - the boolean truth value whether the label holds for the image in
              case of a classification category
        """
        if i is None and raw_ann_row is None:
            raise ValueError("Either index i or the annotation row raw_ann_row"
                             " must be given but both were None")
        if loaded_rgb_masks is None:
            loaded_rgb_masks: Dict[str, List[PIL.Image.Image]] = {}

        if raw_ann_row is None:
            raw_ann_row: pd.Series = self.annotations.iloc[i]
        raw_ann: Union[str, float] = raw_ann_row[label.category]

        # Missing annotation: return None
        if pd.isnull(raw_ann):
            return None

        raw_anns: List[str] = raw_ann.split(';')

        # raw_anns is list of file paths:
        if label.category in self.SEG_CATS:
            # RGB masks with label information encoded in red and green channel
            if label.category not in loaded_rgb_masks:
                # Update loaded mask list with newly loaded mask
                loaded_rgb_masks[label.category] = [
                    PIL.Image.open(
                        os.path.join(self.dataset_root, self.IMAGES_ROOT, fp))
                    for fp in raw_anns]
            ann = self.process_seg_mask(label, loaded_rgb_masks[label.category])
            return ann
        if label.category in self.CLS_CATS:
            # raw_anns is list of classification label numbers
            return str(label.number) in raw_anns
        raise ValueError("Unknown category for label {}; known ones: {}"
                         .format(label, [*self.SEG_CATS, *self.CLS_CATS]))

    def process_seg_mask(self, label: BrodenLabel,
                         rgb_masks: List[PIL.Image.Image]) -> np.ndarray:
        """Collect the binary segmentation mask for ``label`` from given
        relative file paths.
        Pixels belonging to the given ``label`` are 1, others 0.

        :param label: the label to look for
            (:py:attr:`~BrodenLabel.number` needed)
        :param rgb_masks: a list of RGB masks with label information encoded in
            red and green channel; for details on encoding see
            :py:meth:`to_seg_mask`
        :return: binary segmentation mask for ``label`` merged from the
            segmentation masks at given file paths
        :raises: :py:exc:`ValueError` for invalid label category
        """
        if len(rgb_masks) == 0:
            raise ValueError("Empty relative file path list rel_fp!")

        # Convert to binary masks only for self.label:
        masks_np = [self.to_seg_mask(ext_mask, label_num=label.number)
                    for ext_mask in rgb_masks]
        # Add up masks
        return (np.sum(masks_np, axis=0) > 0) \
            if len(masks_np) > 1 else masks_np[0]

    @staticmethod
    def to_seg_mask(seg: PIL.Image.Image, label_num: int) -> np.ndarray:
        """Given a Broden RGB segmentation, reduce it to a binary mask for
        ``label_num``.

        Broden segmentations are saved as RGB images, where the the label
        number of a pixel is
        ``(256 * green + red)`` with ``red`` the red channel value of the pixel,
        and ``green`` its green  channel value.
        A label number of ``0`` means background.

        The label number is the ``'number'`` field from
        :py:attr:`label_info_for` respectively the
        :py:attr:`BrodenLabel.number` attribute.
        One can either specify a single label number as ``int``, or an iterable
        of label numbers.

        :param seg: the original RGB segmentation mask encoded as described
            above
        :param label_num: the label number to restrict the mask to
        :return: union of binary segmentation masks for given label numbers
        """
        # noinspection PyTypeChecker
        seg_np = np.array(seg)
        red, green = seg_np[..., 0], seg_np[..., 1]
        binary_seg_np = (256 * green + red) == label_num
        return binary_seg_np

    def prune(self, condition: Callable[[Tuple[Any, Any]], bool],
              by_target: bool = False,
              show_progress_bar: bool = False) -> 'BrodenHandle':
        """Prune all items that fulfill ``condition`` from this dataset.
        For this, :py:attr:`annotations` is modified accordingly.

        :param condition: callable that accepts the output of
            :py:meth:`~hybrid_learning.datasets.base.BaseDataset.__getitem__`
            and returns a ``bool`` stating whether this item is to be pruned
        :param show_progress_bar: whether to show a progress bar while
            collecting the selector for ``condition``
        :param by_target: only load the target annotations of each item
            (the
            :py:attr:`~hybrid_learning.datasets.base.BaseDataset.transforms`
            are applied with dummy input) and apply ``condition`` to the target;
            asserts that transforms yields a tuple of ``(input, target)``;
            this is useful to avoid the costly loading of input images if they
            do not contribute to the transformations or the ``condition``.
        :return: this instance (with modified :py:attr:`annotations`)
        """
        selector: np.ndarray = self._selector_for(
            condition,
            show_progress_bar=show_progress_bar,
            by_target=by_target)
        self.annotations = self.annotations[~selector]
        return self

    def balance(self, condition: Callable[[Tuple[Any, Any]], bool],
                proportion: float = 0.5,
                by_target: bool = False,
                show_progress_bar: bool = False) -> 'BrodenHandle':
        """Restrict this dataset to a subset with an exact ``proportion``
        fulfilling ``condition``.
        For this, :py:attr:`annotations` is modified accordingly.
        After splitting the dataset by ``condition``, the half which is too
        large to fulfill ``proportion`` is reduced by random sub-sampling,
        determining the final size of the dataset.

        If there is only one class in the dataset, only shuffling is applied.

        :param condition: callable that accepts the output of
            :py:meth:`~hybrid_learning.datasets.base.BaseDataset.__getitem__`
            and returns a ``bool`` stating whether this item belongs to the
            first split
        :param proportion: the aimed-for proportion of the first split on
            the final dataset
        :param show_progress_bar: whether to show a progress bar while
            collecting the selector for ``condition``
        :param by_target: only load the target annotations of each item
            (the
            :py:attr:`~hybrid_learning.datasets.base.BaseDataset.transforms`
            are applied with dummy input) and apply ``condition`` to the target;
            asserts that transforms yields a tuple of ``(input, target)``;
            this is useful to avoid the costly loading of input images if they
            do not contribute to the transformations or the ``condition``.
        :return: self
        """
        selector: np.ndarray = self._selector_for(
            condition,
            by_target=by_target,
            show_progress_bar=show_progress_bar)
        # Reduce positives
        pos: pd.DataFrame = self.annotations.loc[selector]
        if len(pos) / len(self.annotations) > proportion:
            to_reduce: pd.DataFrame = pos
            to_keep: pd.DataFrame = self.annotations.loc[~selector]
            prop_to_keep: float = 1 - proportion
        # Reduce negatives
        else:
            to_reduce: pd.DataFrame = self.annotations.loc[~selector]
            to_keep: pd.DataFrame = pos
            prop_to_keep: float = proportion

        # Is there only one class in the dataset?
        if np.allclose(prop_to_keep, 0):
            return self.shuffle()

        # Calc the final amounts of samples for each slice
        num_to_keep: int = len(to_keep)
        num_all: int = int(num_to_keep / prop_to_keep)
        num_to_reduce: int = max(1, num_all - num_to_keep)

        # Subsample, shuffle:
        self.annotations: pd.DataFrame = pd.concat(
            [to_reduce.sample(n=num_to_reduce),
             to_keep.sample(n=num_to_keep)],
            ignore_index=True)
        self.shuffle()
        return self

    def _selector_for(self, condition: Callable[[Tuple[Any, Any]], bool],
                      show_progress_bar: bool = False,
                      by_target: bool = False) -> np.ndarray:
        """Provide ``bool`` list matching indices of this dataset for which
        ``condition`` holds.
        Optionally show a progress bar while processing the data.

        :param by_target: only load target
            (transforms is applied with dummy input) and apply
            condition to target; asserts that transforms yields a tuple of
            ``(input, target)``
        """
        if by_target:
            dummy_img: PIL.Image.Image = PIL.Image.open(self.image_filepath(0))
            load_fn: Callable[[int], Any] = \
                (lambda i: self.transforms(dummy_img, self.load_anns(i))[1])
        else:
            load_fn: Callable[[int], Any] = lambda i: self[i]

        selector: List[bool] = []
        iterator = range(len(self))
        if show_progress_bar:
            iterator = tqdm(iterator,
                            desc="Iterating " + self.__class__.__name__)
        for i in iterator:
            selector.append(condition(load_fn(i)))
        return np.array(selector, dtype=bool)

    def shuffle(self) -> 'BrodenHandle':
        """Shuffle the held annotations and return self."""
        self.annotations = self.annotations.sample(frac=1
                                                   ).reset_index(drop=True)
        return self

    @classmethod
    def custom_label(cls, dataset_root: str, label: str,
                     prune_empty: Union[bool, str] = True,
                     balance_pos_to: Optional[float] = None,
                     verbose: bool = False,
                     **init_args):
        # pylint: disable=line-too-long
        """Return a :py:class:`BrodenHandle` instance with output restricted to
        single ``label``.

        The transformations in
        :py:attr:`~hybrid_learning.datasets.base.BaseDataset.transforms` will be chosen
        such that :py:meth:`~hybrid_learning.datasets.base.BaseDataset.__getitem__`
        outputs a tuple of ``(input_image, annotation)`` where

        - ``input_image`` is encoded as :py:class:`torch.Tensor`
        - ``annotation`` is a :py:class:`torch.Tensor` holding either the
          binary mask for the specified label or the bool classification value.

        The label may either be a label as would be specified in
        :py:class:`__init__ <BrodenHandle>` or a string representation of a
        :py:class:`~hybrid_learning.datasets.transforms.dict_transforms.Merge` operation.

        :param dataset_root: the ``dataset_root`` parameter for init of the
            :py:class:`BrodenHandle`
        :param label: the label to restrict to; may either be a valid string
            label name, a valid
            :py:class:`BrodenLabel`, or a valid string representation of a
            :py:class:`~hybrid_learning.datasets.transforms.dict_transforms.Merge` operation
            the
            :py:class:`~hybrid_learning.datasets.transforms.dict_transforms.Merge.all_in_keys`
            of which are all valid string label names;
        :param init_args: further init arguments to the :py:class:`BrodenHandle`
        :param balance_pos_to: if a value given, balance the resulting
            :py:class:`BrodenHandle` instance such that the proportion of the
            ``True`` entries is this value;
            only use for classification examples
        :param prune_empty: whether to prune empty entries
            (``None`` values and empty masks) using :py:meth:`prune`
        :param verbose: show progress bars
        :return: :py:class:`BrodenHandle` instance for ``dataset_root`` with
            :py:attr:`~hybrid_learning.datasets.base.BaseDataset.transforms` and
            :py:class:`~BrodenHandle.labels`
            selected such that the output of :py:meth:`getitem` is
            transformed to the format specified above
        """
        # pylint: enable=line-too-long
        # region Value checks
        if "labels" in init_args:
            raise ValueError(("init_args must not contain labels key, "
                              "but were {}").format(init_args))
        # endregion

        merge_op: Optional[trafo.Merge] = None  # Merge op before flatten

        # region Parse the label (and collect Merge operation if necessary):
        # collect: labels, merge_op, final_key (=the final key to which to
        # restrict the dict)
        if isinstance(label, BrodenLabel):
            labels: List[BrodenLabel] = [label]
            final_key: str = label.name
        elif isinstance(label, trafo.Merge):
            merge_op: trafo.Merge = label
            labels: Set[str] = merge_op.all_in_keys
            final_key: str = merge_op.out_key
        elif isinstance(label, str):
            # Can be parsed to merge operation?
            parsed_label: Union[str, trafo.Merge] = trafo.Merge.parse(label)
            if isinstance(parsed_label, str):
                labels: List[str] = [label]
                final_key: str = label
            else:
                merge_op: trafo.Merge = parsed_label
                labels: Set[str] = merge_op.all_in_keys
                final_key: str = merge_op.out_key
        else:
            raise ValueError("label {} has unknown format {}"
                             .format(label, type(label)))

        assert final_key != ""
        assert len(labels) > 0
        # endregion

        # region Collect the transformation
        trafos: List[trafo.TupleTransforms] = []
        trafos += [trafo.OnTarget(merge_op),
                   trafo.OnTarget(trafo.RestrictDict([final_key]))] \
            if merge_op else []
        trafos += [cls.datum_to_tens,
                   trafo.OnTarget(trafo.FlattenDict(final_key))]
        user_defined_trafo = init_args.pop('transforms', None)
        # endregion

        broden_inst = cls(dataset_root=dataset_root, labels=labels, **init_args)
        # specify separately for IDE type inference:
        broden_inst.transforms = trafo.Compose(trafos)

        if prune_empty:
            broden_inst.prune(
                lambda a: a is None or (a.dim() > 0 and a.sum() == 0),
                by_target=True, show_progress_bar=verbose)
        if balance_pos_to is not None:
            broden_inst.balance(lambda a: a, proportion=balance_pos_to,
                                by_target=True, show_progress_bar=verbose)

        # Append the user-defined transforms
        # (after pruning, since this requires control over the output format!)
        if user_defined_trafo is not None:
            broden_inst.transforms.append(user_defined_trafo)

        return broden_inst
