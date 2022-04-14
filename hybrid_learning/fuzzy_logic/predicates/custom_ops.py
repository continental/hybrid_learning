"""Custom fuzzy logic operations."""
#  Copyright (c) 2022 Continental Automotive GmbH
import abc
import logging
from typing import Optional, Tuple, Union, Callable, List, Dict, Any, Sequence

import numpy as np
import torch


from ..logic_base import Merge, TorchOperation
from ..tnorm_connectives import product
from ...datasets.transforms.encoder import same_padding
from ...datasets.base import add_gaussian_peak


class LambdaOp(Merge):
    """Provide a custom transformation on the sequence of values extracted from the annotations dict."""

    def __init__(self, *in_keys: Union[str, 'Merge'], trafo: Callable[[Sequence[Any]], Any],
                 symb: str = 'CustomOp', arity: int = -1, **kwargs):
        super().__init__(*in_keys, symb=symb, **kwargs)
        self.trafo: Callable[[Sequence[Any]], Any] = trafo
        """The operation to apply to the annotation values selected by ``in_keys``.
        The output is added under :py:attr:`out_key`."""
        self.ARITY: int = arity

    @property
    def settings(self) -> Dict[str, Any]:
        return {**super().settings, 'trafo': self.trafo, 'arity': self.ARITY}
    
    @property
    def setting_defaults(self) -> Dict[str, Any]:
        return {**super().setting_defaults, 'arity': -1}
    
    def operation(self, annotation_vals: Sequence) -> Any:
        """Apply ``trafo`` to annotation values."""
        return self.trafo(annotation_vals)


class IsPartOfA(TorchOperation):
    r"""Unary predicate that calculates for each pixel location whether
    it is part of an object segmentation.
    The predicate accepts as input the segmentation mask information for objects
    (encoding the predicate "pixel ``p`` is a ``A``" = ``IsA(p)``).
    It returns a mask that for each pixel holds the truth value, whether
    that pixel is part of an object, i.e. at each pixel position ``p`` the output is

    .. math::
        \exists p_o: \text{IsA}(p_o) \wedge \text{IsPartOf}(p, p_o) \\
        = \max_{p_o} \text{IsA}(p_o) \wedge \text{IsPartOf}(p, p_o)

    It accepts 1D or 2D object masks of the shape ``[[batch x [channels x]] height x] width``,
    in any format that can be parsed to a :py:class:`torch.Tensor`.

    **Fuzzification**

    As fuzzification of the exists quantifier, ``max`` is used.
    The :py:attr:`logical_and` defines the fuzzification of ``AND``.
    The values of the predicate ``IsA(.)`` are given by the pixel values of the input
    segmentation mask.
    The fuzzy ``IsPartOf`` relation is chosen to calculate as a Gaussian distance,
    with a threshold of value :py:attr:`thresh` and distance
    ``thresh_radius=int(kernel_size - 1) / 2`` for better computational performance:

    .. math::
        d(p, p') = \exp(- \frac{\|p - p'\|_2^2}{2 \sigma^2}) \\
        \text{IsPartOf}(p, p') := \begin{cases}
        d(p, p') \text{if} \|p, p'\|_1 <= \text{thresh\_radius} \\
        0 \text{else}
        \end{cases}

    with :math:`\sigma` chosen such that :math:`d(p, p') = \text{thresh}` for
    :math:`\|p - p'\|_2 = \text{thresh\_radius}`.
    The per-pixel values of the ``IsPartOf`` predicate are stored in the
    :py:attr:`_ispartof_values` mask which is then convoluted with the ``IsA`` values.

    .. note::
        An even :py:attr:`kernel_size` will have the same :py:attr:`_ispartof_values`
        pixel values as the odd ``kernel_size-1``, only with a row and a column added
        (on top/left for :py:attr:`conv_hang_front` ``True``, else on bottom/right).

    **Choosing the right ``kernel_size``**

    The shape of the used Gaussian can be defined by a pair of x-y-values ``(r, t)``.
    The Gaussian then looks like

    .. math:: G_{r,t}(x) = \exp\left(- \frac{ x^2 }{ \frac{r^2}{- \ln(t)}}} \right)

    As ``thresh_radius``, ``int(kernel_size - 1) / 2`` is used, which also serves as
    the cutoff distance. To choose the ``kernel_size`` such that
    (1) the point ``(r, t)`` lies on the Gaussian, and
    (2) the ``kernel_size`` cuts at threshold ``thresh``,
    solve :math:`G_{r,t}(\frac{\text{k}-1}{2}) = \text{thresh}` for :math:`k` and gets
    ``kernel_size=int(k)``.
    E.g. to match the constraints that at 4 pixel shift (``r=4``), :math:`G(r)` should still be 0.8,
    and the kernel should only cut off at a threshold of 0.1, the ``kernel_size`` must be
    ``2*int(12.4) + 1=25``.

    **Visualization example**

    Visualization of the different logics:

    >>> import hybrid_learning.fuzzy_logic as fl
    >>> import torch, torchvision.transforms.functional as F, matplotlib.pyplot as plt
    >>> # Builder for IsPartOf
    >>> P = fl.predicates.custom_ops.IsPartOfA.with_(kernel_size=10, thresh=0.01, conv_hang_front=False)
    >>> # The sample input mask
    >>> t = torch.zeros([1, 1, 11, 11])
    >>> t[:, :, 4, 4], t[:, :, 5, 8] = 0.75, 0.5
    >>> # The tensors to compare
    >>> figs = dict(segmask=t)  # input
    >>> for logic in (fl.LukasiewiczLogic(), fl.ProductLogic(), fl.GoedelLogic()):
    ...    title = f'IsPartOfA ({logic.__class__.__name__.replace("Logic", "")})'
    ...    p = P("pedestrian", logical_and=logic.logical_('AND'))
    ...    figs[title] = p({"pedestrian": t})[p.out_key]
    >>> # Plot
    >>> fig, axes = plt.subplots(1, len(figs), figsize=(len(figs)*3, 3), squeeze=False)
    >>> for i, (title, tens) in enumerate(figs.items()):
    ...     shown_img = axes[0, i].imshow(F.to_pil_image(tens.view(t.size()[2:])))
    ...     _ = axes[0, i].set_title(title)
    >>> plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    >>> cax = plt.axes([.85, .25, 0.01, 0.5])
    >>> _ = plt.colorbar(shown_img, cax=cax, orientation='vertical')
    >>> plt.show()
    """
    SYMB: str = "IsPartOfA"
    ARITY: int = 1

    @property
    def kernel_size(self) -> int:
        """The size of the kernel to use.
        The standard deviation of the gaussian is determined using
        this as radius and :py:attr:`thresh`."""
        return self._kernel_size[0]

    @property
    def thresh(self) -> float:
        """The value the Gaussian should have at ``1/2 kernel_size``."""
        return self._thresh

    @property
    def conv_hang_front(self) -> bool:
        """Whether the cutoff modelled by the ``IsPartOfA`` convolution should
        hang in the front dimensions in case of unequal padding.
        Internals: To achieve hanging towards the back, the padding must be chosen to hang front."""
        padding: np.ndarray = np.array(self._zero_pad.padding).reshape(-1, 2)
        rear_minus_front: np.ndarray = padding[:, 1] - padding[:, 0]
        return (rear_minus_front >= 0).all()

    @property
    def _kernel_size(self) -> Tuple[int, int]:
        """Quadratic kernel size tuple for internal usage."""
        kernel_size: Tuple[int, int] = tuple(self._ispartof_values.size()[2:4])
        assert kernel_size[0] == kernel_size[1], "Non-quadratic kernel size {} found!".format(kernel_size)
        return kernel_size

    def __init__(self, in_key: Union[str, Merge], *,
                 kernel_size: int = 10, thresh: float = 0.1,
                 logical_and: Callable[[List[torch.Tensor]], torch.Tensor] = None,
                 conv_hang_front: bool = True, **other_setts):
        """Init.

        The parameters ``kernel_size`` and ``thresh`` are specifiers
        for the distance function encoding the is-part-of relation.
        The ``logical_and`` is the logical ``AND`` callable that accepts a list
        of two tensors with values in [0, 1] and returns their pixel-wise logical ``AND``.
        The ``in_key`` is the key of the fuzzy mask which encodes the per-pixel
        membership degree to the class of interest.
        E.g. if "is part of a pedestrian" should be calculated, ``"pedestrian"``
        is a good choice.
        """
        super().__init__(in_key, **other_setts)

        # AND
        self.logical_and: Callable[[List[torch.Tensor]], torch.Tensor] = logical_and
        """Callable accepting a list of two tensors with values in [0, 1]
        and returning their pixel-wise logical ``AND``.
        Mandatory argument."""
        if self.logical_and is None:
            raise TypeError("Please specify a logical_and.")

        # Specification / calculation of the IsPartOf relation
        self._thresh: float = thresh
        """Internal storage for :py:attr:`thresh`."""

        # Operations
        if kernel_size % 2 == 1 and not conv_hang_front:
            logging.getLogger().warning("IsPartOfA initialized with odd kernel_size and conv_hang_front==False: "
                                        "For odd kernel sizes conv_hang_front is always True.")
        self._ispartof_values: torch.Tensor = self.create_ispartof_kernel(kernel_size, self.thresh,
                                                                          conv_hang_front=conv_hang_front)
        r"""The kernel storing the non-zero values of the distance function.
        The pixel at coordinate :math:`p` has the value :math:`d(p, \text{center\_of\_kernel})`
        for the distance function :math:`d` used to encode the ``IsPartOf`` predicate.
        Shape: ``(batch, channels, kernel_h, kernel_w, height, width) = (1, 1, kernel_h, kernel_w, 1, 1)``.
        The non-zero ``IsPartOf`` relation values at each pixel location are stored in this kernel. 
        """
        self._unfold: torch.nn.Unfold = torch.nn.Unfold(kernel_size=self.kernel_size)
        self._zero_pad: torch.nn.ZeroPad2d = torch.nn.ZeroPad2d(
            padding=same_padding(self._kernel_size, hang_front=not conv_hang_front))

    @staticmethod
    def create_ispartof_kernel(kernel_size: int, thresh: float,
                               thresh_radius: float = None,
                               conv_hang_front: bool = True) -> torch.Tensor:
        """Create the kernel tensor defining the ``IsPartOf`` relation.
        For details see :py:attr:`_ispartof_values`."""
        thresh_radius: float = thresh_radius if thresh_radius is not None else int((kernel_size - 1) / 2)
        kernel_np: np.ndarray = np.zeros([kernel_size, kernel_size])
        pad_top, _, pad_left, _ = same_padding((kernel_size, kernel_size), hang_front=not conv_hang_front)
        kernel_np = add_gaussian_peak(
            kernel_np,
            centroid=(pad_top + 0.5, pad_left + 0.5),
            binary_radius=max(.5, thresh_radius), radius_value=thresh
        ).reshape([1, 1, *kernel_np.shape])
        kernel: torch.Tensor = torch.tensor(kernel_np, dtype=torch.float
                                            ).view([1, 1, kernel_size, kernel_size, 1, 1])
        return kernel

    @property
    def settings(self) -> Dict[str, Any]:
        return dict(**dict(kernel_size=self.kernel_size,
                           thresh=self.thresh,
                           conv_hang_front=self.conv_hang_front,
                           logical_and=self.logical_and),
                    **super().settings)

    @property
    def setting_defaults(self):
        """Defaults used for settings."""
        return dict(kernel_size=10, thresh=0.1, logical_and=None, conv_hang_front=True,
                    **super().setting_defaults)

    def torch_operation(self, mask: torch.Tensor) -> torch.Tensor:
        """Calculate value of IsPartOfA predicate for given object segmentation mask.
        Allowed mask shapes (1D-4D): ``([[batch, [1,]] height,] width)``
        The output shape is that of the input mask.
        Invalid shapes raise a :py:class:`ValueError`.

        The device of ``mask`` is used for calculations and will be the device
        of the output tensor.
        As a side effect, the internal storage of the ``IsPartOf`` values is
        moved to that device, assuming that the device of inputs does not change frequently."""
        # region Correct size: ensure a view where image is batch x 1 x h x w
        orig_size: List[int] = list(mask.size())
        if 1 <= len(orig_size) <= 4:
            mask = mask.view([-1, 1, *([1] if len(orig_size) == 1 else []), *orig_size[-2:]])
        else:
            raise ValueError("Unexpected mask size {}. Expected mask of size [batch [x channel]] x h x w"
                             .format(mask.size()))
        # endregion

        # pad & unfold the "is a ..." mask
        isa_mask: torch.Tensor = self._zero_pad(mask)
        # unfolded shape: (batch, channels x kernel_h x kernel_w, h x w)
        isa_mask = isa_mask.to(torch.promote_types(isa_mask.dtype, torch.float16))
        isa_mask = self._unfold(isa_mask)
        # new shape: (batch, channels, kernel_h, kernel_w, h, w)
        isa_mask = isa_mask.view(*mask.size()[:2], *self._kernel_size, *mask.size()[-2:])

        # apply AND to merge "is a ..." and "is part of ..." (on common device)
        self._ispartof_values = self._ispartof_values.to(isa_mask.device)  # select device
        isa_and_ispartof_mask: torch.Tensor = self.logical_and([
            isa_mask, self._ispartof_values])

        # apply exists quantifier
        # new shape: (batch, channels, h, w)
        out_mask = torch.amax(isa_and_ispartof_mask, dim=[2, 3])
        out_mask = out_mask.view(orig_size)
        return out_mask


class AbstractFuzzyIntersect(TorchOperation, abc.ABC):
    """Abstract class to define fuzzy intersection (over union) operations.
    The core method provided in this class is :py:meth:`torch_intersect_proportion`."""
    ARITY: int = 2

    @property
    def settings(self) -> Dict[str, Any]:
        return {**dict(logical_and=self.logical_and, logical_or=self.logical_or,
                       mask_dims=self.mask_dims, keep_dims=self.keep_dims,),
                **super().settings}

    @property
    def setting_defaults(self) -> Dict[str, Any]:
        return {**dict(logical_and=None, logical_or=None,
                       mask_dims=(-2, -1), keep_dims=False,),
                **super().setting_defaults}

    def __init__(self, *in_keys,
                 logical_and: Merge = None, logical_or: Merge = None,
                 mask_dims: Union[int, Sequence[int]] = (-2, -1),
                 keep_dims: bool = False,
                 **kwargs):
        super().__init__(*in_keys, **kwargs)
        self.logical_and: Optional[Merge] = logical_and
        """The logical AND operation to use for calculating mask intersection
        (assumed to be commutative).
        If ``None``, product logic AND is used."""
        self.logical_or: Optional[Merge] = logical_or
        """The logical OR operation to use for calculating mask union
        (assumed to be commutative).
        If ``None``, it calculates as ``1-self.logical_and([1-mask_a, 1-mask_b])``."""
        self.mask_dims: Tuple[int, ...] = (mask_dims,) if isinstance(mask_dims, int) else tuple(mask_dims)
        """The dimensions along which the area of a mask is defined
        (after broadcasting)."""
        if not all(isinstance(i, int) for i in self.mask_dims):
            raise TypeError("Received invalid mask_dims specification; "
                            "should be int or sequence of ints but was of type {}: {}"
                            .format(type(mask_dims), mask_dims))
        self.keep_dims: bool = keep_dims
        """Whether to keep dim 1 entries in the ``mask_dims`` dimensions."""

    def torch_intersect(self, *masks: torch.Tensor) -> torch.FloatTensor:
        intersect_mask = self.logical_and(masks) if self.logical_and is not None \
            else product.AND.torch_operation(*masks)
        return torch.clamp(intersect_mask.float(), min=0, max=1)

    def torch_union(self, *masks: torch.Tensor) -> torch.FloatTensor:
        if self.logical_or is not None:
            union_mask: torch.FloatTensor = self.logical_or(masks)
        else:
            outer = [1 - mask.float() for mask in masks]
            union_mask: torch.FloatTensor = 1 - self.torch_intersect(*outer)
        return torch.clamp(union_mask, min=0, max=1).float()

    def torch_intersect_proportion(self, *masks: torch.Tensor,
                                   iou: bool = True, mask_dims: Tuple[int, ...] = None
                                  ) -> torch.FloatTensor:
        r"""Calculate to what degree ``mask_a`` is covered by ``mask_b``.
        
        :return: Depending on ``iou``, this is the following for masks
            ``A=mask_a``, ``B=mask_b``, and fuzzy set membership function :math:`\in`:
        
            - ``iou=True``: intersection over union between ``mask_a`` and ``mask_b`` as
              :math:`\min(1, \frac{\sum_c c\in A \wedge c\in B}{(\sum_c c \in A \vee c \in B)})`
            - ``iou=False``: what proportion of ``mask_a`` area intersects with ``mask_b`` as
              :math:`\min(1, \frac{\sum_c c\in A \wedge c\in B}{\sum_a a \in A})`
              (here, :math:`a\vee b` is calculated as :math:`1 - ((1-a) \wedge (1-b))`)
        """
        mask_dims = mask_dims or self.mask_dims
        assert len(masks) > 0
        masks = torch.broadcast_tensors(*masks)
        if not iou:  # mask a
            denom_area: torch.FloatTensor = masks[0].float()
        else:  # masks logical union
            denom_area: torch.FloatTensor = self.torch_union(*masks)
        denom: torch.FloatTensor = denom_area.sum(dim=mask_dims, keepdim=self.keep_dims)

        intersect: torch.FloatTensor = \
            self.torch_intersect(*masks).sum(dim=mask_dims, keepdim=self.keep_dims)

        return torch.where(torch.isclose(denom, torch.zeros_like(denom)),
                           torch.ones_like(denom),
                           torch.clamp(intersect / denom, min=0., max=1.))


class CoveredBy(AbstractFuzzyIntersect):
    r"""Calculate the proportion of one mask covered by another.
    The value calculates for masks :math:`A, B` and fuzzy set membership function
    :math:`\in` as:

    .. math:: \min(1, \frac{\sum_c c\in A \wedge c\in B}{\sum_a a \in A})

    This is the summed fuzzy set intersection over the summed mask :math:`A`.
    """
    SYMB: str = "CoveredBy"

    def torch_operation(self, mask_a: torch.Tensor, mask_b: torch.Tensor):
        """Calculate to what degree ``mask_a`` is covered by ``mask_b``."""
        return self.torch_intersect_proportion(mask_a, mask_b, iou=False)


class IoUWith(AbstractFuzzyIntersect):
    r"""Calculate the intersection over union between two masks.
    The value calculates for masks :math:`A, B` and fuzzy set membership function
    :math:`\in` as:

    .. math:: \min(1, \frac{\sum_c c\in A \wedge c\in B}{(\sum_c c \in A \vee c \in B)})

    """
    SYMB: str = "IoUWith"
    IS_COMMUTATIVE: bool = True

    def torch_operation(self, mask_a: torch.Tensor, mask_b: torch.Tensor):
        """Calculate set intersection over union between ``mask_a`` and ``mask_b``."""
        return self.torch_intersect_proportion(mask_a, mask_b, iou=True)


class BestIoUWith(AbstractFuzzyIntersect):
    r"""Given two stacked sets of masks ``masks_a`` and ``masks_b``,
    calculate for each mask in ``masks_a`` the best IoU with any mask in ``masks_b``.
    Precisely, all entries in ``masks_a`` and ``masks_b`` are compared via IoU
    by varying over all dimensions except for :py:attr:`mask_dims` and :py:attr:`batch_dims`.
    The returned result is the stacked best IoUs, one for each mask in ``masks_a``.
    The input masks are assumed to have the same dimensionality in the :py:attr:`mask_dims`.
    The output mask will have the same size as ``masks_a`` only with :py:attr:`mask_dims` squeezed.

    Consider ``mask_a.size()==[batch, stack_a, h, w]`` and ``mask_b.size()==[batch, stack_b, h, w]``.
    The settings :py:attr:`mask_dim` = ``(-2, -1)`` (``h`` and ``w``) and
    :py:attr:`batch_dim` = ``(0,)`` then mean:

    - The output will have size ``[batch, stack_a]``.
    - The entry at index ``[batch_idx, s_a]`` is the maximum of IoUs between the mask ``masks_a[batch_idx, s_a]``
      and mask ``masks_b[batch_idx, s_b]`` for any value ``s_b in \range(stack_b)``.
    """
    SYMB: str = 'BestIoUWith'

    def __init__(self, *in_keys,
                 batch_dims: Optional[Sequence[int]] = None,
                 **kwargs):
        super().__init__(*in_keys, **kwargs)
        self.batch_dims: Sequence[int] = batch_dims
        """The dimensions to match of masks_a and masks_b before IoU comparison.
        Defaults to ``(0,)`` in case the masks have more than ``len(self.mask_dims)+1`` dimensions,
        else defaults to ``(,)``."""

    def torch_operation(self, masks_a: torch.Tensor, masks_b: torch.Tensor) -> torch.Tensor:
        """Calculate for each mask in ``masks_a`` with those in ``masks_b`` at same non-stack dims."""
        batch_dims = self.batch_dims
        if batch_dims is None:
            batch_dims = (0,) if len(masks_a.shape) > (len(self.mask_dims) + 1) else tuple()
        
        # No negative dims:
        assert len(masks_a.shape) == len(masks_b.shape), \
            "Cannot calc BestIoU for shapes of different length; got shapes {} and {}".format(
                masks_a.shape, masks_b.shape)
        batch_dims = [d if d>=0 else len(masks_a.shape)+d for d in batch_dims]
        mask_dims = [d if d>=0 else len(masks_a.shape)+d for d in self.mask_dims]
        other_dims = [*batch_dims, *mask_dims]
        
        batch_dim_vals = [masks_a.shape[i] for i in batch_dims]
        assert all(masks_b.shape[dim] == val for dim, val in zip(batch_dims, batch_dim_vals))

        # Move mask_dims to back and stacked dims to front to get shape [-1, *batch_dim_vals, *mask_dim_vals]
        stack_dims_a = [i for i in range(len(masks_a.shape)) if i not in other_dims]
        stack_dim_vals_a = [masks_a.shape[i] for i in stack_dims_a]
        other_dim_vals_a = [masks_a.shape[i] for i in other_dims]
        permutation_a = [*stack_dims_a, *other_dims]
        masks_a_flat = masks_a.permute(permutation_a).reshape([-1, *other_dim_vals_a])

        stack_dims_b = [i for i in range(len(masks_b.shape)) if i not in other_dims]
        other_dim_vals_b = [masks_b.shape[i] for i in other_dims]
        masks_b_flat = masks_b.permute([*stack_dims_b, *other_dims]).reshape([-1, *other_dim_vals_b])

        # Retrieve output of shape [prod(stack_dim_vals_a), *batch_dim_vals]
        best_ious: List[torch.FloatTensor] = []
        for mask_a in masks_a_flat:
            best_ious.append(self.torch_intersect_proportion(
                mask_a.unsqueeze(0), masks_b_flat, iou=True,
                mask_dims=list(range(-len(mask_dims), 0))
                ).amax(dim=0))
        best_ious_t: torch.FloatTensor = torch.stack(best_ious, dim=0)

        # Permute back:
        # shape [prod(stack_dim_vals_a), *batch_dim_vals] -> [*stack_dim_vals_a, *batch_dim_vals, 1...1]
        best_ious_t = best_ious_t.view([*stack_dim_vals_a, *batch_dim_vals, *([1]*len(mask_dims))])
        best_ious_t = best_ious_t.permute([permutation_a.index(i) for i in range(len(permutation_a))])
        
        # Discard 1s in mask dims if requested:
        if not self.keep_dims:
            best_ious_t = best_ious_t.view([best_ious_t.shape[i] for i in range(len(best_ious_t.shape))
                                            if i not in mask_dims])

        return best_ious_t



class AllNeighbors(TorchOperation):
    r"""Given a mask of truth values representing the output of a formula apply average pooling.
    Formally, the output mask :math:`(\text{AllNeighbors}(p))_{p\in\text{mask}}`
    for a mask representing the output of the predicate :math:`M` is defined as

    .. math::
        \text{AllNeighbors}(p) \coloneqq \left(
        \forall p_2\in\text{Neighborhood}(p)\colon M(p_2) \right)

    with the choice :math:`\forall=\text{mean}` and

    .. math::
        (p_2\in\text{Neighborhood}(p)) \coloneqq
        ( \|p-p_2\|_1 \leq \lfloor\frac{1}{2}(\text{kernel\_size} - 1)\rfloor ) \;.

    To ensure that neighborhoods are centralized around their output pixel,
    even kernel sizes are reduced by 1.
    A zero padding ensures that the returned mask is of the same size as the input.
    """
    SYMB: str = "AllNeighbors"
    ARITY: int = 1

    @property
    def settings(self) -> Dict[str, Any]:
        return dict(**dict(kernel_size=self.kernel_size,), **super().settings)

    @property
    def setting_defaults(self):
        """Defaults used for settings."""
        return dict(kernel_size=17, **super().setting_defaults)

    @property
    def kernel_size(self) -> int:
        """The width of the square representing a neighborhood."""
        return self._avg_pool.kernel_size

    def __init__(self, in_key: Union[str, Merge], *,
                 kernel_size: int = 17,
                 **other_setts):
        super().__init__(in_key, **other_setts)

        if kernel_size % 2 == 0:
            logging.getLogger().warning("Increased even kernel_size %d by 1 to ensure centralized neighborhoods.",
                                        kernel_size)
            kernel_size += 1
        self._avg_pool: torch.nn.AvgPool2d = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=1)
        self._zero_pad: torch.nn.ZeroPad2d = torch.nn.ZeroPad2d(
            padding=same_padding((self.kernel_size, self.kernel_size)))

    def torch_operation(self, mask: torch.Tensor) -> torch.Tensor:
        """Apply average pooling to single input and return mask of the same size."""
        out_mask: torch.Tensor = self._zero_pad(mask)
        out_mask = self._avg_pool(out_mask.to(torch.float))
        return out_mask
