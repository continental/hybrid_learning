#  Copyright (c) 2022 Continental Automotive GmbH
"""Loss and metric functions and classes that can be calculated per batch.

A good overview and collection can be found e.g. here:
https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
*(the contained code samples are quite instructive but in tensorflow, thus not
used here)*.
"""

import abc
import enum
from typing import Tuple, Callable, Union, Dict, Sequence, Any, Optional

import torch
import torch.nn.functional


# When overriding the forward method, the parameters should get more specific:
# pylint: disable=arguments-differ

def _settings_to_repr(obj, settings: Dict) -> str:
    """Given an object and a dict of its settings, return a representation str.
    The object is just used to derive the class name."""
    return "{}({})".format(str(obj.__class__.__name__),
                           ', '.join(['='.join([str(k), str(v)])
                                      for k, v in settings.items()]))


class BatchReduction(enum.Enum):
    """Aggregation types to reduce the 0th (meaning the batch) dimension of a
    tensor. The values are tuples of description and function."""
    mean = ("Reduce by mean", lambda t: t.mean())
    sum = ("Reduce by sum", lambda t: t.sum())

    def __call__(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """Reduce the given tensor according to the chosen aggregation method.
        """
        # pylint: disable=unsubscriptable-object
        return self.value[1](batch_tensor)
        # pylint: enable=unsubscriptable-object


class WeightedLossSum(torch.nn.Module):  # TODO: tests
    """Weighted sum of loss results."""

    def __init__(self,
                 losses: Sequence[Union[torch.nn.Module, Callable]],
                 weights: Sequence[float] = None):
        """Init.

        :param losses: list of losses to sum; all losses must return the same
            output format for sum
        :param weights: list of weights for the losses
        """
        # Value checks:
        if len(losses) == 0:
            raise ValueError("Empty loss list")
        if weights is not None and len(weights) != len(losses):
            raise ValueError(("Lengths of losses ({}) and weights ({}) do not "
                              "coincide").format(len(losses), len(weights)))

        super().__init__()

        self.losses: Sequence[Union[torch.nn.Module, Callable]] = losses
        """The losses the results of which to sum up."""
        self.weights: Sequence[float] = \
            weights or [1 / len(self.losses)] * len(self.losses)
        """Weights for each loss. Defaults to equal weights summing up to 1."""

    def __repr__(self) -> str:
        return (str(self.__class__.__name__) + '(' +
                'losses=[' + ', '.join(
                    [repr(loss_fn) for loss_fn in self.losses]) + '], ' +
                'weights=' + str(self.weights))

    def forward(self, *inp: Any) -> Any:
        """Forward method: Weighted sum of the loss values.
        All losses from :py:attr:`losses` are considered."""
        # pylint: disable=no-member
        return torch.stack([w * l(*inp)
                            for w, l in zip(self.weights, self.losses)]
                           ).sum(dim=0)
        # pylint: enable=no-member


class BalancedBCELoss(torch.nn.Module):
    r"""Balanced binary cross entropy loss.
    This is a wrapper around torch.nn.functional.binary_cross_entropy which
    allows to enter a class weighting factor :math:`b` to have for a batch
    :math:`B` of outputs and targets :math:`(x, y)` the formula

    .. math::
        \text{BalancedBCELoss}(B) = \text{reduction}(
            \sum_{(x,y)\in B}  b \cdot y \cdot \log(x) + (1-b)(1-y)\log(1-x)
        )

    If no fixed :py:attr:`factor_pos_class` is given, this is determined
    batch-wise as ``1-target.mean()``.
    Target values are assumed to be binary.
    Input values are assumed to be in :math:`(0,1]` if :py:attr:`from_logit`
    is false, else they are assumed to be in logit space.
    The reduction can be ``mean``, ``sum``, or ``none``.
    """

    def __init__(self, factor_pos_class: Optional[float] = None,
                 reduction: str = 'mean', from_logit: bool = False):
        """Init.

        :param factor_pos_class: balancing factor in [0,1] applied to the
            zero class
        :param reduction: one of

            - ``none``: no reduction
            - ``mean``: mean over batch dimension 0;
            - ``sum``: sum over batch dimension 0
        """
        if factor_pos_class and not 0 <= factor_pos_class <= 1:
            raise ValueError("factor_pos_class must be in [0,1], but was {}"
                             .format(factor_pos_class))
        super().__init__()

        self.factor_pos_class: Optional[float] = factor_pos_class
        """Balancing factor :math:`b` applied to the positive class;
        :math:`(1-b)` is applied to the negative class.
        If set to ``None``, it is calculated batch-wise as
        ``1-targets.mean()``."""

        self.reduction: str = reduction
        """Reduction method to aggregate batch results.
        One of 'none', 'mean', 'sum'"""

        self.from_logit: bool = from_logit
        """Whether to treat the input values as logits or as values in
        :math:`(0, 1]`."""

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor
                ) -> torch.Tensor:
        """Pytorch forward method."""
        if self.factor_pos_class is not None:
            factor_pos_class = self.factor_pos_class
        else:
            factor_pos_class = 1.0 - (targets.sum() / targets.numel())
        weight = (factor_pos_class * (targets > 0).float()) + \
                 ((1 - factor_pos_class) * (targets == 0).float())

        if self.from_logit:
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                inputs, targets, weight=weight, reduction=self.reduction)
        else:
            bce = torch.nn.functional.binary_cross_entropy(
                inputs, targets, weight=weight, reduction=self.reduction)
        return bce

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings dict to reproduce the instance"""
        return dict(factor_pos_class=self.factor_pos_class,
                    reduction=self.reduction)

    def __repr__(self) -> str:
        return _settings_to_repr(self, self.settings)

    def __str__(self) -> str:
        return repr(self)


class BalancedPenaltyReducedFocalLoss(torch.nn.Module):
    r"""Balanced version of the penalty reduced focal loss from CenterNet.
    Formula (with focal loss parameters :math:`\alpha,\beta` and balancing
    factor :math:`b\in[0,1]`):

    .. math::
        L(M, M^{pred}) = - \frac{1}{#(M\eqiv 1} \cdot \left(
        b\cdot \sum_{xy, M_{xy} \equiv 1}
        (1-M_{xy}^{pred})^\alpha \log(M_{xy}^{pred}) +
        (1-b)\cdot \sum_{xy, M_{xy} < 1}
        (1-M_{xy})^\beta (M_{xy}^{pred})^\alpha \log(M_{xy}^{pred})
        \right)
    """

    def __init__(self, factor_pos_class: float = 0.5,
                 alpha: float = 2, beta: float = 4):
        super().__init__()
        if not 0 <= factor_pos_class <= 1:
            raise ValueError("factor_pos_class must be in [0,1], but was {}"
                             .format(factor_pos_class))
        self.factor_pos_class = factor_pos_class
        """Balancing factor :math:`b` applied to the positive class;
        :math:`(1-b)` is applied to the negative class."""
        self.alpha: float = alpha
        """Focal loss hyper-param:: potency for pixels of the predicted mask."""
        self.beta: float = beta
        """Focal loss hyper-param:: potency for pixels of the target mask."""

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings dict to reproduce the instance"""
        return dict(factor_pos_class=self.factor_pos_class,
                    alpha=self.alpha,
                    beta=self.beta)

    def __repr__(self) -> str:
        return _settings_to_repr(self, self.settings)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor
                ) -> torch.Tensor:
        """Pytorch forward method.

        :param inputs: tensor of shape ``(batch, channels, height, width)``
        :param targets: same as ``inputs``
        """
        pos_indices = (targets == 1).float()
        neg_indices = (targets < 1).float()

        # For stability: Ensure the entries not considered for the sum are
        # masked by values >> 0 in the log.
        pos_loss = (self.factor_pos_class *
                    pos_indices *
                    torch.pow(1 - inputs, self.alpha) *
                    torch.log(inputs + neg_indices)  # masking
                    )
        neg_loss = ((1 - self.factor_pos_class) *
                    neg_indices *
                    torch.pow(1 - targets, self.beta) *
                    torch.pow(inputs, self.alpha) *
                    torch.log(1 - inputs + pos_indices)  # masking
                    )
        num_pos = pos_indices.sum()

        loss = ((- neg_loss.sum()) if num_pos == 0 else
                (- (pos_loss.sum() + neg_loss.sum()) / num_pos))
        return loss


class MaskRCNNLoss(torch.nn.Module):
    """Loss and associated data for a standard Mask R-CNN model."""

    # noinspection PyMethodMayBeStatic
    def forward(self,
                *args: Dict[str, torch.Tensor]
                ):  # pylint: disable=no-self-use
        # pylint: disable=line-too-long
        """Calculate loss from output of the standard pytorch Mask R-CNN model
        in train mode.
        The model is assumed to provide an output format as is expected
        in the original
        `pytorch source code <https://github.com/pytorch/vision/blob/master/references/detection/engine.py>`_
        which is a dict of loss values for each optimization aspect.

        :param args: first argument is the output of pytorch Mask R-CNN
        """
        # pylint: enable=line-too-long
        loss_dict = args[0]
        return sum(loss for loss in loss_dict.values())


class AbstractIoULike(torch.nn.Module, metaclass=abc.ABCMeta):
    """General functions for intersection over union like calculation on
    binarized in- and output.
    See sub-classes for details."""

    @staticmethod
    def _validate_dimensions(labels: torch.Tensor,
                             outputs: torch.Tensor) -> None:
        """Validate whether the dimensions of labels and outputs are correct.
        Raise if not. Criteria: must have

        - same sizes
        - be at least 1D
        """
        tensor_dimensionality: int = len(outputs.size())
        if len(outputs.size()) != len(labels.size()):
            raise ValueError(("Outputs ({}) and labels ({}) have different "
                              "sizes!").format(outputs.size(), labels.size()))
        if not tensor_dimensionality >= 1:
            raise ValueError(("Output dimension ({}) too small; must be at "
                              "least 1D").format(outputs.size()))

    @staticmethod
    def get_area_axes(outputs: torch.Tensor
                      ) -> Union[Tuple[int, int], Tuple[int]]:
        """Get axe[s] that describe width [and height], meaning 2D or 1D areas
        to test for IoU.

        :If >=2D:
            the last 2 axes (2D areas)

        :If 1D:
            the last axis (1D area)

        :return: tuple with the indices of the area axes
        """
        tensor_dimensionality: int = len(outputs.size())
        w_axis = tensor_dimensionality - 1
        area_axes: Tuple[int] = (w_axis, w_axis - 1) \
            if tensor_dimensionality > 1 else (w_axis,)
        return area_axes

    @staticmethod
    def binarize(tensor: torch.Tensor, thresh: float) -> torch.Tensor:
        """Binarize a tensor to an int tensor according to threshold.

        :return: binarized tensor of dtype ``bool``
            with entries 1 if > threshold, and 0 if < threshold.
        """
        return (tensor > thresh).bool()

    def __repr__(self):
        return _settings_to_repr(self, self.settings)

    def __str__(self):
        return repr(self)

    @abc.abstractmethod
    def forward(self, *inp: Any, **kwargs: Any) -> Any:
        """Loss or metric function definition in sub-classes."""
        raise NotImplementedError()


class AbstractIoUMetric(AbstractIoULike):
    """Common properties of IoU calculation."""

    def __init__(self, output_thresh: float = 0.5, label_thresh: float = 0.,
                 smooth: float = 1e-6):
        """Init.

        :param output_thresh: threshold for binarizing the output
        :param label_thresh: threshold for binarizing the labels
        :param smooth: summand to smooth the IoU value (evade division by 0)
        """
        super().__init__()
        self.output_thresh: float = output_thresh
        """Threshold for binarizing the output; 1 if > output, 0 else."""
        self.label_thresh: float = label_thresh
        """Threshold for binarizing the labels; 1 if > output, 0 else."""
        self.smooth: float = smooth
        r"""Smoothening summand to avoid division by zero.
        Division :math:`\frac{a}{b}` is changed to
        :math:`\frac{a + \text{smooth}}{b + \text{smooth}}`."""

    @abc.abstractmethod
    def forward(self, *inp: Any, **kwargs: Any) -> Any:
        """Metric function definition in sub-classes."""
        raise NotImplementedError()

    @property
    def settings(self):
        """Dictionary with settings to reproduce instance."""
        return dict(output_thresh=self.output_thresh,
                    label_thresh=self.label_thresh,
                    smooth=self.smooth)

    def smooth_division(self, dividend, divisor):
        """Smoothed division using smoothening summand to avoid division by 0.

        :return: result of smooth division."""
        return (dividend + self.smooth) / (divisor + self.smooth)


class IoU(AbstractIoUMetric):
    r"""Calc sample-wise intersection over union (IoU) values output batch.
    The intersection over union for one instance calculates as

    .. math::
        \frac{intersection}{union} =  \frac{TP} {(TP + TN + FP + FN)}

    with

    - FP / TP: false / true positives,
      i.e. in- / correctly predicted foreground pixels
    - FN / TN: false / true positives,
      i.e. in- / correctly predicted background pixels

    The following tensor dimensions are allowed:

    - 1D: The tensor is assumed to be 1D without batch dimension.
    - 2D: The tensor is assumed to be 2D without batch dimension.
    - >2D: The tensor is assumed to be 2D with batch dimension 0,
      width dim. -1, height dim. -2.
    """

    def __init__(
            self,
            reduction: Union[
                BatchReduction, Callable[[torch.Tensor], torch.Tensor]
            ] = BatchReduction.mean,
            output_thresh: float = 0.5, label_thresh: float = 0.,
            smooth: float = 1e-6):
        """Init.

        :param reduction: reduction method to aggregate the instance-wise
            results of the batch;
            must be a callable on a tensor which reduces the 0th dimension;
            see BatchReduction instances for examples
        :param output_thresh: threshold for binarizing the output
        :param label_thresh: threshold for binarizing the labels
        :param smooth: summand to smooth the IoU value (evade division by 0)
        """
        super().__init__(output_thresh=output_thresh,
                         label_thresh=label_thresh,
                         smooth=smooth)
        self.reduction: Union[
            BatchReduction,
            Callable[[torch.Tensor], torch.Tensor]
        ] = reduction
        """Reduction method to aggregate the instance-wise results of the
        batch into one value."""

    def forward(self, outputs: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """Sample-wise reduced IoU between binarized in- and output.
        Applied reduction is :py:attr:`reduction`.

        :param outputs: Output tensors of shape ``(BATCH x H x W)``;
            values must be in [0, 1], and a pixel value > output_thresh means
            it is foreground
        :param labels: Label tensors of shape ``(BATCH x H x W)``;
            values must be in [0, 1], and a pixel value > label_thresh means
            it is foreground
        :return: tensor containing IoU for each sample along axis 0 reduced
            by reduction scheme
        """
        # Validate, binarize and turn into integer tensors:
        self._validate_dimensions(labels, outputs)
        labels = self.binarize(labels, self.label_thresh)
        outputs = self.binarize(outputs, self.output_thresh)

        # Get axes that describe width (and height), i.e. 2D or 1D areas to
        # test on IoU
        area_axes = self.get_area_axes(outputs)

        # Calculate IoU per sample
        # intersections for each area:
        intersections = (outputs & labels).float().sum(area_axes)
        # unions for each area:
        unions = (outputs | labels).float().sum(area_axes)
        # smoothed set IoU for each area:
        ious = self.smooth_division(intersections, unions)

        return self.reduction(ious)

    @property
    def settings(self):
        """Settings dict for reproduction of instance."""
        return dict(**super().settings, reduction=self.reduction)


class AbstractIoULoss(AbstractIoULike):
    """Shared settings for intersection over union based losses.
    The difference to IoU based metrics is that only the targets are binarized,
    not the outputs.
    Thus, the function on the DNN outputs stays smoothly differentiable.
    """

    def __init__(self,
                 reduction: Union[
                     BatchReduction, Callable[[torch.Tensor], torch.Tensor]
                 ] = BatchReduction.mean,
                 target_thresh: float = 0.):
        """Init.

        :param target_thresh: threshold to binarize targets
        :param reduction: reduction method to aggregate the instance-wise
            results of the batch;
            must be a callable on a tensor which reduces the 0th dimension;
            for examples see BatchReduction instances
        """
        super().__init__()
        self.target_thresh: float = target_thresh
        """Threshold to binarize the targets."""
        self.reduction: Union[BatchReduction,
                              Callable[[torch.Tensor], torch.Tensor]] = \
            reduction
        """Reduction method to aggregate the instance-wise results of the batch.
        """

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings dict to reproduce instance."""
        return dict(target_thresh=self.target_thresh, reduction=self.reduction)

    @abc.abstractmethod
    def forward(self, *inp: Any, **kwargs: Any) -> Any:
        """Loss function definition in sub-classes."""
        raise NotImplementedError()


class TverskyLoss(AbstractIoULoss):  # TODO: tests
    # noinspection SpellCheckingInspection
    r"""Calc Tversky loss (balanced Dice loss) for given outputs amd targets.
    The Tversky loss [Salehi2017]_ works on masks of prediction and ground
    truth (gt) indicating the foreground (fg) area.
    The masks may be binary, non-binary or mixed.
    The target masks are binarized.

    Given a balancing factor b, the loss is calculated for one instance as

    .. math::
        :label: tversky

        \text{Tversky} = \frac{TP} {(TP + b\cdot FP + (1-b) \cdot FN)}

    with

    - TP: true positives,
      respectively the intersection of predicted fg area and gt fg area
    - FP: false positives,
      respectively the predicted fg area minus the gt fg area

    For b=0.5 this is regular Dice loss.

    The following tensor dimensions are allowed:

    - 1D: The tensor is assumed to be 1D without batch dimension.
    - 2D: The tensor is assumed to be 2D without batch dimension.
    - >2D: The tensor is assumed to be 2D with batch dimension 0,
      width dim. -1, height dim. -2.

    .. [Salehi2017] S. S. M. Salehi, D. Erdogmus, and A. Gholipour.
        Tversky loss function for image segmentation using 3D fully
        convolutional deep networks, 2017.
        https://arxiv.org/abs/1706.05721
    """

    def __init__(self,
                 factor_false_positives: float = 0.7,
                 reduction: Union[
                     BatchReduction, Callable[[torch.Tensor], torch.Tensor]
                 ] = BatchReduction.mean,
                 target_thresh: float = 0.,
                 from_logit: bool = False):
        """Init.

        :param target_thresh: threshold to binarize targets
        :param factor_false_positives: factor in [0,1] applied to the false
            positives (see Tversky loss formula :math:numref:`tversky`)
        :param reduction: reduction method to aggregate the instance-wise
            results of the batch;
            must be a callable on a tensor which reduces the 0th dimension;
            for examples see instances of
            :py:class:`~hybrid_learning.concepts.train_eval.kpis.batch_kpis.BatchReduction`.
        """
        # Value check:
        if not 0 <= factor_false_positives <= 1:
            raise ValueError(("factor_false_positives must be in [0,1] but "
                              "was {}").format(factor_false_positives))

        super().__init__(target_thresh=target_thresh,
                         reduction=reduction)
        self.factor_false_positives: float = factor_false_positives
        """Factor applied to the false positives"""
        self.from_logit = from_logit

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        return dict(factor_false_positives=self.factor_false_positives,
                    **super().settings)

    def forward(self, outputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """Tversky loss :math:numref:`tversky` calculation.

        :param outputs: input tensor (at least 1D); items must be floats
            in the range [0,1]
        :param targets: targets to compare outputs with (at least 1D;
            same dimension as input)
        :return: aggregated Tversky loss :math:numref:`tversky` of outputs
            for given targets
        """

        # Validate dimensions and binarize targets:
        self._validate_dimensions(outputs, targets)
        targets: torch.Tensor = self.binarize(
            targets, self.target_thresh).float()

        # Get axes to work on (i.e. 2D or 1D areas to test on IoU)
        area_axes: Tuple[int] = self.get_area_axes(outputs)

        if self.from_logit:
            outputs = torch.sigmoid(outputs)
        # Calculate Tversky loss
        factor_false_negatives = 1.0 - self.factor_false_positives
        true_pos = (targets * outputs).sum(area_axes)
        false_pos = (- (targets - 1) * outputs).sum(area_axes)
        false_neg = (- targets * (outputs - 1)).sum(area_axes)
        tversky: torch.Tensor = \
            (true_pos / (true_pos +
                         self.factor_false_positives * false_pos +
                         factor_false_negatives * false_neg))
        loss: torch.Tensor = - tversky + 1

        # reduction
        loss = self.reduction(loss)

        return loss


class Net2VecLoss(AbstractIoULoss):  # TODO: tests
    # noinspection SpellCheckingInspection,SpellCheckingInspection
    r"""Simplified intersection over union as loss.
    This loss is the one used for the
    `original implementation <https://github.com/ruthcfong/net2vec>`_  of the
    Net2Vec framework [Fong2018]_
    *(even though this is a rewrite and no code is used from there)*.
    It works on masks of prediction and ground truth (gt) indicating the
    foreground (fg) area.
    The masks may be binary, non-binary or mixed.
    The target masks are binarized.

    Given For an instance, it calculates as

    .. math::
        :label: net2vec

        \text{Net2Vec}(instance) = b \cdot TP + (1-b) \cdot TN

    with

    - TP: true positives, resp. the intersection of predicted fg area and
      gt fg area
    - TN: true negatives, resp. the intersection of predicted background (bg)
      area and gt bg area

    The following tensor dimensions are allowed:

    - 1D: The tensor is assumed to be 1D without batch dimension.
    - 2D: The tensor is assumed to be 2D without batch dimension.
    - >2D: The tensor is assumed to be 2D with batch dimension 0,
      width dim. -1, height dim. -2.

    .. [Fong2018] R. Fong and A. Vedaldi, “Net2Vec: Quantifying and explaining
        how concepts are encoded by filters in deep neural networks”
        in Proc. 2018 IEEE Conf. Comput. Vision and Pattern Recognition,
        Salt Lake City, UT, USA, 2018, pp. 8730–8738,
        https://arxiv.org/abs/1801.03454
    """

    def __init__(
            self,
            factor_pos_class: float = 0.5,
            reduction: Union[
                BatchReduction, Callable[[torch.Tensor], torch.Tensor]
            ] = BatchReduction.mean,
            target_thresh: float = 0.):
        """Init.

        :param target_thresh: threshold to binarize targets
        :param factor_pos_class: balancing factor :math:`b` in [0,1] applied
            to the foreground (1) class; defaults to 0.5 (i.e. equal weighting)
        :param reduction: reduction method to aggregate the instance-wise
            results of the batch;
            must be a callable on a tensor which reduces the 0th dimension;
            for examples see instances of
            :py:class:`~hybrid_learning.concepts.train_eval.kpis.batch_kpis.BatchReduction`
        """
        # Value check:
        if not 0 <= factor_pos_class <= 1:
            raise ValueError("factor_pos_class must be in [0,1] but was {}"
                             .format(factor_pos_class))

        super().__init__(target_thresh=target_thresh,
                         reduction=reduction)
        self.factor_pos_class: float = factor_pos_class
        """Balancing factor :math:`b` applied to the foreground (value 1) class.
        See loss formula :math:numref:`net2vec`."""

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings to reproduce the instance."""
        return dict(factor_pos_class=self.factor_pos_class,
                    **super().settings)

    def forward(self, outputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """Calculate Net2Vec loss :math:numref:`net2vec`.

        :param outputs: tensor of predicted masks (at least 1D); items must be
            floats in [0,1]
        :param targets: ground truth masks
        :return: net2vec loss for each instance; reduced to one value if batch
        """
        # Validate dimensions and binarize targets
        self._validate_dimensions(targets, outputs)
        targets = self.binarize(targets, self.target_thresh).float()

        # Get axes to work on (i.e. 2D or 1D areas to test on IoU)
        area_axes: Tuple[int] = self.get_area_axes(outputs)

        # Calculate net2vec loss
        net2vec = - (self.factor_pos_class * targets * outputs +
                     (1 - self.factor_pos_class) * (1 - targets) * (1 - outputs)
                     ).sum(area_axes)

        return self.reduction(net2vec)
