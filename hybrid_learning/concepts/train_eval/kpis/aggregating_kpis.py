"""
Implementation of KPIs that cannot be computed on a batch level.
They aggregate the result over all batches and can return an aggregated result,
if requested.

@author: Christian Wirth
"""
#  Copyright (c) 2022 Continental Automotive GmbH

# Typical plot names are not supported by pylint:
# pylint: disable=invalid-name
import abc
from abc import abstractmethod, ABC
from math import sqrt
from typing import Optional, List, Union, Dict, Callable

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from ....datasets.transforms import ReduceTuple

matplotlib.use("Agg")  # prevent from creating tkinter windows


def filter_aggregating_kpi_keys(metric_fns: Dict[
    str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
                                ) -> List[str]:
    """Given named metric functions return the keys of such behaving like an
    AggregatingKpi.
    This may be either an :py:class:`AggregatingKpi` directly or a
    :py:class:`hybrid_learning.datasets.transforms.tuple_transforms.ReduceTuple`
    thereof."""
    return [name for name, metric_fn in metric_fns.items()
            if (isinstance(metric_fn, AggregatingKpi)
                or (isinstance(metric_fn, ReduceTuple)
                    and isinstance(metric_fn.reduction, AggregatingKpi))
                or (isinstance(metric_fn, ReduceTuple)
                    and isinstance(metric_fn.reduction, ReduceTuple)
                    and isinstance(metric_fn.reduction.reduction,
                                   AggregatingKpi)))]


class AggregatingKpi(torch.nn.Module, ABC):
    """
    Abstract base class for all aggregating kpis, meaning they can not be
    computed on batch level, but need to be aggregated over batches.
    """

    def __init__(self, name):
        super().__init__()
        self.name = name

    @abstractmethod
    def update(self, outputs: torch.Tensor,
               labels: torch.Tensor):
        """
        Called every batch. Shall be used to update the aggregating statistics.

        :param outputs: Output tensors of shape ``(BATCH x H x W)``;
        :param labels: Label tensors of shape ``(BATCH x H x W)``;
        """
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        """
        Called once at the beginning of a new epoch.
        Shall reset the aggregating statistics.
        """
        raise NotImplementedError()

    def value(self) -> torch.Tensor:
        """
        Shall return the aggregated metric value as a scalar.

        :return: tensor containing the given metric value.
        """
        raise NotImplementedError()

    def value_and_reset(self):
        """
        Shorthand for subsequent calls to ``value`` and to ``reset``.
        E.g. use at the end of a run to make sure that no side effects of previous runs can leak.
        """
        val = self.value()
        self.reset()
        return val

    def forward(self, outputs: torch.Tensor,
                labels: torch.Tensor
                ):
        """Calculate KPI without gradient.
        Uses :py:meth:`~AggregatingKpi.value`."""
        with torch.no_grad():
            self.update(outputs, labels)


class ConfusionMatrix(AggregatingKpi, abc.ABC):
    """Abstract base class for all confusion matrix based metrics.
    Saves counts for the confusion matrix in :py:attr:`matrix` as follows:

    - ``matrix[:, 0]``: True predictions
    - ``matrix[:, 1]``: False predictions
    - ``matrix[0, :]``: Positive predictions
    - ``matrix[1, :]``: Negative predictions
    """

    def __init__(self, threshold: float = 0.5):
        """Init.

        :param threshold: see :py:attr:`~ConfusionMatrix.threshold`
        """
        super().__init__(
            f'{str(self.__class__).split(self.__class__.__module__)[1][1:-2]}@{threshold}'.replace(
                ".", ""))
        self.threshold: float = threshold
        """Threshold for the positive class."""

        self.matrix: torch.Tensor = torch.zeros(2, 2)
        """Storage for the current confusion matrix values."""

        self.register_buffer(self.name + "_matrix", self.matrix,
                             persistent=False)

    def update(self, outputs: torch.Tensor,
               labels: torch.Tensor
               ):  # pylint: disable=invalid-name,singleton-comparison
        outputs = (outputs > self.threshold).bool()
        labels = (labels > self.threshold).bool()

        tp = ((outputs == labels) & (outputs == True)).float().sum()
        tn = ((outputs == labels) & (outputs == False)).float().sum()
        fp = ((outputs != labels) & (outputs == True)).float().sum()
        fn = ((outputs != labels) & (outputs == False)).float().sum()

        self.matrix[0, 0] = self.matrix[0, 0] + tp
        self.matrix[1, 0] = self.matrix[1, 0] + tn
        self.matrix[0, 1] = self.matrix[0, 1] + fp
        self.matrix[1, 1] = self.matrix[1, 1] + fn

    def reset(self):
        torch.nn.init.constant_(self.matrix, 0)


def save_division(numer: torch.Tensor, denom: torch.Tensor, div_by_zero_val: float = None) -> torch.Tensor:
    """Yield sensible values in case of division by zero.
    In case of 0/0 return 1, in case of x/0 return ``div_by_zero_val``."""
    numer, denom = torch.broadcast_tensors(numer, denom)
    denom_is_zero = torch.isclose(denom, torch.zeros_like(denom))
    numer_is_zero = torch.isclose(numer, torch.zeros_like(numer))
    ones_like_denom = torch.ones_like(denom)

    res = torch.where(denom_is_zero & numer_is_zero,
                      ones_like_denom,
                      numer / denom)
    if div_by_zero_val is None:
        return res
    return torch.where(denom_is_zero & torch.logical_not(numer_is_zero),
                       ones_like_denom * div_by_zero_val,
                       res)


class Accuracy(ConfusionMatrix):
    """Binary accuracy.
    Calculates as: ``(TP + TN) / (TP + FN + TN + FP)``
    """

    def value(self):
        return save_division((self.matrix[0, 0] + self.matrix[1, 0]), self.matrix.sum())


class F1Score(ConfusionMatrix):
    """Binary F1-score.
    Calculates as: ``2 (precision * recall) / (precision + recall) = 2*TP / (2TP + FP + FN)``
    """

    def value(self):
        denom: torch.Tensor = 2 * self.matrix[0, 0] + self.matrix[0, 1] + self.matrix[1, 1]
        return save_division(2 * self.matrix[0, 0], denom)


class Recall(ConfusionMatrix):
    """Binary recall (resp. sensitivity resp. true positive rate).
    Calculates as: ``TP / (TP + FN)``
    """

    def value(self):
        return save_division(self.matrix[0, 0], (self.matrix[0, 0] + self.matrix[1, 1]))


class Specificity(ConfusionMatrix):
    """Binary specificity (resp. true negative rate).
    Calculates as: ``TN / (TN + FP)``
    """

    def value(self):
        return save_division(self.matrix[1, 0], (self.matrix[1, 0] + self.matrix[0, 1]))


class Precision(ConfusionMatrix):
    """Binary precision.
    Calculates as: ``TP / (TP + FP)``
    """

    def value(self):
        return save_division(self.matrix[0, 0], (self.matrix[0, 0] + self.matrix[0, 1]))


class NegativePredictiveValue(ConfusionMatrix):
    """Binary negative predictive value.
    Calculates as: ``TN / (TN + FN)``
    """

    def value(self):
        return save_division(self.matrix[1, 0], (self.matrix[1, 0] + self.matrix[1, 1]))


class ECE(AggregatingKpi):  # pylint: disable=too-many-instance-attributes
    r"""
    Expected calibration error.
    The formula first calculates per bin the absolute difference between
    the fraction of predictions that are correct (accuracy) and
    the mean of the predicted probabilities in the bin (confidence).
    The final expected calibration error then is the sum of these bin-wise
    differences, each weighted by the fraction of predictions in the bin
    and total number of predictions:

    .. math::
        ECE = \sum_{b=1}^{B} \frac{N_b}{N} | text{acc}(b) - text{conf}(b) |

    for bins :math:`B`, and :math:`N, N_b` the total number of predictions
    respectively the predictions falling into bin :math:`b`.
    Bins are here selected to be equally sized with respect to lower and
    upper bound.

    For details see [Naeini2015]_ and [Nixon2019]_

    .. [Naeini2015] Naeini, Mahdi Pakdaman, Cooper, Gregory F,
        and Hauskrecht, Milos.
        Obtaining well calibrated probabilities using bayesian binning.
        In AAAI, pp. 2901, 2015. http://europepmc.org/article/PMC/4410090
    .. [Nixon2019] Nixon, Jeremy, Michael W. Dusenberry, Linchuan Zhang, Ghassen Jerfel,
        and Dustin Tran.
        Measuring Calibration in Deep Learning.
        In CVPR WS, pp. 38-41. 2019. https://arxiv.org/abs/1904.01685
    """

    def __init__(self, n_bins: int = 10, threshold: float = 0.5,
                 threshold_discard: float = 0.0,
                 only_class: int = None,
                 max_prob: bool = True,
                 adaptive: bool = False,
                 class_conditional: bool = False,
                 aggregation: str = "expectation"):
        """Init."""
        name = f'ECE@{threshold}'.replace(".", "")
        if only_class:
            name += f'cls{only_class}'
        if max_prob:
            name += "maxP"
        if adaptive:
            name += "Adap"
        if class_conditional:
            name += "Cond"

        super().__init__(name)

        self.n_bins = n_bins
        """Number of bins. Upper and lower bounds are chosen according to the
        setting in :py:attr:`adaptive`."""

        self.threshold: float = threshold
        """Threshold for the selecting positive class."""
        self.only_class: int = only_class
        """Compute ECE only for a single class, condition on the predicted class"""
        self.max_prob: bool = max_prob
        """Only consider the predicted class (equals maximal probability class
        for threshold 0.5)"""
        self.adaptive: bool = adaptive
        """Whether to use adaptive binning. Adaptive means the upper and
        lower bounds of bins are calculated from the complete final results
        such that each bin contains ``1/n_bins`` samples."""
        self.aggregation: str = aggregation
        """Aggregation specifier. Must be one of ``"mean"``, ``"expectation"``,
        or ``"max"``."""
        self.threshold_discard: float = threshold_discard
        """If not ``None``, discard all values below this threshold."""
        self.class_conditional: bool = class_conditional
        """Whether to bin each class separately."""

        if adaptive:
            self.prediction = []
            self.ground_truth = []

        self.prob_sum: torch.Tensor = torch.zeros(
            (2 if class_conditional and only_class is None else 1, n_bins))
        """Buffer collecting the sum of probabilities for each bin."""
        self.correct_sum: torch.Tensor = torch.zeros(
            (2 if class_conditional and only_class is None else 1, n_bins))
        """Buffer collecting the total amount of true positives and true
        negatives per bin."""
        self.count: torch.Tensor = torch.zeros(
            (2 if class_conditional and only_class is None else 1, n_bins))
        """Total amount of predictions per bin."""

        self.register_buffer(self.name + "_prob_sum", self.prob_sum,
                             persistent=False)
        self.register_buffer(self.name + "_correct_sum", self.correct_sum,
                             persistent=False)
        self.register_buffer(self.name + "_count", self.count, persistent=False)

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        outputs = torch.flatten(outputs)
        labels = torch.flatten(labels)

        if self.threshold_discard is not None:
            relevant = outputs >= self.threshold_discard
            outputs = outputs[relevant]
            labels = labels[relevant]

        if self.adaptive:
            self._add_to_adaptive(outputs, labels)
        else:
            prob, pred, y_true = self._create_2d(outputs, labels)
            prob, y_true = self._filter_relevant(prob, pred, y_true)
            self._add_to_static_bin(prob, y_true)

    def _create_2d(self, outputs: torch.Tensor, labels: torch.Tensor):
        """Represent probabilities, predictions, and labels as 2D tensors
        with dims for pos and neg class."""
        # Binarize
        y_true = (labels > self.threshold).float()
        pred = (outputs > self.threshold).float()

        # As (class, examples) matrix
        prob = torch.stack([1.0 - outputs, outputs])
        pred = torch.stack([1.0 - pred, pred])
        y_true = torch.stack([1.0 - y_true, y_true])

        return prob, pred, y_true

    def _filter_relevant(self, prob, pred, y_true):
        """Filter values according to settings."""
        if self.only_class is not None:
            # Select examples belonging to the according class
            prob_pair = [[], []]
            y_true_pair = [[], []]

            prob_pair[self.only_class] = prob[self.only_class]
            y_true_pair[self.only_class] = y_true[self.only_class]
            pred = pred[self.only_class]

            if self.max_prob:
                idx = torch.eq(pred, 1)
                prob_pair[self.only_class] = prob_pair[self.only_class][idx]
                y_true_pair[self.only_class] = y_true_pair[self.only_class][idx]

        elif self.max_prob:
            # Gather a tensor with the confidences for the predicted class,
            # so output for a positive label and 1-output for a negative label
            idx = torch.argmax(pred, dim=0, keepdim=False)

            prob_pair = [prob[0, idx == 0], prob[1, idx == 1]]
            y_true_pair = [y_true[0, idx == 0], y_true[1, idx == 1]]
        else:
            prob_pair = [prob[0], prob[1]]
            y_true_pair = [y_true[0], y_true[1]]

        return prob_pair, y_true_pair

    def _add_to_adaptive(self, outputs, labels):
        """Store outputs and labels."""
        self.prediction.append(outputs.detach().cpu().half())
        self.ground_truth.append(labels.detach().cpu().half())

    def _binning(self, prob, y_true, class_id):
        """Add counts and sums per bin.
        Bin lower and upper bounds are chosen according to the
        :py:attr:`adaptive` setting."""
        if self.adaptive:
            linspace = np.interp(
                np.linspace(0, len(prob) / 4, self.n_bins + 1),
                np.arange(len(prob) / 4),
                np.sort(prob[::4]))  # approx. binning by quarter of datapoints
            linspace[0] -= 1e-6
        else:
            # if max_prob is active, predictions can only be > threshold
            # as only the predicted class is considered
            linspace = np.linspace(
                np.minimum(self.threshold, 1.0 - self.threshold)
                - 1e-6 if self.max_prob else -1e-6,
                1, self.n_bins + 1)

        low_bound: np.ndarray = linspace[0:-1]  # Lower bound per bin
        high_bound: np.ndarray = linspace[1:]  # Upper bound per bin

        for bin_nr in range(self.n_bins):
            relevant = ((prob > low_bound[bin_nr]) &
                        (prob <= high_bound[bin_nr]))
            # Summed probabilities per bin
            self.prob_sum[class_id, bin_nr] = \
                self.prob_sum[class_id, bin_nr] + prob[relevant].float().sum()
            # Correct predicted per class - This is TP + TN (per bin)
            # when using max_prob (as we only consider the predicted label class)
            self.correct_sum[class_id, bin_nr] = \
                self.correct_sum[class_id, bin_nr] + y_true[relevant].sum()
            self.count[class_id, bin_nr] = (self.count[class_id, bin_nr]
                                            + relevant.float().sum())

    def _add_to_static_bin(self, prob, y_true):
        """Add counts and sums per bin using static bins.
        May be called each batch."""
        if not self.class_conditional:
            # Concatenate positive and negative class examples in one list
            prob = torch.cat(prob, dim=0)
            y_true = torch.cat(y_true, dim=0)

            self._binning(prob, y_true, class_id=0)
        else:
            self._binning(prob[0], y_true[0], class_id=0)
            self._binning(prob[1], y_true[1], class_id=1)

    def reset(self):
        if self.adaptive:
            self.ground_truth = []
            self.prediction = []

        torch.nn.init.constant_(self.prob_sum, 0)
        torch.nn.init.constant_(self.correct_sum, 0)
        torch.nn.init.constant_(self.count, 0)

    def value(self) -> torch.Tensor:
        if self.adaptive:
            outputs = torch.cat(self.prediction, dim=0)
            labels = torch.cat(self.ground_truth, dim=0)

            prob, pred, y_true = self._create_2d(outputs, labels)
            prob, y_true = self._filter_relevant(prob, pred, y_true)

            torch.nn.init.constant_(self.prob_sum, 0)
            torch.nn.init.constant_(self.correct_sum, 0)
            torch.nn.init.constant_(self.count, 0)

            if not self.class_conditional:
                # Concatenate positive and negative class examples in one list
                prob = torch.cat(prob, dim=0)
                y_true = torch.cat(y_true, dim=0)

                self._binning(prob, y_true, class_id=0)
            else:
                self._binning(prob[0], y_true[0], class_id=0)
                self._binning(prob[1], y_true[1], class_id=1)

        acc = self.correct_sum / self.count  # accuracy
        conf = self.prob_sum / self.count  # mean probability
        if self.aggregation == "mean":
            err = torch.abs(acc - conf)
            class_ece = torch.stack(
                [err[i, ~torch.isnan(err[i])].mean() for i in range(len(err))])
        elif self.aggregation == "expectation":
            rel_count = self.count / self.count.sum(dim=1, keepdim=True)
            class_ece = (rel_count * torch.abs(acc - conf)).nansum(dim=1)
        elif self.aggregation == 'max':
            err = torch.abs(acc - conf)
            class_ece = err[~torch.isnan(err)].max()
        else:
            raise ValueError(
                ("Unexpected value {} for aggregation. Must be one of {}"
                 ).format(self.aggregation, ('mean', 'expectation', 'max')))
        return class_ece.mean()


class CalibrationCurve(AggregatingKpi):
    """
    Calibration Curve: Shows the difference between true positive rate
    and the average predicted confidence over n_bins.
    Uses static binning.

    :param n_bins: number of bins
    :param threshold: Positive class threshold for ground truth
    :param for_neg_class: If True, cc is plotted for the negative class instead of the positive
    """

    def __init__(self, n_bins: int = 10, threshold: float = 0.5,
                 for_neg_class: bool = False):
        super().__init__(f'CC_cls{for_neg_class}@{threshold}'.replace(".", ""))
        self.threshold = threshold
        """Threshold for the positive class for the ground truth"""
        self.for_neg_class = for_neg_class
        """Defined for which class the curve should be computed"""

        self.prob_sum: torch.Tensor = torch.zeros(n_bins)
        """Buffer collecting the sum of probabilities for each bin."""
        self.tp_sum: torch.Tensor = torch.zeros(n_bins)
        """Buffer collecting the true positives for each bin."""
        self.count: torch.Tensor = torch.zeros(n_bins)
        """Total amount of predictions per bin."""

        self.register_buffer(self.name + "_prob_sum", self.prob_sum,
                             persistent=False)
        self.register_buffer(self.name + "_tp_sum", self.tp_sum,
                             persistent=False)
        self.register_buffer(self.name + "_count", self.count, persistent=False)

        linspace = torch.linspace(-1e-6, 1, len(self.count) + 1)
        self.low_bound = linspace[0:-1]
        """Lower bound per bin."""
        self.high_bound = linspace[1:]
        """Upper bound per bin."""
        self.register_buffer(self.name + "_low_bound", self.low_bound,
                             persistent=False)
        self.register_buffer(self.name + "_high_bound", self.high_bound,
                             persistent=False)

    def update(self, outputs: torch.Tensor,
               labels: torch.Tensor):
        # Assuming binary class prediction. Hence, the negative class results are the inverse of the positive class
        if self.for_neg_class:
            outputs = 1.0 - outputs
            labels = 1.0 - labels
        for bin_nr in range(len(self.count)):
            relevant = (outputs > self.low_bound[bin_nr]) & (
                    outputs <= self.high_bound[bin_nr])
            # Sum probabilities per bin
            self.prob_sum[bin_nr] = self.prob_sum[bin_nr] + outputs[
                relevant].float().sum()
            # Sum true positives per bin
            self.tp_sum[bin_nr] = self.tp_sum[bin_nr] + (
                    labels[relevant] > self.threshold).float().sum()
            self.count[bin_nr] = self.count[bin_nr] + relevant.float().sum()

    def reset(self):
        torch.nn.init.constant_(self.prob_sum, 0)
        torch.nn.init.constant_(self.tp_sum, 0)
        torch.nn.init.constant_(self.count, 0)

    @staticmethod
    def _plot_curve(conf: np.ndarray, tpr: np.ndarray,
                    rel_count: np.ndarray
                    ) -> plt.Figure:
        """
        Create a :py:class:`Figure` object, showing the calibration curve
        as well as the relative occurrence count within each bin.
        All input arguments have to be ordered according the the confidence

        :param conf: Array of confidences
        :param tpr: Array of true positive rates (e.g. true negative rates for the negative class)
        :param rel_count: Array of the relative number of samples in each bin
        :return: The plot
        """
        my_dpi = 150
        fig = plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
        ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        ax.plot(conf, tpr)
        ax.plot(conf, rel_count)
        for c, t, r in zip(conf, tpr, rel_count):
            ax.annotate(f'e:{np.abs(c - t):.2f}', xy=(c, t), textcoords='data',
                        fontsize=7)
            ax.annotate(f'c:{r:.2E}', xy=(c, r), textcoords='data', rotation=90,
                        fontsize=7)
        plt.legend(["tpr", "count"])
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        ax.set_xlabel('Confidence')
        ax.set_ylabel('tpr/count')
        ax.set_aspect('equal', adjustable='box')
        plt.grid(True)
        return fig

    def value(self) -> plt.Figure:
        tpr = self.tp_sum / self.count
        conf = self.prob_sum / self.count
        rel_count = self.count / torch.sum(self.count)

        return self._plot_curve(conf.detach().cpu().numpy(),
                                tpr.detach().cpu().numpy(),
                                rel_count.detach().cpu().numpy())


class PrecisionRecallCurve(AggregatingKpi):
    """
    Approximated precision / recall curve.
    The approximation lies in the selection of a fixed set of thresholds to be
    evaluated, as opposed to forming the convex hull over the precision/recall curve.

    :param num_steps: Number of thresholds to evaluate. Equally spaced over [0,1].
    """

    def __init__(self, num_steps: int = 20):
        super().__init__('PRCurve')
        self.estimates = [ConfusionMatrix(threshold) for threshold in
                          np.linspace(0.0, 1.0, num_steps)]

    def reset(self):
        for estimate in self.estimates:
            estimate.reset()

    def update(self, outputs: torch.Tensor,
               labels: torch.Tensor):
        for estimate in self.estimates:
            estimate.update(outputs, labels)

    @staticmethod
    def _plot_curve(precisions: np.ndarray, recalls: np.ndarray,
                    thresholds: np.ndarray
                    ) -> plt.Figure:  # pylint: disable=invalid-name
        """
        Create a :py:class:`Figure` object, showing the precision recall curve.

        :param precisions: Array of precision values
        :param recalls: Array of recall values
        :param thresholds: Array of thresholds, use to obtained the precision and recall values
        :return: The plot
        """
        my_dpi = 150
        fig = plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
        ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        ax.plot(precisions, recalls)
        for p, r, t in zip(precisions, recalls, thresholds):
            ax.annotate(f'{t:.2f}', xy=(p, r), textcoords='data', fontsize=7)
        plt.xlim([0, 1.0])
        plt.ylim([0, 1.0])
        ax.set_xlabel('Precision')
        ax.set_ylabel('Recall')
        plt.grid(True)
        return fig

    def value(self) -> plt.Figure:
        precisions = []
        recalls = []
        thresholds = []
        for estimate in self.estimates:
            precision = estimate.matrix[0, 0] / (
                    estimate.matrix[0, 0] + estimate.matrix[0, 1])
            recall = estimate.matrix[0, 0] / (
                    estimate.matrix[0, 0] + estimate.matrix[1, 1])
            precisions.append(precision.item())
            recalls.append(recall.item())
            thresholds.append(estimate.threshold)

        nans = np.logical_or(np.isnan(precisions), np.isnan(recalls))
        precisions = np.asarray(precisions)[np.logical_not(nans)]
        recalls = np.asarray(recalls)[np.logical_not(nans)]
        thresholds = np.asarray(thresholds)[np.logical_not(nans)]
        return self._plot_curve(precisions, recalls, thresholds)


class SetIoU(AggregatingKpi):
    r"""Calc set intersection over union (IoU) value for a batch of outputs.
    The set intersection over union is a special reduction of the
    intersection over union for batch tensors. Given a dataset :math:`B` it
    calculates as

    .. math::
        \frac{\sum_B intersection} {\sum_B union}
        =  \frac{\sum_B TP} {\sum_B TP + FP + FN}

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

    def __init__(self, output_thresh: float = 0.5, label_thresh: float = 0.,
                 smooth: float = 1e-6):
        """Init.

        :param output_thresh: threshold for binarizing the output
        :param label_thresh: threshold for binarizing the labels
        :param smooth: summand to smooth the IoU value (evade division by 0)
        """
        super().__init__(f'SetIoU@{output_thresh}')

        self.output_thresh = output_thresh
        self.label_thresh = label_thresh
        self.smooth = smooth

        self.intersection_sum: torch.Tensor = torch.tensor(0)
        self.union_sum: torch.Tensor = torch.tensor(0)

    def reset(self):
        torch.nn.init.constant_(self.intersection_sum, 0)
        torch.nn.init.constant_(self.union_sum, 0)

    def smooth_division(self, dividend, divisor):
        """Smoothed division using smoothening summand to avoid division by 0.

        :return: result of smooth division."""
        return (dividend + self.smooth) / (divisor + self.smooth)

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        """Smooth set intersection over union between binarized in- and output.

        :param outputs: Output tensors of shape ``(BATCH x 1 x H x W)``
        :param labels: Label tensors of shape ``(BATCH x H x W)``
        """
        # Binarize and turn into integer tensors:
        labels = (labels > self.label_thresh).bool()
        outputs = (outputs > self.output_thresh).bool()

        # Calculate and sum each intersections and unions
        # zero if Truth=0 or Prediction=0
        intersection_sum = (outputs & labels).float().sum()
        union_sum = (outputs | labels).float().sum()  # zero if both are 0

        self.intersection_sum = self.intersection_sum + intersection_sum
        self.union_sum = self.union_sum + union_sum

    def value(self) -> torch.Tensor:
        # Smoothed division to avoid division by 0:
        set_iou_tensor = self.smooth_division(self.intersection_sum, self.union_sum)
        return set_iou_tensor


class SetIoUThresholdCurve(AggregatingKpi):
    """
    Create a curve showing the setIoU value for multiple thresholds.

    :param num_steps: Number of thresholds to consider. Equally spaced over [0,1]
    """

    def __init__(self, num_steps: int = 20):
        super().__init__('SetIoUTSCurve')
        self.estimates = [SetIoU(threshold) for threshold in
                          np.linspace(0.0, 1.0, num_steps)]

    def reset(self):
        for estimate in self.estimates:
            estimate.reset()

    def update(self, outputs: torch.Tensor,
               labels: torch.Tensor):
        for estimate in self.estimates:
            estimate.update(outputs, labels)

    @staticmethod
    def _plot_curve(thresholds: Union[np.ndarray, List[float]],
                    setiou: Union[np.ndarray, List[float]]
                    ) -> plt.Figure:  # pylint: disable=invalid-name
        """
        Create a :py:class:`Figure` object, showing setIoU/threshold curve

        :param thresholds: Array of thresholds, use to obtained the setIoU values
        :param setiou: Array of setIoU values.
        :return: The plot
        """
        my_dpi = 150
        fig = plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
        ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        ax.plot(thresholds, setiou)

        maxidx = np.argmax(setiou)
        t = thresholds[maxidx]
        s = setiou[maxidx]
        ax.annotate(f'{s:.2f}@{t:.2f}', xy=(t, s), textcoords='data',
                    fontsize=7)
        plt.xlim([0, 1.0])
        plt.ylim([0, np.max(setiou)])
        ax.set_xlabel('Threshold')
        ax.set_ylabel('setIoU')
        plt.grid(True)
        return fig

    def value(self) -> plt.Figure:
        thresholds = []
        results = []
        for estimate in self.estimates:
            thresholds.append(estimate.output_thresh)
            results.append(estimate.value().detach().cpu().numpy())

        return self._plot_curve(thresholds, results)


class _OnPredictions(AggregatingKpi, abc.ABC):
    """Base class for aggregating KPIs that ignore the labels."""

    def forward(self, outputs: torch.Tensor,
                _labels: torch.Tensor = None
                ):
        """Calculate KPI without gradient.
        Uses :py:meth:`~AggregatingKpi.value`."""
        with torch.no_grad():
            self.update(outputs, _labels)

    def update(self, outputs: torch.Tensor, _labels: torch.Tensor = None):
        """Update according to ``outputs``."""
        raise NotImplementedError


class Mean(_OnPredictions):
    """Calculate the mean of all output tensor entries (ignoring labels)."""

    def __init__(self, name: str = None):
        super().__init__(name=name or self.__class__.__name__)
        self.sum: float = 0
        self.count: int = 0

    def value(self) -> Optional[float]:
        """Return the mean up to the outputs obtained so far."""
        return (self.sum / self.count) if self.count != 0 else None

    def update(self, outputs: torch.Tensor, _labels: torch.Tensor = None):
        """Update according to ``outputs``."""
        self.sum += outputs.sum().item()
        self.count += outputs.numel()

    def reset(self):
        """Reset the intermediate storage."""
        self.sum, self.count = 0, 0


class Variance(Mean):
    r"""Calculate the variance of all output tensor entries (ignoring labels).
    Use the formula :math:`Var(X) = E[(X-E[X])^2] = E[X^2] - E[X]^2` with
    Bessel bias correction (see :py:attr:`ddof`):

    .. math::
        Var_{\text{unbiased}}(X) = \frac{N}{N-ddof} Var(X)
        = \frac{N}{N-ddof} \left(
        \frac{1}{N}\sum_{i=1}^{N} x_i^2  -  \left(\frac{1}{N}\sum_{i=1}^{N} x_i\right)^2
        \right)

    """

    def __init__(self, ddof: int = 1, name: str = None):
        super().__init__(name=name or self.__class__.__name__)
        self.squared_sum: float = 0
        self.ddof: int = ddof
        r"""The degrees of freedom for the Bessel correction.
        The :math:`\frac{N}{N-ddof}` is applied to the variance before return."""

    def value(self) -> Optional[float]:
        """Return the variance up to the outputs obtained so far."""
        if self.count == 0 or self.count <= self.ddof:
            return None
        mean = super().value()
        squared_mean = (self.squared_sum / self.count)
        bessel_corr = self.count / (self.count - self.ddof)
        return bessel_corr * (squared_mean - (mean ** 2))

    def update(self, outputs: torch.Tensor, _labels: torch.Tensor = None):
        """Update according to ``outputs``."""
        super().update(outputs)
        self.squared_sum += ((outputs ** 2).sum().item())

    def reset(self):
        """Reset the intermediate storage."""
        super().reset()
        self.squared_sum = 0


class StandardDev(Variance):
    r"""Calculate the standard deviation of all output tensor entries (ignoring labels).
    Use the formula :math:`StdDev(X) = \sqrt{Var(X)}`, see :py:class:`Variance`."""

    def value(self):
        """Return the standard deviation up to the outputs obtained so far."""
        variance = super().value()
        return sqrt(variance) if variance is not None else None


class Extremum(_OnPredictions):
    """Find the extreme value of all output tensor entries over all batches
    (ignoring labels). This can be min or max."""

    def __init__(self, name: str = None):
        super().__init__(name=name or self.__class__.__name__)
        self.extremum: Optional[float] = None

    def value(self) -> Optional[float]:
        """Return the mean up to the outputs obtained so far."""
        return self.extremum

    def reset(self):
        """Reset the intermediate storage."""
        self.extremum = None

    def update(self, outputs: torch.Tensor, _labels: torch.Tensor = None):
        """Check whether a new extremum is found in this ``outputs`` batch."""
        raise NotImplementedError


class Maximum(Extremum):
    """Find the max value within all batches."""

    def update(self, outputs: torch.Tensor, _labels: torch.Tensor = None):
        """Check whether a new maximum is found in this ``outputs`` batch."""
        outputs_max: float = outputs.max().item()
        if self.extremum is None or self.extremum < outputs_max:
            self.extremum = outputs_max


class Minimum(Extremum):
    """Find the max value within all batches."""

    def update(self, outputs: torch.Tensor, _labels: torch.Tensor = None):
        """Check whether a new minimum is found in this ``outputs`` batch."""
        outputs_min: float = outputs.min().item()
        if self.extremum is None or self.extremum > outputs_min:
            self.extremum = outputs_min


class Histogram(_OnPredictions):
    """Collect a histogram of the model output values."""

    def __init__(self, n_bins: int = 100, lower_bound: float = 0, upper_bound: float = 1.,
                 device: Union[str, torch.device] = None):
        """Init.

        :param n_bins: the number of bins to use
        :param lower_bound: the expected lower bound of the output values
        :param upper_bound: the expected upper bound of the output values
        :param device: the device onto which to put the count tensor
        """
        super().__init__(name=self.__class__.__name__)
        if not n_bins >= 0:
            raise ValueError("n_bins must be integer greater zero but was {}".format(n_bins))
        self.n_bins: int = n_bins
        """The number of bins to use."""
        self.count: torch.Tensor = torch.zeros(self.n_bins, device=device)
        """Total amount of predictions per bin."""
        self.register_buffer(self.name + "_count", self.count, persistent=False)
        self._bounds = np.linspace(lower_bound - 1e-6, upper_bound, self.n_bins + 1)
        """The boundaries of the bins as a sequence of values."""

    def update(self, outputs: torch.Tensor,
               _labels: torch.Tensor = None):
        """Update the bin count."""
        for bin_nr in range(self.n_bins):
            relevant = ((outputs > self._bounds[bin_nr]) &
                        (outputs <= self._bounds[bin_nr + 1]))
            self.count[bin_nr] += relevant.float().sum().item()

    def reset(self):
        """Reset the bin count."""
        self.count = torch.zeros(self.n_bins)

    def value(self) -> plt.Figure:
        """Plot the histogram."""
        my_dpi = 150
        fig: plt.Figure = plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
        ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        ax.hist(self._bounds[:-1], bins=self._bounds, weights=self.count.detach().cpu().numpy())
        ax.set_ylabel('# pixels')
        ax.set_xlabel('pixel value ranges')
        plt.grid(True, axis='y')
        return fig


class ApproximateMean(Histogram):
    """Approximate the mean in the data by binning the values.
    During the updates, the number of samples per bin are counted.
    For value calculation, all samples within one bin are assumed to
    have the mean bin value, i.e. the mean of the bin's lower and upper bound.
    For details on the binning see super class.

    .. note::
        :py:class:`ApproximateMean` is to be preferred over :py:class:`Mean`
        in case a buffer overflow is expected if all items in the data are
        summed up.
        By binning, the samples may get distributed into several bins,
        reducing the maximum value of the sums to calculate per bin.

    .. note::
        Choose larger number of bins (``n_bins`` argument) to get a better approximation.
    """

    @property
    def bin_centers(self) -> torch.Tensor:
        """1D tensor holding the values of the center points of each bin."""
        bin_centers_np: np.ndarray = self._bounds[:-1] + 0.5 / self.n_bins
        bin_centers: torch.Tensor = torch.from_numpy(bin_centers_np).to(self.count.device)
        return bin_centers

    def value(self) -> torch.Tensor:
        """Return the approximate variance according to the binning so far."""
        tot_sum: torch.Tensor = (self.count * self.bin_centers).sum()
        tot_count: torch.Tensor = self.count.sum()
        approx_mean: torch.Tensor = tot_sum / tot_count if tot_count != 0 else None
        return approx_mean


class ApproximateVariance(ApproximateMean):
    """Approximate the variance in the data by binning the values.
    Approximation works the same as in the super class.
    See there for details.
    """

    def __init__(self, ddof: int = 1, **binning_args):
        super().__init__(**binning_args)
        self.ddof: int = ddof
        """The delta degrees of freedom to use for divisors.
        The divisor is ``N-ddof``, cf. the pandas implementation."""

    def value(self) -> Optional[torch.Tensor]:
        """Return the approximate variance according to the binning so far."""
        tot_count: torch.Tensor = self.count.sum()
        if tot_count.item() <= self.ddof:
            return None
        approx_mean: torch.Tensor = super().value()
        if approx_mean is None:
            return None
        squared_diffs: torch.Tensor = ((approx_mean - self.bin_centers) ** 2 * self.count)
        approx_var: torch.Tensor = squared_diffs.sum() / (tot_count - self.ddof)
        return approx_var


class ApproximateStdDev(ApproximateVariance):
    """Approximate the standard deviation in the data by binning the values.
    Approximation works the same as in the super class.
    See there for details.
    """

    def value(self) -> Optional[torch.Tensor]:
        """Return the approximate standard deviation according to the binning so far."""
        approx_var = super().value()
        approx_std: torch.Tensor = torch.sqrt(approx_var) if approx_var is not None else None
        return approx_std
