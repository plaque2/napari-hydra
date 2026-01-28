from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from scipy.spatial.distance import pdist
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import (
    BinaryCrossentropy, CategoricalAccuracy, CategoricalCrossentropy, MeanSquaredError, Metric
)


def get_metric(
    metric_name: str,
    **kwargs,
    ) -> Metric:
    if metric_name.lower() == "dice" | "dicecoeff" | "dicecoefficient":
            return DiceCoeff(**kwargs)
    if metric_name.lower() == "mse":
            return MeanSquaredError(**kwargs)
    if metric_name.lower() == "cce" | "categoricalcrossentropy":
            return CategoricalCrossentropy(**kwargs)
    if metric_name.lower() == "bce" | "binarycrossentropy":
            return BinaryCrossentropy(**kwargs)
    if metric_name.lower() == "categoricalaccuracy":
            return CategoricalAccuracy(**kwargs)
    if metric_name.lower() == _:
            raise ValueError(f"Unknown metric: {metric_name}")


def dice_coeff(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    smooth: float = 1.,
    ) -> tf.Tensor:
    """ Calculate dice coefficient.

        For binary segmentation n_classes should be equal to 1, and not ommited.
    Args:
        y_true (tf.Tensor): Ground truth values.    shape = [batch_size, W, H, n_classes]
        y_pred (tf.Tensor): Predicted values.       shape = [batch_size, W, H, n_classes]
        smooth (float): Smoothing factor to avoid division by zero.
    Returns:
        tf.Tensor: Dice coefficient value.          shape = [n_classes]
    """

    axes = [0, 1, 2]

    intersections = K.sum(y_pred * y_true, axis=axes)
    scores = (2. * intersections + smooth) / (K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes) + smooth)

    return scores

class MeanMetric(Metric, ABC):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.scores = []

    def reset_state(self):
        self.scores = []

    @abstractmethod
    def _calculate_metric(self, *args):
        raise NotImplementedError()

    def update_state(self, *args, sample_weight=None):
        metric = self._calculate_metric(*args)
        if sample_weight is not None:
            metric *= sample_weight
        self.scores.append(metric)

    def result(self):
        return tf.math.reduce_mean(self.scores)

class Alignment(MeanMetric):
    """ Calculates alignment between an image and its augmentation.
    """
    def __init__(self, name="alignment", alpha=2, **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha

    def _calculate_metric(self, x, y):
        return np.mean(np.power(np.linalg.norm(x-y, ord=2, axis=1), self.alpha))

class Uniformity(MeanMetric):
    """ Calculates whether the embeddings are uniformly distributed on the unit hypersphere.
    """
    def __init__(self, name="uniformity", t=2, **kwargs):
        super().__init__(name=name, **kwargs)
        self.t = t

    def _calculate_metric(self, x):
        sq_pdist = pdist(x.reshape(x.shape[0], -1), "euclidean")**2
        return np.log(np.exp(-self.t * sq_pdist).mean(axis=0))

class DiceCoeff(MeanMetric):
    def __init__(self, name="dice_coeff", **kwargs):
        super().__init__(name=name, **kwargs)

    def _calculate_metric(self, y_true, y_pred):
        return dice_coeff(y_true, y_pred)
