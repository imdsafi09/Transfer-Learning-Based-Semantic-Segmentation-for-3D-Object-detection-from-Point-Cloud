import sys

import cv2


import helpers
from helpers import *
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np


def masked_loss(y_true, y_pred):
    """Defines a masked loss that ignores border/unlabeled pixels (represented as -1).
    
    Args:
      y_true: Ground truth tensor of shape [B, H, W, 1].
      y_pred: Prediction tensor of shape [B, H, W, N_CLASSES].
    """
    gt_validity_mask = tf.cast(tf.greater_equal(y_true[:, :, :, 0], 0), dtype=tf.float32) # [B, H, W]
    
    # The sparse categorical crossentropy loss expects labels >= 0. 
    # We just transform -1 into any valid class label, it will then be masked anyways.
    y_true = K.abs(y_true)
    raw_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)  # [B, H, W].

    masked = gt_validity_mask * raw_loss
    return tf.reduce_mean(masked)

