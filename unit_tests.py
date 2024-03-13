import pytest
import test_fixtures

from random_radon_transform import random_radon_transformation

import numpy as np

_MARGIN_OF_ERROR = 0.05

def get_mean_error(arr1, arr2):
    deltas = np.abs(arr1 - arr2)
    err_sum = np.sum(deltas.flatten())

    return err_sum / np.prod(arr1.shape)