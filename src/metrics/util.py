import numpy as np


def np_one_hot(array, num_classes=None):
    if num_classes is None:
        num_classes = array.max() + 1

    hot_vector = np.zeros((array.size, num_classes))
    hot_vector[np.arange(array.size), array] = 1
    return hot_vector
