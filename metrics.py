import numpy as np


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def manhattan_distance(x, y):
    return np.sum(np.absolute(x - y))


def get_metric(mectic):
    metric_map = {
        'euclidean': euclidean_distance,
        'manhattan': manhattan_distance,
    }
    return metric_map.get(mectic)
