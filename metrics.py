import math


def euclidean_distance(x, y):
    return math.sqrt(sum(map(lambda t: (t[0] - t[1])**2, zip(x, y))))


def manhattan_distance(x, y):
    return sum(map(lambda t: abs(t[0] - t[1]), zip(x, y)))


def get_metric(mectic):
    metric_map = {
        'euclidean': euclidean_distance,
        'manhattan': manhattan_distance,
    }
    return metric_map.get(mectic)
