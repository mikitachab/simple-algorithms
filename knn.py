import os
import concurrent.futures
import contextlib
import functools
import time

import numpy as np
from more_itertools import flatten, chunked

import metrics

class KNN:

    def __init__(self, n_neighbors=3, metric='euclidean', parallel=False, chunksize=5, n_jobs=-1):
        self.n_neighbors = n_neighbors
        self.parallel = parallel
        self.n_jobs = n_jobs if n_jobs != -1 else os.cpu_count()
        self.chunksize = chunksize
        self.metric = metrics.get_metric(metric)
        if not self.metric:
            raise ValueError(f'Unkown metric: {self.metric}')

    def _get_distances(self, sample, train):
        return list(map(lambda x: self.metric(x, sample), train))

    def _get_distances_parallel(self, sample):
        chunks = chunked(self.train, self.chunksize)
        distances_func = functools.partial(self._get_distances, sample)
        with concurrent.futures.ProcessPoolExecutor(self.n_jobs) as executor:
            return flatten(executor.map(distances_func, chunks))

    def _get_neighbors(self, sample):
        if not self.parallel:
            distances = self._get_distances(sample, self.train)
        else:
            distances = self._get_distances_parallel(sample)

        distances = zip(distances, self.target)
        return list(map(lambda x: x[1], sorted(distances)[:self.n_neighbors]))

    def fit(self, train, target):
        self.train = train
        self.target = target

    def predict(self, sample):
        neighbors = self._get_neighbors(sample)
        return max(set(neighbors), key=neighbors.count)


@contextlib.contextmanager
def timer():
    try:
        start = time.perf_counter()
        yield
    finally:
        end = time.perf_counter()
        print('time:', end - start)


def get_result():
    dataset = np.array([[2.7810836, 2.550537003, 1],
                        [1.465489372, 2.362125076, 1],
                        [3.396561688, 4.400293529, 1],
                        [1.38807019, 1.850220317, 1],
                        [3.06407232, 3.005305973, 1],
                        [7.627531214, 2.759262235, 2],
                        [5.332441248, 2.088626775, 2],
                        [6.922596716, 1.77106367, 2],
                        [8.675418651, -0.242068655, 2],
                        [7.673756466, 3.508563011, 2]])

    x = np.array([[p[0], p[1]] for p in dataset])
    y = np.array([p[2] for p in dataset])

    knn = KNN(n_neighbors=3, parallel=True, metric='manhattan')
    knn.fit(x, y)

    return knn.predict([1.0, 2.0])


with timer():
    print(get_result())
