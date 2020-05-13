import functools
import operator


def get_chunks(iterable, chunksize):
    for i in range(0, len(iterable), chunksize):
        yield iterable[i:i + chunksize]


def flat(iterable):
    return functools.reduce(operator.iconcat, iterable, [])
