import numpy as np
from numba import njit
from scipy.sparse import dok_array


def foo():
    d = dok_array((100, 100))
    i = np.random.randint(100)
    j = np.random.randint(100)
    d[j, i] = 1
    print(d)
    print(list(d.keys()))


if __name__ == "__main__":
    foo()
