from time import time
from random import randint
import numpy as np
from numba import njit
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import colors
from height_map import generate
from scipy.sparse import dok_matrix


@njit
def update(z, s, h, r, n_iters, length):
    n = len(z)
    r0 = r / length
    for iteration in range(n_iters):
        j = randint(0, 100)
        H0 = h[j] + s[j] + z[j]
        for k in range(length):
            nj = 0
            for dj in [-1, 1]:
                H1 = h[j + dj] + s[j + dj] + z[j + dj]
                if H1 < H0:
                    nj = dj
                    H0 = H1
            h[j] += r0

            if nj != 0:
                dz = r0 * np.exp(-(h[j] + s[j]))
                z[j] -= dz
                j += nj
                s[j] += dz
            if j < 1 or j > n-1 or z[j] + s[j] < 0:
                break
        ds = np.zeros_like(s)
        for k in range(n):
            h[k] -= min(h[k], r0)
            # if z[k] + s[k] < 0:
            #     h[k] = -(z[k] + s[k])
        for k in range(1, n-1):
            ds[k] -= 0.02 * s[k]
            ds[k-1] += 0.01 * s[k]
            ds[k+1] += 0.01 * s[k]
        for k in range(n):
            s[k] += ds[k]
        s[0] = 0
        s[n-1] = 0



np.random.seed(42)
dz = np.random.normal(size=101)
z = np.cumsum(dz) + 15
h = np.zeros_like(z)
s = np.zeros_like(z)
r = 0.1

i = 0
while True:
    if i % 100 == 0:
        plt.title(f"{i}, {np.sum(h):2.0f}, {np.sum(z + s):2.0f}")
        plt.plot(z, c="black")
        plt.plot(z + s, c="grey")
        plt.plot(z + s + h, c="blue")
        plt.plot(s, c="grey")
        plt.ylim(-10, 25)
        update(z, s, h, r, n_iters=1000, length=len(z))
        plt.pause(0.001)
        plt.cla()
    i += 1
# plt.show()
