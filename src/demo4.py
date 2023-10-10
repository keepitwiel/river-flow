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
def update(z, h, r, n_iters, length):
    idx = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    mh, mw = z.shape
    r0 = r / length
    for iteration in range(n_iters):
        j = randint(0, mh-1)
        i = randint(0, mw-1)
        H0 = h[j, i] + z[j, i]
        for k in range(length):
            nj, ni = (0, 0)
            for dj, di in idx:
                H1 = h[j + dj, i + di] + z[j + dj, i + di]
                if H1 < H0:
                    nj, ni = dj, di
                    H0 = H1
            h[j, i] += r0
            j, i = j + nj, i + ni
            if j < 1 or j > mh-1 or i < 1 or i > mw-1 or z[j, i] < 0:
                break



RAINFALL = 0.001
SEA_LEVEL = 64

z = generate(
    scale=8,
    random=0.02,
    smoothness=0.5,
    random_seed=116,
    slope=np.array([[True, False], [True, True]])
)
z -= SEA_LEVEL
z0 = z.copy()

mh, mw = z.shape
h = np.zeros_like(z)
n_iters = 10000

update(z, h, r=RAINFALL, n_iters=1, length=mw*2)
t0 = time()
update(z, h, r=RAINFALL, n_iters=n_iters-1, length=mw*2)
t1 = time()
print(f"{(n_iters-1) / (t1 - t0):2.0f} iterations per second")
fig, axes = plt.subplots(1, 3, figsize=(8, 4))
axes[0].imshow(z, cmap="terrain")
axes[0].set_title(f"Total terrain: {np.sum(z):2.2f}")
axes[1].imshow(h)
axes[1].set_title(f"Total water: {np.sum(h):2.2f}")
axes[2].imshow(z - z0)
axes[2].set_title(f"Terrain erosion: {np.sum(np.abs(z - z0)):2.2f}")
# plt.pause(0.001)
plt.show()
