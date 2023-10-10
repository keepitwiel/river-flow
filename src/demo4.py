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
        j = randint(1, mh-2)
        i = randint(1, mw-2)
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
            if j < 0 or j > mh or i < 0 or i > mw:
                break
    # h -= np.minimum(h, r0)


RAINFALL = 0.0001


z = generate(
    scale=8,
    random=0.01,
    smoothness=0.5,
    random_seed=116,
    slope=np.array([[True, False], [True, True]])
)
mh, mw = z.shape
h = np.zeros_like(z)


fig, axes = plt.subplots(1, 2, figsize=(8, 4))

for count in tqdm(range(10000)):
    update(z, h, r=RAINFALL, n_iters=100, length=mw)
    if count % 100 == 0:
        axes[0].imshow(z, cmap="terrain")
        axes[0].set_title(f"Terrain altitude, Total terrain: {np.sum(z):2.2f}")
        axes[1].imshow(h, vmax=RAINFALL*100)
        axes[1].set_title(f"Iteration: {count}, Total water: {np.sum(h):2.2f}")
        plt.pause(0.001)
