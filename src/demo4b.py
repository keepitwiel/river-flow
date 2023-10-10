from random import random
import numpy as np
from numba import njit
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import colors
from height_map import generate
from scipy.sparse import dok_matrix


@njit
def update(i, j, z, h, r, n_iters):
    # idx = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    mh, mw = z.shape
    r0 = r / n_iters
    H0 = h[j, i] + z[j, i]
    for k in range(n_iters):
        H_left = h[j, i - 1] + z[j, i - 1]
        H_right = h[j, i + 1] + z[j, i + 1]
        H_up = h[j - 1, i] + z[j - 1, i]
        H_down = h[j + 1, i] + z[j + 1, i]

        h[j, i] += r0

        left = max(H0 - H_left, 0) if i - 1 > 0 else 0
        right = max(H0 - H_right, 0) if i + 1 < mw else 0
        up = max(H0 - H_up, 0) if j - 1 > 0 else 0
        down = max(H0 - H_down, 0) if j + 1 < mh else 0
        total = left + right + up + down
        if total > 0:
            p = total * random()  # np.random.choice(4, size=1, p=p)[0]
            if p < left:
                i -= 1
            elif left <= p < left + right:
                i += 1
            elif right <= p < left + right + up:
                j -= 1
            elif up <= p:
                j += 1

            if j < 1 or j > mh-1 or i < 1 or i > mw-1:
                break


RAINFALL = 0.1


z = generate(
    scale=8,
    random=0.3,
    smoothness=0.5,
    slope=np.array([[True, False], [True, True]])
)
mh, mw = z.shape
h = np.zeros_like(z)


fig, axes = plt.subplots(1, 2, figsize=(8, 4))

for count in tqdm(range(1000000)):
    j = 128 # np.random.randint(1, mh-2)
    i = 128 # np.random.randint(1, mw-2)
    update(i, j, z, h, r=RAINFALL, n_iters=1000)
    if count % 10000 == 0:
        axes[0].imshow(z, cmap="terrain")
        axes[0].set_title(f"Terrain altitude, Total terrain: {np.sum(z):2.2f}")
        axes[1].imshow(h, vmax=100*RAINFALL)
        axes[1].set_title(f"Iteration: {count}, Total water: {np.sum(h):2.2f}")
        plt.pause(0.001)
        h *= 0.9
