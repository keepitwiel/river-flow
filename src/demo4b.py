from time import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from height_map import generate

from fast_solver import update


RAINFALL = 0.01
SEA_LEVEL = 64

z = generate(
    scale=10,
    random=0.01,
    smoothness=0.5,
    random_seed=116,
    slope=np.array([[True, True], [False, True]])
)
z = z[:-2, :-2]
z -= SEA_LEVEL

mh, mw = z.shape
h = np.zeros_like(z)
n_iters = 10000

update(z, h, r=RAINFALL, n_iters=1, length=mw*2)
t0 = time()
update(z, h, r=RAINFALL, n_iters=n_iters-1, length=mw*2)
t1 = time()
print(f"{(n_iters-1) / (t1 - t0):2.0f} iterations per second")
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(z, cmap="terrain")
axes[0].set_title(f"Total terrain: {np.sum(z):2.2f}")
axes[1].imshow(h, vmax=RAINFALL)
axes[1].set_title(f"Total water: {np.sum(h):2.2f}")
# plt.pause(0.001)
plt.show()