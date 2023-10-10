import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import colors
from height_map import generate

from pseudo_solver import update_water, update_terrain


SEA_LEVEL = 64
RAINFALL = 0.1


z = generate(
    scale=7,
    random=0.2,
    smoothness=0.5,
    slope=np.array([[True, False], [True, True]])
)
mh, mw = z.shape
z -= SEA_LEVEL
dz = np.zeros_like(z)
r = np.zeros_like(z)
r[(3 * mh) // 4, mh // 4] = RAINFALL
h = -np.minimum(0.0, z)
dh = np.zeros_like(z)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

for count in tqdm(range(100000)):
    if count % 1000 == 0:
        axes[0].imshow(z, cmap="terrain")
        axes[0].set_title(f"Terrain altitude: {np.min(z):2.0f}-{np.max(z):2.0f}")
        axes[1].imshow(h, vmax=RAINFALL)
        axes[1].set_title(f"Iteration: {count}, Total water: {np.sum(h):2.2f}")
        plt.pause(0.001)
    update_water(z, h, dh, r, dt=0.1)
    # update_terrain(z, h, dz, erosion=0.01, dt=0.1)
