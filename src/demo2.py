import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import colors
from height_map import generate

from solver import update_water


SEA_LEVEL = 32
RAINFALL = 0.1
EVAP = 0.00001
EROSION = 0.1


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
H = np.zeros_like(z)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].set_title("Terrain altitude")
axes[2].set_title("Terrain + water surface altitude")
# x, y = np.meshgrid(np.)

for count in tqdm(range(100000)):
    update_water(H, z, h, dh, r, dt=0.1, evap=EVAP)
    if count % 1000 == 0:
        axes[0].imshow(z, cmap="terrain")
        axes[1].imshow(h, vmax=RAINFALL)
        axes[1].set_title(f"Iteration: {count}, Total water: {np.sum(h):2.2f}")
        axes[2].imshow(H, cmap="terrain")
        plt.pause(0.001)
    count += 1
