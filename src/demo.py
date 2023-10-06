import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import colors
from height_map import generate_height_map

from solver import update_water


SEED = 116
MAP_WIDTH = 128
MAP_HEIGHT = 128
SEA_LEVEL = 16
RANDOM_AMPLITUDE = 16
SLOPE_AMPLITUDE = 64
SMOOTHNESS = 0.6
RAINFALL = 0.1
EVAP = 0.0001


def main():
    z = generate_height_map(MAP_HEIGHT, MAP_WIDTH, SEED, SMOOTHNESS) * RANDOM_AMPLITUDE
    z += np.linspace(0, 1, MAP_HEIGHT).reshape(-1, 1).dot(np.ones(MAP_WIDTH).reshape(1, -1)) * SLOPE_AMPLITUDE
    z -= SEA_LEVEL
    r = np.zeros_like(z)
    r[MAP_HEIGHT - 8, MAP_WIDTH // 2 - 8] = RAINFALL
    h = -np.minimum(0.0, z)
    dh = np.zeros_like(z)
    H = np.zeros_like(z)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].set_title("Terrain altitude")
    cmap = colors.ListedColormap(["blue", "cyan", "green", "yellow", "orange", "red"])
    bounds = [-128, 0, 16, 32, 48, 64, 128]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    axes[2].set_title("Terrain + water surface altitude")

    for count in tqdm(range(100000)):
        update_water(H, z, h, dh, r, dt=0.1, evap=EVAP)
        if count % 1000 == 0:
            axes[0].imshow(z, cmap=cmap, norm=norm)
            axes[1].imshow(h, vmax=RAINFALL)
            axes[1].set_title(f"Iteration: {count}, Total water: {np.sum(h):2.2f}")
            axes[2].imshow(H, vmin=-1, vmax=SLOPE_AMPLITUDE + RANDOM_AMPLITUDE)
            plt.pause(0.001)
        count += 1


if __name__ == "__main__":
    main()
