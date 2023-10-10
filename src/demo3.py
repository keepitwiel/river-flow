import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import colors
from height_map import generate
from scipy.sparse import dok_matrix


def update(z, h, r, active, dt):
    height, width = z.shape
    dh = dok_matrix(z.shape)
    for (j, i) in active.keys():
        dh[j, i] += r[j, i] * dt
        h_ = h[j, i]
        if h_ > 0:
            H_ = h_ + z[j, i]
            H_left = h[j, i - 1] + z[j, i - 1]
            H_right = h[j, i + 1] + z[j, i + 1]
            H_up = h[j - 1, i] + z[j - 1, i]
            H_down = h[j + 1, i] + z[j + 1, i]

            left = max(H_ - H_left, 0) if i - 1 > 0 else 0
            right = max(H_ - H_right, 0) if i + 1 < width else 0
            up = max(H_ - H_up, 0) if j - 1 > 0 else 0
            down = max(H_ - H_down, 0) if j + 1 < height else 0

            total = left + right + up + down
            if total > 0:
                delta = h_ / total * dt
                dh[j, i] -= total * delta
                dh[j, i - 1] += left * delta
                dh[j, i + 1] += right * delta
                dh[j - 1, i] += up * delta
                dh[j + 1, i] += down * delta
    h += dh

    for (j, i) in dh.keys():
        if dh[j, i] != 0 or r[j, i] > 0:
            active[j, i] = True
        else:
            active[j, i] = False

    return h, active


RAINFALL = 0.1


z = generate(
    scale=7,
    random=0.2,
    smoothness=0.5,
    slope=np.array([[True, False], [True, True]])
)
mh, mw = z.shape
r = np.zeros_like(z)
j, i = (3 * mh) // 4, mh // 4
r[j, i] = RAINFALL
h = dok_matrix((mh, mw))
active = dok_matrix(r > 0)


fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].set_title("Terrain altitude")

for count in tqdm(range(100000)):
    h, active = update(z, h, r, active, dt=0.1)
    if count % 100 == 0:
        axes[0].imshow(z, cmap="terrain")
        axes[1].imshow(h.todense(), vmax=RAINFALL)
        axes[1].set_title(f"Iteration: {count}, nodes: {active.size}, Total water: {np.sum(h):2.2f}")
        plt.pause(0.001)
    # count += 1
