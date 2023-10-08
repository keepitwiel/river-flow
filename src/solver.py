import numpy as np
from numba import njit


@njit
def update_water(H, z, h, dh, r, dt, evap):
    dh += r * dt
    height, width = z.shape
    H[:, :] = z[:, :] + h[:, :]
    for j in range(1, height-1):
        for i in range(1, width-1):
            h_ = h[j, i]
            if h_ > 0:
                H_ = H[j, i]
                left  = max(H_ - H[j, i-1], 0) if i - 1 > 0 else 0
                right = max(H_ - H[j, i+1], 0) if i + 1 < width else 0
                up    = max(H_ - H[j-1, i], 0) if j - 1 > 0 else 0
                down  = max(H_ - H[j+1, i], 0) if j + 1 < width else 0
                total = left + right + up + down
                if total > 0:
                    delta = h_ / total * dt
                    dh[j, i] -= total * delta
                    dh[j, i-1] += left * delta
                    dh[j, i+1] += right * delta
                    dh[j-1, i] += up * delta
                    dh[j+1, i] += down * delta
    h += dh
    dh.fill(0.0)
    h -= np.minimum(h, evap * dt)
