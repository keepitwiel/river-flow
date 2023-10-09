import numpy as np
from numba import njit


@njit
def update_water(z, h, dh, r, dt, evap):
    height, width = z.shape
    for j in range(1, height-1):
        for i in range(1, width-1):
            h_ = h[j, i]
            if h_ > 0:
                h[j, i] += r[j, i] * dt
                H_ = h_ + z[j, i]
                H_left = h[j, i-1] + z[j, i-1]
                H_right = h[j, i+1] + z[j, i+1]
                H_up = h[j-1, i] + z[j-1, i]
                H_down = h[j-1, i] + z[j+1, i]

                left  = max(H_ - H_left, 0) if i - 1 > 0 else 0
                right = max(H_ - H_right, 0) if i + 1 < width else 0
                up    = max(H_ - H_up, 0) if j - 1 > 0 else 0
                down  = max(H_ - H_down, 0) if j + 1 < width else 0
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
    if evap > 0:
        h -= np.minimum(h, evap * dt)
