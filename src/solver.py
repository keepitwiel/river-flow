import numpy as np
from numba import njit

DEPTH_LIMIT = 10.0

@njit
def update_water(z, h, dh, r, dt):
    height, width = z.shape
    for j in range(1, height-1):
        for i in range(1, width-1):
            h_ = h[j, i]
            if (h_ > 0 or r[j, i] > 0) and z[j, i] > 0:
                h[j, i] += r[j, i] * dt
                H_ = h_ + z[j, i]
                H_left = h[j, i-1] + z[j, i-1]
                H_right = h[j, i+1] + z[j, i+1]
                H_up = h[j-1, i] + z[j-1, i]
                H_down = h[j-1, i] + z[j+1, i]

                left  = max(H_ - H_left, 0) if i - 1 > 0 else 0
                right = max(H_ - H_right, 0) if i + 1 < width else 0
                up    = max(H_ - H_up, 0) if j - 1 > 0 else 0
                down  = max(H_ - H_down, 0) if j + 1 < height else 0
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

@njit
def update_terrain(z, h, dz, erosion, dt):
    height, width = z.shape
    for j in range(1, height-1):
        for i in range(1, width-1):
            h_ = h[j, i] + h[j, i - 1] + h[j, i + 1] + h[j - 1, i] + h[j + 1, i]
            if h_ > 0:
                z_ = z[j, i]
                left  = max(z_ - z[j, i-1], 0) if i - 1 > 0 else 0
                right = max(z_ - z[j, i+1], 0) if i + 1 < width else 0
                up    = max(z_ - z[j-1, i], 0) if j - 1 > 0 else 0
                down  = max(z_ - z[j+1, i], 0) if j + 1 < height else 0

                total = left + right + up + down
                if total > 0:
                    h_ = max(0, 0.1 - (h_-0.1)**2)
                    delta = h_ * erosion * dt
                    dz[j, i] -= total * delta
                    dz[j, i-1] += left * delta
                    dz[j, i+1] += right * delta
                    dz[j-1, i] += up * delta
                    dz[j+1, i] += down * delta
    z += dz
    dz.fill(0.0)
