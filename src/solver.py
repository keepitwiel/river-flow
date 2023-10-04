import numpy as np
from numba import njit


def update(z, dz, h, dh, flux, flux_out, r, evap, erosion, dt):
    update_water(z, h, dh, flux, r, dt)
    diffuse_terrain(z, dz, h, flux, erosion, dt)
    # evaporate(h, flux, evap, dt)
    flux_out[:, :] = flux[:, :]
    flux.fill(0.0)


@njit
def update_water(z, h, dh, flux, r, dt):
    dh += r
    height, width = z.shape
    for j in range(1, height-1):
        for i in range(1, width-1):
            h_ = h[j, i]
            z_ = z[j, i]

            if h_ > 0:
                left = z[j, i-1] + h[j, i-1] < z_ + h_ if i - 1 > 0 else False
                right = z[j, i+1] + h[j, i+1] < z_ + h_ if i + 1 < width else False
                up = z[j-1, i] + h[j-1, i] < z_ + h_ if j - 1 > 0 else False
                down = z[j+1, i] + h[j+1, i] < z_ + h_ if j + 1 < width else False

                total = left + right + up + down

                if total > 0:
                    delta = h_ / total * dt
                    dh[j, i] -= h_ * dt
                    dh[j, i-1] += left * delta
                    dh[j, i+1] += right * delta
                    dh[j-1, i] += up * delta
                    dh[j+1, i] += down * delta

                    flux[j, i] += h_ * dt
                    flux[j, i-1] += left * delta
                    flux[j, i+1] += right * delta
                    flux[j-1, i] += up * delta
                    flux[j+1, i] += down * delta

    h += dh
    dh.fill(0.0)


@njit
def diffuse_terrain(z, dz, h, flux, erosion, dt):
    height, width = z.shape
    for j in range(1, height-1):
        for i in range(1, width-1):
            delta = (flux[j, i]) / (h[j, i]**2 + 1) * erosion * dt
            dz[j, i] -= delta * 4
            dz[j, i - 1] += delta
            dz[j, i + 1] += delta
            dz[j - 1, i] += delta
            dz[j + 1, i] += delta
    z += dz
    dz.fill(0.0)


@njit
def evaporate(h, flux, evap, dt):
    delta = np.minimum(evap * dt, h / (flux + 1))
    h -= np.minimum(h, delta)
