from random import randint

from numba import njit


@njit
def update(z, h, r, n_iters, length):
    idx = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    mh, mw = z.shape
    r0 = r / length
    for iteration in range(n_iters):
        j = randint(0, mh-1)
        i = randint(0, mw-1)
        H0 = h[j, i] + z[j, i]
        for k in range(length):
            nj, ni = (0, 0)
            for dj, di in idx:
                H1 = h[j + dj, i + di] + z[j + dj, i + di]
                if H1 < H0:
                    nj, ni = dj, di
                    H0 = H1
            h[j, i] += r0
            j, i = j + nj, i + ni
            if j < 1 or j > mh-1 or i < 1 or i > mw-1 or z[j, i] < 0:
                break
