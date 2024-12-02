import numpy as np
from scipy import fft, integrate


def calculate_depth_poisson(N):
    height, width, _ = N.shape
    zx = N[:, :, 0] / N[:, :, 2]
    zy = N[:, :, 1] / N[:, :, 2]
    f = np.zeros((height, width))
    f[1:-1, 1:-1] = zx[1:-1, :-2] - zx[1:-1, 2:] + zy[:-2, 1:-1] - zy[2:, 1:-1]
    fx = np.fft.fftfreq(width).reshape(1, width)
    fy = np.fft.fftfreq(height).reshape(height, 1)
    denom = (2 * np.cos(2 * np.pi * fx) - 2) + (2 * np.cos(2 * np.pi * fy) - 2)
    denom[0, 0] = 1
    depth_map = fft.ifft2(fft.fft2(f) / denom).real
    return depth_map


def calculate_depth_integration(N):
    zx = -N[:, :, 0] / N[:, :, 2]
    zy = N[:, :, 1] / N[:, :, 2]
    depth_map = integrate.cumulative_trapezoid(
        zy, axis=0, initial=0
    ) + integrate.cumulative_trapezoid(zx, axis=1, initial=0)
    return depth_map
