import numpy as np
import cv2
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2

# load images
image_paths = [
    "datasets/14/camera1_light1.png",
    "datasets/14/camera1_light2.png",
    "datasets/14/camera1_light3.png",
    "datasets/14/camera1_light4.png",
]
images = []
slicing = True
x1, y1 = 500, 700
x2, y2 = 2000, 2200
for image_path in image_paths:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if slicing:
        image = image[y1:y2, x1:x2]
    images.append(image)

# to np.array
images = np.stack(images, axis=-1)

# light source directions
S = np.array(
    [[0.58, 0.58, 0.58], [-0.58, 0.58, 0.58], [-0.58, -0.58, 0.58], [0.58, -0.58, 0.58]]
)

height, width, num_images = images.shape
# I.shape (num_images, height, width)
I = images.reshape(-1, num_images).T
# computing the normal vector matrix by Least Square Method
N = np.linalg.lstsq(S, I, rcond=None)[0]
# normalize the normal vector matrix
N = N / np.linalg.norm(N, axis=0)
N = N.T.reshape(height, width, 3)

# generate depth map by integration method
zx = N[:, :, 0] / N[:, :, 2]  # dz/dx
zy = N[:, :, 1] / N[:, :, 2]  # dz/dy
# integration approach
# depth_map = integrate.cumulative_trapezoid(
#     zy, axis=0, initial=0
# ) + integrate.cumulative_trapezoid(zx, axis=1, initial=0)
# poisson approach
f = np.zeros((height, width))
f[1:-1, 1:-1] = zx[1:-1, :-2] - zx[1:-1, 2:] + zy[:-2, 1:-1] - zy[2:, 1:-1]
fx = np.fft.fftfreq(width).reshape(1, width)
fy = np.fft.fftfreq(height).reshape(height, 1)
denom = (2 * np.cos(2 * np.pi * fx) - 2) + (2 * np.cos(2 * np.pi * fy) - 2)
denom[0, 0] = 1  # 避免除零
depth_map = ifft2(fft2(f) / denom).real

# visualize depth map
plt.figure(figsize=(8, 6))
plt.imshow(depth_map, cmap="gray")
plt.colorbar()
plt.title("Depth Map from Photometric Stereo")
plt.show()
