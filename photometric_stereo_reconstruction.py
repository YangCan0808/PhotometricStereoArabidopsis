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
x1, y1 = 1000, 1500
x2, y2 = 1400, 1900
lower_green = np.array([35, 50, 50])
upper_green = np.array([120, 255, 255])
for image_path in image_paths:
    image = cv2.imread(image_path)[y1:y2, x1:x2]

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    extracted_plant = cv2.bitwise_and(image, image, mask=mask)

    gray_plant = cv2.cvtColor(extracted_plant, cv2.COLOR_BGR2GRAY)

    images.append(gray_plant)

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
zero_vectors = np.all(N == 0, axis=0)
for i in range(N.shape[1]):
    if zero_vectors[i]:
        N[:, i] = [0, 0, 1]
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
denom[0, 0] = 1
depth_map = ifft2(fft2(f) / denom).real

# visualize depth map
plt.figure(figsize=(8, 6))
plt.imshow(depth_map, cmap="gray")
plt.colorbar()
plt.title("Depth Map from Photometric Stereo")
plt.show()
