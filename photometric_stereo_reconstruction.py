import numpy as np
import cv2
from scipy import integrate
import matplotlib.pyplot as plt

# load images
image_paths = [
    "datasets/14/camera1_light1.png",
    "datasets/14/camera1_light2.png",
    "datasets/14/camera1_light3.png",
    "datasets/14/camera1_light4.png",
]
images = []
for image_path in image_paths:
    images.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))

# to np.array
images = np.stack(images, axis=-1)

# light source directions
S = np.array(
    [[0.58, 0.58, 0.58], [-0.58, 0.58, 0.58], [-0.58, -0.58, 0.58], [0.58, -0.58, 0.58]]
)

# computing the normal vector matrix
height, width, num_images = images.shape
I = images.reshape(-1, num_images).T
N = np.linalg.lstsq(S, I, rcond=None)[0]
N = N / np.linalg.norm(N, axis=0)
N = N.T.reshape(height, width, 3)

# generate depth map
zx = N[:, :, 0] / N[:, :, 2]  # dz/dx
zy = N[:, :, 1] / N[:, :, 2]  # dz/dy
depth_map = integrate.cumtrapz(zy, axis=0, initial=0) + integrate.cumtrapz(
    zx, axis=1, initial=0
)

# visualize depth map
plt.figure(figsize=(8, 6))
plt.imshow(depth_map, cmap="gray")
plt.colorbar()
plt.title("Depth Map from Photometric Stereo")
plt.show()
