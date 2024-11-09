import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import depth_methods

with open("config.json", "r") as config_file:
    config = json.load(config_file)

# load images
image_paths = config["image_paths"]
roi_coordinates = config["roi_coordinates"]
x1, y1 = roi_coordinates["x1"], roi_coordinates["y1"]
x2, y2 = roi_coordinates["x2"], roi_coordinates["y2"]
hsv_threshold = config["hsv_thresholds"]
lower_green = np.array([hsv_threshold["lower_green"]])
upper_green = np.array([hsv_threshold["upper_green"]])

images = []
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
S = np.array(config["light_source_directions"])

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

# Calculate depth map
DEPTH_METHODS = {
    "poisson": depth_methods.calculate_depth_poisson,
    "integration": depth_methods.calculate_depth_integration,
}
depth_method_key = config["depth_method"]
depth_method = DEPTH_METHODS.get(depth_method_key)
depth_map = depth_method(N)

# visualize depth map
plt.figure(figsize=(8, 6))
plt.imshow(depth_map, cmap="gray")
plt.colorbar()
plt.title("Depth Map from Photometric Stereo")
plt.show()
