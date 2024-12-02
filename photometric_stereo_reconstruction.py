import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import depth_methods
from pathlib import Path


def load_images(image_paths, roi_coordinates):
    x1, y1 = roi_coordinates["x1"], roi_coordinates["y1"]
    x2, y2 = roi_coordinates["x2"], roi_coordinates["y2"]
    images = []
    for image_path in image_paths:
        full_path = Path(image_path)
        if full_path.suffix == ".png":
            image = cv2.imread(image_path)
            if x1 != None and y1 != None and x2 != None and y2 != None:
                image = image[y1:y2, x1:x2]
            images.append(image)
        elif full_path.suffix == ".txt":
            directory = full_path.parent
            with open(full_path, "r") as file:
                lines = file.readlines()
                for line in lines:
                    line = line.rstrip("\n")
                    image = cv2.imread(directory / line)
                    if x1 != None and y1 != None and x2 != None and y2 != None:
                        image = image[y1:y2, x1:x2]
                    images.append(image)
    return images


def hsv_segment(hsv_method_params):
    image_path = Path(hsv_method_params["image_path"])
    image = cv2.imread(image_path)
    lower = np.array([hsv_method_params["lower"]])
    upper = np.array([hsv_method_params["upper"]])

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def generate_mask(config_data):
    if "mask_path" in config_data:
        maskImagePath = Path(config_data["mask_path"])
        img_mask = cv2.imread(maskImagePath)
        gray_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_mask, 128, 255, cv2.THRESH_BINARY)
        return binary_mask
    elif "hsv_method" in config_data:
        return hsv_segment(config_data["hsv_method"])
    return None


def calculate_normal_vector_matrix():
    pass


def main():
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
    images = load_images(config["image_paths"], config["roi_coordinates"])
    segmented_images = hsv_segment(images, config["hsv_thresholds"])

    grayscale_images = []
    for segmented_image in segmented_images:
        grayscale_images.append(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY))

    # to np.array
    images = np.stack(grayscale_images, axis=-1)

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


if __name__ == "__main__":
    main()
