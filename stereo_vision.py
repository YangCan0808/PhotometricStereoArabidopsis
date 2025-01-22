import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_disparity_map(left_image, right_image):
    left_image = cv2.equalizeHist(left_image)
    right_image = cv2.equalizeHist(right_image)
    left_image = cv2.GaussianBlur(left_image, (5, 5), 0)
    right_image = cv2.GaussianBlur(right_image, (5, 5), 0)

    min_disparity = 20
    num_disparities = 16 * 18
    block_size = 5
    p1 = 8 * 3 * block_size ** 2
    p2 = 32 * 3 * block_size ** 2
    disp12_max_diff = 5
    uniqueness_ratio = 5
    speckle_window_size = 200
    speckle_range = 2

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=disp12_max_diff,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range
    )

    disparity = stereo.compute(left_image, right_image).astype(np.float32)

    return disparity


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-l", "--left", type=str)
    parser.add_argument("-r", "--right", type=str)

    return parser.parse_args()

def main():
    args = parse_args()

    left_image = cv2.imread(args.left, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(args.right, cv2.IMREAD_GRAYSCALE)

    disparity = compute_disparity_map(left_image, right_image)

    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_normalized = np.uint8(disparity_normalized)

    # plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Left Image")
    plt.imshow(left_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Disparity Map (StereoSGBM)")
    plt.imshow(disparity_normalized, cmap='jet')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
