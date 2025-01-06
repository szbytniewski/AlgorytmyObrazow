import os

import cv2
import numpy as np

image = cv2.imread("zdjecia/zdjeciaPakiet3/roze.png", cv2.IMREAD_GRAYSCALE)

# 1. Global Thresholding (Otsu)
_, global_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite("global_otsu.png", global_thresh)


# 2. Iterative Three-Class Thresholding with Otsu
def iterative_three_class_thresholding(img, delta_threshold=2):
    hist, bin_edges = np.histogram(img.ravel(), bins=256, range=(0, 256))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Initial threshold using Otsu's method
    t1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    t2 = t1 / 2  # Initial guess for t2
    delta = delta_threshold + 1

    while delta > delta_threshold:
        # Calculate means for the two regions
        low_region = bin_centers[bin_centers < t1]
        high_region = bin_centers[bin_centers >= t1]
        low_mean = np.mean(low_region) if len(low_region) > 0 else 0
        high_mean = np.mean(high_region) if len(high_region) > 0 else 0

        # Update t2 and delta
        new_t2 = (low_mean + high_mean) / 2
        delta = abs(new_t2 - t2)
        t2 = new_t2

    # Apply thresholds
    binary_image = np.where(img > t2, 255, 0).astype(np.uint8)
    return binary_image


three_class_thresh = iterative_three_class_thresholding(image)
cv2.imwrite("iterative_three_class_otsu.png", three_class_thresh)


# 3. Local Thresholding with Otsu in 11x11 Neighborhood
def local_otsu_thresholding(img, window_size=11):
    # Pad the image symmetrically
    half_window = window_size // 2
    padded_img = np.pad(img, pad_width=half_window, mode="symmetric")
    output = np.zeros_like(img)

    for i in range(half_window, padded_img.shape[0] - half_window):
        for j in range(half_window, padded_img.shape[1] - half_window):
            # Extract local window
            window = padded_img[
                i - half_window : i + half_window + 1,
                j - half_window : j + half_window + 1,
            ]
            # Compute Otsu threshold for the window
            _, local_thresh = cv2.threshold(
                window, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            # Assign the central value of the thresholded window to the output
            output[i - half_window, j - half_window] = local_thresh[
                half_window, half_window
            ]

    return output


local_thresh = local_otsu_thresholding(image)
cv2.imwrite("local_otsu.png", local_thresh)
