import cv2
import numpy as np


def calculate_global_contrast(image):
    return np.std(image)


def calculate_local_contrast(image):
    # Padding image to handle border pixels
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode="reflect")
    h, w = image.shape
    local_contrast = np.zeros((h, w), dtype=float)

    # Define 8-connected neighbors relative positions
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # Compute local contrast
    for i in range(1, h + 1):
        for j in range(1, w + 1):
            pixel = padded_image[i, j]
            contrast_sum = 0
            for dy, dx in neighbors:
                neighbor_pixel = padded_image[i + dy, j + dx]
                contrast_sum += abs(int(pixel) - int(neighbor_pixel))
            local_contrast[i - 1, j - 1] = contrast_sum / 8.0

    # Average local contrast for the image
    return np.mean(local_contrast)


images = [
    "zdjecia/zdjeciaPakiet2/muchaA.png",
    "zdjecia/zdjeciaPakiet2/muchaB.png",
    "zdjecia/zdjeciaPakiet2/muchaC.png",
]

for img_name in images:
    image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    global_contrast = calculate_global_contrast(image)
    local_contrast = calculate_local_contrast(image)

    print(
        f"{img_name} - Global Contrast: {global_contrast}, Local Contrast: {local_contrast}"
    )
