from skimage import io, color, morphology, img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt


def zhang_suen_thinning_iteration(image, iteration):
    """
    Perform one sub-iteration of the Zhang-Suen thinning algorithm.
    """
    marker = np.zeros_like(image, dtype=bool)
    rows, cols = image.shape
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if image[r, c]:
                # Extract 3x3 neighborhood
                P = [
                    image[r - 1, c],  # P2
                    image[r - 1, c + 1],  # P3
                    image[r, c + 1],  # P4
                    image[r + 1, c + 1],  # P5
                    image[r + 1, c],  # P6
                    image[r + 1, c - 1],  # P7
                    image[r, c - 1],  # P8
                    image[r - 1, c - 1],  # P9
                ]

                transitions = sum((P[i] == 0 and P[(i + 1) % 8]) for i in range(8))

                neighbors = sum(P)

                if (
                    2 <= neighbors <= 6
                    and transitions == 1
                    and (
                        (P[0] * P[2] * P[4]) if iteration == 0 else (P[2] * P[4] * P[6])
                    )
                    == 0
                    and (
                        (P[2] * P[4] * P[6]) if iteration == 0 else (P[0] * P[2] * P[6])
                    )
                    == 0
                ):
                    marker[r, c] = True
    return marker


def zhang_suen_thinning(image):
    """
    Apply the Zhang-Suen thinning algorithm to an image.
    """
    thinned_image = image.copy()
    while True:
        marker1 = zhang_suen_thinning_iteration(thinned_image, 0)
        thinned_image[marker1] = False
        marker2 = zhang_suen_thinning_iteration(thinned_image, 1)
        thinned_image[marker2] = False
        if not (marker1.any() or marker2.any()):
            break
    return thinned_image


image_path = "zdjecia/zdjeciaPakiet6/mikolaj.png"
image = io.imread(image_path)


image_rgb = image[:, :, :3]
binary_image = color.rgb2gray(image_rgb) < 0.5

marker1 = zhang_suen_thinning_iteration(binary_image, 0)
binary_after_sub_iteration = binary_image.copy()
binary_after_sub_iteration[marker1] = False

# Step 2: First iteration
marker2 = zhang_suen_thinning_iteration(binary_after_sub_iteration, 1)
binary_after_iteration = binary_after_sub_iteration.copy()
binary_after_iteration[marker2] = False

skeletonized_image = zhang_suen_thinning(binary_image)

# Visualize the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Binary Image")
plt.imshow(binary_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Removed Pixels: 1st Sub-Iteration")
plt.imshow(marker1, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Fully Skeletonized Image")
plt.imshow(skeletonized_image, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
