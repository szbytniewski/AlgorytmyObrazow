import cv2
import numpy as np
import matplotlib.pyplot as plt

image = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
    ],
    dtype=np.uint8,
)

structuring_elements = [
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8),  # (a)
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=np.uint8),  # (b)
    np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=np.uint8),  # (c)
    np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.uint8),  # (d)
]

results = []
for structuring_element in structuring_elements:
    hit_miss = cv2.morphologyEx(image, cv2.MORPH_HITMISS, structuring_element)
    results.append(hit_miss)

fig, axs = plt.subplots(1, 5, figsize=(24, 4))
axs[0].imshow(image, cmap="gray")
axs[0].set_title("Original Image")
axs[0].axis("off")

titles = ["(a) Result", "(b) Result", "(c) Result", "(d) Result"]
for i, result in enumerate(results):
    axs[i + 1].imshow(result, cmap="gray")
    axs[i + 1].set_title(titles[i])
    axs[i + 1].axis("off")

plt.tight_layout()
plt.show()
