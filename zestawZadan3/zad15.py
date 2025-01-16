import cv2
import numpy as np
from skimage.exposure import equalize_hist
import matplotlib.pyplot as plt

# Wczytaj obrazy: torus, dark frame, flat frame
torus = cv2.imread("zdjecia/zdjeciaPakiet3/torus.png", cv2.IMREAD_GRAYSCALE).astype(
    np.float32
)
dark_frame = cv2.imread(
    "zdjecia/zdjeciaPakiet3/Dark_Frame.png", cv2.IMREAD_GRAYSCALE
).astype(np.float32)
flat_field = cv2.imread(
    "zdjecia/zdjeciaPakiet3/Flat_Frame.png", cv2.IMREAD_GRAYSCALE
).astype(np.float32)

# (a) Korekcja Flat-Field
corrected = (torus - dark_frame) / (flat_field - dark_frame)
corrected = np.clip(corrected, 0, 1)  # Normalizacja do zakresu [0, 1]

# (b) Transformacja histogramu
equalized = equalize_hist(corrected)

# Zapis wyników
cv2.imwrite("flat_field_corrected.png", (corrected * 255).astype(np.uint8))
cv2.imwrite("flat_field_histogram_equalized.png", (equalized * 255).astype(np.uint8))

# Wizualizacja wyników
plt.figure(figsize=(10, 5))
titles = ["Flat-Field Corrected", "Histogram Equalized"]
images = [corrected, equalized]

for i in range(len(images)):
    plt.subplot(1, 2, i + 1)
    plt.title(titles[i])
    plt.imshow(images[i], cmap="gray")
    plt.axis("off")

plt.tight_layout()
plt.show()
