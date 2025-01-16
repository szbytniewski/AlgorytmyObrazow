import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from skimage.exposure import adjust_gamma
import matplotlib.pyplot as plt

# Wczytaj obraz ptaki.png
image = cv2.imread("zdjecia/zdjeciaPakiet3/ptaki.png", cv2.IMREAD_GRAYSCALE)

# (a) Okno sinusoidalne
rows, cols = image.shape
sin_window = np.outer(
    np.sin(np.linspace(0, np.pi, rows)), np.sin(np.linspace(0, np.pi, cols))
)
windowed_image = image * sin_window

# (b) Filtr uśredniający
smoothed_image = uniform_filter(windowed_image, size=3)

# (c) Korekta gamma
gamma_corrected = adjust_gamma(smoothed_image, gamma=1.2)

# (d) Wygładzanie bezpośrednie
direct_smoothed = uniform_filter(image, size=3)

# Zapis wyników
cv2.imwrite("okienkowanie_sinusoidalne.png", (windowed_image * 255).astype(np.uint8))
cv2.imwrite("filtr_usredniajacy.png", (smoothed_image * 255).astype(np.uint8))
cv2.imwrite("korekta_gamma.png", (gamma_corrected * 255).astype(np.uint8))
cv2.imwrite("bezposrednie_usrednianie.png", (direct_smoothed * 255).astype(np.uint8))

# Wizualizacja wyników
plt.figure(figsize=(12, 8))
titles = [
    "Oryginalny obraz",
    "Okienkowanie sinusoidalne",
    "Filtr uśredniający",
    "Korekta gamma",
    "Bezpośrednie wygładzanie",
]
images = [image, windowed_image, smoothed_image, gamma_corrected, direct_smoothed]

for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    plt.title(titles[i])
    plt.imshow(images[i], cmap="gray")
    plt.axis("off")

plt.tight_layout()
plt.show()
