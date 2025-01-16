import matplotlib.pyplot as plt
import cv2
from skimage.io import imread
from skimage.transform import resize

# Wczytaj oryginalny obraz
image = imread("zdjecia/zdjeciaPakiet2/PlytkaFresnela.png", as_gray=True)

# Redukcja rozdzielczości do 30x30
downsampled = resize(image, (30, 30), anti_aliasing=False)

# Rekonstrukcja obrazu (powiększenie do 512x512)
reconstructed = resize(downsampled, image.shape, anti_aliasing=False)

# Wizualizacja
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Oryginalny obraz")
plt.imshow(image, cmap="gray")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("Próbkowany obraz (30x30)")
plt.imshow(downsampled, cmap="gray")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title("Zrekonstruowany obraz")
plt.imshow(reconstructed, cmap="gray")
plt.colorbar()

plt.show()

cv2.imwrite("zestawZadan2/plytka30.png", downsampled)
