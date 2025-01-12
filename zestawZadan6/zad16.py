import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, morphology, filters

# Załaduj obraz
image = io.imread("zdjecia/zdjeciaPakiet6/StoLat.png", as_gray=True)

# Przekształć obraz na obraz binarny (czarno-biały)
binary_image = (image > 0.5).astype(np.uint8)

selem = np.ones((3, 3))
filtered_image = morphology.opening(binary_image, selem)

# Wykrycie poziomych linii taktu - dylatacja z poziomym elementem strukturalnym
horizontal_selem = np.ones((1, 30))  # Długi poziomy element strukturalny
dilated_image = morphology.dilation(filtered_image, horizontal_selem)

# Wykrycie poszczególnych taków - erozja, aby oddzielić linie taktu
eroded_image = morphology.erosion(dilated_image, horizontal_selem)

# Zliczanie liczby taktów (po wykryciu linii poziomych)
# Zliczamy liczbę wykrytych linii poziomych (odpowiadających taktom)
tacts = np.sum(eroded_image)  # Liczymy liczbę pikseli w obrazie erozji
tacts = int(
    tacts / 15
)  # Dzielenie przez 15, ponieważ 15 pikseli to długość linii poziomej

# Wyświetlanie wyników
fig, ax = plt.subplots(1, 5, figsize=(15, 6))
ax[0].imshow(binary_image, cmap="gray")
ax[0].set_title("Obraz binarny")

ax[1].imshow(filtered_image, cmap="gray")
ax[1].set_title("Po filtracji szumów (Otwarcie)")

ax[2].imshow(dilated_image, cmap="gray")
ax[2].set_title("Po dylatacji (linie poziome)")

ax[3].imshow(eroded_image, cmap="gray")
ax[3].set_title("Po erozji (detekcja taktów)")

ax[4].imshow(eroded_image, cmap="gray")
ax[4].set_title(f"Liczba taktów: {tacts}")

plt.show()
