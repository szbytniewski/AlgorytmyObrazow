import cv2
import numpy as np


def floyd_steinberg_dithering(image, palette):
    # Konwersja obrazu na paletę szarości
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Kopia obrazu do ditheringu
    dithered_image = np.copy(gray_image).astype(float)

    # Pętla przez każdy piksel (z pominięciem brzegów)
    for y in range(1, gray_image.shape[0] - 1):
        for x in range(1, gray_image.shape[1] - 1):
            old_pixel = dithered_image[y, x]
            new_pixel = find_closest_palette_color(old_pixel, palette)
            dithered_image[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            # Propagowanie błędu na sąsiadujące piksele
            dithered_image[y, x + 1] += quant_error * 7 / 16
            dithered_image[y + 1, x - 1] += quant_error * 3 / 16
            dithered_image[y + 1, x] += quant_error * 5 / 16
            dithered_image[y + 1, x + 1] += quant_error * 1 / 16

    # Zwracamy obraz, przekształcony z float na uint8
    return np.clip(dithered_image, 0, 255).astype(np.uint8)


def find_closest_palette_color(value, palette):
    # Znalezienie najbliższego koloru w palecie
    return min(palette, key=lambda x: abs(x - value))


# Wczytanie obrazu
image = cv2.imread("../zdjecia/zdjeciaPakiet1/ds3.jpg")

# (a) Redukcja palety szarości do 1-bitowej (czarno-białej) z progiem 39
threshold = 39
binary_palette = [0, 255]
binary_dithered = floyd_steinberg_dithering(image, binary_palette)

# Zapis obrazu wynikowego
cv2.imwrite("ds3.png", binary_dithered)

# (b) Redukcja palety szarości do 5 wartości {0, 64, 128, 192, 255}
palette = [0, 64, 128, 192, 255]
dithered_5_colors = floyd_steinberg_dithering(image, palette)

# Zapis obrazu wynikowego
cv2.imwrite("ds3_dithered_5_colors.png", dithered_5_colors)
