import cv2
import numpy as np


def bayer_dithering(image, bayer_matrix, palette):
    # Konwersja obrazu na skalę szarości
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray_image.shape

    # Powiększenie macierzy Bayera, aby dopasować rozmiar obrazu
    bayer_matrix = np.tile(bayer_matrix, (height // 8 + 1, width // 8 + 1))
    bayer_matrix = bayer_matrix[:height, :width]

    # Normalizacja macierzy Bayera (dla poziomów 0-255)
    bayer_matrix_normalized = (bayer_matrix / np.max(bayer_matrix)) * 255

    # Zastosowanie ditheringu z Bayer matrix
    dithered_image = np.where(
        gray_image > bayer_matrix_normalized, palette[1], palette[0]
    )

    return dithered_image


def bayer_dithering_4_levels(image, bayer_matrix):
    # Konwersja obrazu na skalę szarości
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray_image.shape

    # Powiększenie macierzy Bayera, aby dopasować rozmiar obrazu
    bayer_matrix = np.tile(bayer_matrix, (height // 8 + 1, width // 8 + 1))
    bayer_matrix = bayer_matrix[:height, :width]

    # Normalizacja macierzy Bayera (dla poziomów 0-255)
    bayer_matrix_normalized = (bayer_matrix / np.max(bayer_matrix)) * 255

    # Poziomy szarości
    levels = np.array([50, 100, 150, 200])

    # Zastosowanie ditheringu z Bayer matrix
    dithered_image = (
        np.digitize(gray_image + bayer_matrix_normalized - 128, bins=[64, 128, 192])
        * 50
        + 50
    )

    return dithered_image


# Macierz Bayera 8x8
bayer_matrix_8x8 = np.array(
    [
        [0, 48, 12, 60, 3, 51, 15, 63],
        [32, 16, 44, 28, 35, 19, 47, 31],
        [8, 56, 4, 52, 11, 59, 7, 55],
        [40, 24, 36, 20, 43, 27, 39, 23],
        [2, 50, 14, 62, 1, 49, 13, 61],
        [34, 18, 46, 30, 33, 17, 45, 29],
        [10, 58, 6, 54, 9, 57, 5, 53],
        [42, 26, 38, 22, 41, 25, 37, 21],
    ]
)

# Wczytanie obrazu
image = cv2.imread("../zdjecia/zdjeciaPakiet1/stanczyk.png")

# (a) Dithering z Bayer matrix (1-bitowa paleta)
binary_palette = [0, 255]
dithered_image_bayer_binary = bayer_dithering(image, bayer_matrix_8x8, binary_palette)

# Zapis wyniku
cv2.imwrite("stanczyk_bayer_dithered_binary.png", dithered_image_bayer_binary)

# (b) Dithering z Bayer matrix (paleta {50, 100, 150, 200})
dithered_image_bayer_4_levels = bayer_dithering_4_levels(image, bayer_matrix_8x8)

# Zapis wyniku
cv2.imwrite("stanczyk_bayer_dithered_4_levels.png", dithered_image_bayer_4_levels)
