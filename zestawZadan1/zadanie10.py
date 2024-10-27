import cv2
import numpy as np


def variable_threshold_dithering(image, dither_matrix):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray_image.shape

    dither_matrix = np.tile(dither_matrix, (height // 4 + 1, width // 4 + 1))
    dither_matrix = dither_matrix[:height, :width]

    dither_matrix_normalized = (dither_matrix / np.max(dither_matrix)) * 255

    dithered_image = np.where(gray_image > dither_matrix_normalized, 255, 0)

    return dithered_image


# Macierz dla grupy 4x4
dither_matrix_4x4 = np.array(
    [[6, 14, 2, 8], [4, 0, 10, 11], [12, 15, 5, 1], [9, 3, 13, 7]]
)

# Wczytanie obrazu
image = cv2.imread("../zdjecia/zdjeciaPakiet1/stanczyk.png")

# Wykonanie ditheringu zmiennego progu
dithered_image_variable_threshold = variable_threshold_dithering(
    image, dither_matrix_4x4
)

# Zapis wyniku
cv2.imwrite(
    "stanczyk_variable_threshold_dithered.png", dithered_image_variable_threshold
)
