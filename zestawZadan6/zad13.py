import cv2
import numpy as np


def hit_or_miss(input_image):
    # Definiowanie elementu strukturalnego (SE) do znalezienia lewego górnego rogu
    structuring_element = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.uint8)

    # Oddzielenie SE na część "foreground" i "background"
    fg_se = structuring_element == 1  # Foreground
    bg_se = structuring_element == 0  # Background

    eroded_fg = cv2.erode(input_image, fg_se.astype(np.uint8))

    input_complement = cv2.bitwise_not(input_image)

    eroded_bg = cv2.erode(input_complement, bg_se.astype(np.uint8))

    hit_or_miss_result = cv2.bitwise_and(eroded_fg, eroded_bg)

    return hit_or_miss_result


input_image = cv2.imread("zdjecia/zdjeciaPakiet6/zad13.png", cv2.IMREAD_GRAYSCALE)

_, binary_image_cv = cv2.threshold(input_image, 127, 255, cv2.THRESH_BINARY_INV)


output_image = hit_or_miss(binary_image_cv)

binary_image = cv2.bitwise_not(output_image)


# Zapisanie wyniku
cv2.imwrite("zestawZadan6/zad13.png", binary_image)
