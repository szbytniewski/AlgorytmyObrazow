import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("zdjecia/zdjeciaPakiet6/zad4.png", cv2.IMREAD_GRAYSCALE)

_, binary_image_cv = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

# plt.imshow(binary_image_cv)
# plt.show()
# plt.close()

structuring_element_cv = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

img_erosion = cv2.erode(binary_image_cv, structuring_element_cv, iterations=1)
img_dilation = cv2.dilate(img_erosion, structuring_element_cv, iterations=1)


img_open = cv2.bitwise_not(img_dilation)

cv2.imwrite("zestawZadan6/open_img.png", img_open)
