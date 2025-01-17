import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("zdjecia/zdjeciaPakiet6/test.png", cv2.IMREAD_GRAYSCALE)

_, binary_image_cv = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# plt.imshow(binary_image_cv)
# plt.show()
# plt.close()

# structuring_element_cv = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
structuring_element_cv = np.array([[0, 0], [0, 1]], dtype=np.uint8)


img_erosion = cv2.erode(binary_image_cv, structuring_element_cv, iterations=1)

# img_erosion = cv2.bitwise_not(img_erosion)

cv2.imwrite("zestawZadan6/test2.png", img_erosion)
