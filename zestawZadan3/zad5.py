import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.exposure import adjust_sigmoid
import cv2

# Wczytaj obraz czaszki
image = imread("zestawZadan3/czaszka_equalized.png", as_gray=True)

# Hiperbolizacja z parametrem α = -1/3
alpha = -1 / 3
hyperbolized = adjust_sigmoid(image, cutoff=0.5, gain=-1 / alpha)

# Wizualizacja
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Oryginalny obraz")
# plt.imshow(image, cmap="gray")
# plt.colorbar()

# plt.subplot(1, 2, 2)
# plt.title("Hiperbolizowany obraz")
# plt.imshow(hyperbolized, cmap="gray")
# plt.colorbar()
# plt.show()

cv2.imwrite("zestawZadan3/czaska_hiper2.png", hyperbolized)
