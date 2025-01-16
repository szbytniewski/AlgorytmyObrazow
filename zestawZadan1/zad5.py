import cv2
import numpy as np
import random

# Wczytaj obraz oryginalny
image_path = "zdjecia/zdjeciaPakiet1/ds3.jpg"
image = cv2.imread(image_path)

# Ustawienia
output_image = image.copy()
num_points = 10000000  # Liczba punktów
circle_radius = 10  # Promień okręgów
color_intensity = 255

# Generowanie efektu
for _ in range(num_points):
    x = random.randint(0, image.shape[1] - 1)
    y = random.randint(0, image.shape[0] - 1)
    color = image[y, x].tolist()
    cv2.circle(output_image, (x, y), circle_radius, color, -1)

# Zapisz wynik
output_path = "zestawZadan1/output_puentylizm.jpg"
cv2.imwrite(output_path, output_image)

print(f"Efekt puentylizmu zapisano do: {output_path}")
