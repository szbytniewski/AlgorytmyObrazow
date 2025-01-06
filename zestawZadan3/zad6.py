import cv2
import matplotlib.pyplot as plt
import numpy as np

# Wczytanie obrazu w kolorze
image = cv2.imread("zdjecia/zdjeciaPakiet3/czaszka.png")
if image is None:
    print("Błąd: Obraz nie został wczytany.")
    exit()


# Funkcja do mapowania wartości zgodnie z diagramem
def piecewise_linear_transform_color(img):
    output = np.zeros_like(img, dtype=np.uint8)

    for channel in range(3):  # Dla każdego kanału R, G, B
        for g in range(256):
            if 0 <= g <= 31:
                output[:, :, channel][img[:, :, channel] == g] = int((255 / 31) * g)
            elif 31 < g <= 95:
                output[:, :, channel][img[:, :, channel] == g] = int(
                    255 - ((255 / 64) * (g - 31))
                )
            elif 95 < g <= 159:
                output[:, :, channel][img[:, :, channel] == g] = int(
                    ((255 / 64) * (g - 95))
                )
            elif 159 < g <= 223:
                output[:, :, channel][img[:, :, channel] == g] = int(
                    255 - ((255 / 64) * (g - 159))
                )
            elif 223 < g <= 255:
                output[:, :, channel][img[:, :, channel] == g] = int(
                    ((255 / 32) * (g - 223))
                )
    return output


# Zastosowanie transformacji
transformed_image = piecewise_linear_transform_color(image)

# Zapisanie i wyświetlenie wyników
cv2.imwrite("transformed_czaszka_color.png", transformed_image)

# Wizualizacja
plt.figure(figsize=(12, 6))

# Oryginalny obraz
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Oryginalny obraz")
plt.axis("off")

# Obraz po transformacji
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
plt.title("Obraz po transformacji")
plt.axis("off")

plt.tight_layout()
plt.show()
