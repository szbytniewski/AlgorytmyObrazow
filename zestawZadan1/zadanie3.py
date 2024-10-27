import numpy as np

# Dane RGB
colors = np.array([[200, 200, 25],
                   [200, 25, 200],
                   [25, 200, 200]])

# Wagi odpowiadające wrażliwości oczu
weights = np.array([0.299, 0.587, 0.114])

# Konwersja do skali szarości
gray_values = np.dot(colors, weights)

# Znalezienie najjaśniejszego koloru
max_gray_value = np.max(gray_values)
brightest_color_index = np.argmax(gray_values)

# Wyniki
for i, gray in enumerate(gray_values):
    print(f"Kolor RGB {colors[i]} -> Skala szarości: {gray}")

print(f"\nNajjaśniejszy kolor po konwersji to: {colors[brightest_color_index]} o jasności {max_gray_value}")
