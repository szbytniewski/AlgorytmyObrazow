import numpy as np

neighborhood = [
    (125, 130, 240),
    (120, 250, 75),
    (235, 55, 130),
    (130, 130, 190),
    (255, 240, 0),
    (35, 180, 75),
    (165, 75, 165),
    (195, 195, 195),
    (255, 175, 200),
]

data = np.array(neighborhood)

# Filtr minimalny
min_pixel = np.min(data, axis=0)

# Filtr maksymalny
max_pixel = np.max(data, axis=0)

# Filtr medianowy
median_pixel = np.median(data, axis=0)

# Wyniki
print("Filtr minimalny: (r, g, b) =", tuple(min_pixel))
print("Filtr maksymalny: (r, g, b) =", tuple(max_pixel))
print("Filtr medianowy: (r, g, b) =", tuple(map(int, median_pixel)))
