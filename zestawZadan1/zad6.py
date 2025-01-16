import cv2
import numpy as np

# Wczytaj obraz
input_image_path = "zdjecia/zdjeciaPakiet1/potworek.png"  # Podmień na ścieżkę do obrazu
image = cv2.imread(input_image_path)
b, g, r = cv2.split(image)

# Wymiary docelowe
target_width = 600
target_height = 360


# 1. Skalowanie algorytmem Nearest Neighbor
def scale_nearest(channel, width, height):
    return cv2.resize(channel, (width, height), interpolation=cv2.INTER_NEAREST)


b_nearest = scale_nearest(b, target_width, target_height)
g_nearest = scale_nearest(g, target_width, target_height)
r_nearest = scale_nearest(r, target_width, target_height)
nearest_scaled = cv2.merge((b_nearest, g_nearest, r_nearest))


# 2. Rozszerzony Nearest Neighbor
def extended_nearest_neighbor(channel, target_size):
    height, width = target_size
    output = np.zeros((height, width), dtype=np.uint8)
    scale_x = channel.shape[1] / width
    scale_y = channel.shape[0] / height

    for y in range(height):
        for x in range(width):
            src_x = int(x * scale_x)
            src_y = int(y * scale_y)

            # Pobranie dwóch najbliższych pikseli
            neighbors = [
                channel[src_y, src_x],
                channel[min(src_y + 1, channel.shape[0] - 1), src_x],
            ]
            # Średnia dwóch pikseli
            output[y, x] = np.mean(neighbors).astype(np.uint8)

    return output


b_extended = extended_nearest_neighbor(b, (target_height, target_width))
g_extended = extended_nearest_neighbor(g, (target_height, target_width))
r_extended = extended_nearest_neighbor(r, (target_height, target_width))
extended_nearest_scaled = cv2.merge((b_extended, g_extended, r_extended))


# 3. Interpolacja dwuliniowa
def scale_bilinear(channel, width, height):
    return cv2.resize(channel, (width, height), interpolation=cv2.INTER_LINEAR)


b_bilinear = scale_bilinear(b, target_width, target_height)
g_bilinear = scale_bilinear(g, target_width, target_height)
r_bilinear = scale_bilinear(r, target_width, target_height)
bilinear_scaled = cv2.merge((b_bilinear, g_bilinear, r_bilinear))


# 4. Interpolacja na podstawie średniej 4 pikseli
def custom_interpolation(channel, target_size):
    height, width = target_size
    output = np.zeros((height, width), dtype=np.uint8)
    scale_x = channel.shape[1] / width
    scale_y = channel.shape[0] / height

    for y in range(height):
        for x in range(width):
            src_x = x * scale_x
            src_y = y * scale_y

            x0 = int(np.floor(src_x))
            x1 = min(x0 + 1, channel.shape[1] - 1)
            y0 = int(np.floor(src_y))
            y1 = min(y0 + 1, channel.shape[0] - 1)

            # Cztery najbliższe piksele
            neighbors = [
                channel[y0, x0],
                channel[y1, x0],
                channel[y0, x1],
                channel[y1, x1],
            ]
            # Średnia najwyższej i najniższej wartości
            output[y, x] = np.mean([np.min(neighbors), np.max(neighbors)]).astype(
                np.uint8
            )

    return output


b_custom = custom_interpolation(b, (target_height, target_width))
g_custom = custom_interpolation(g, (target_height, target_width))
r_custom = custom_interpolation(r, (target_height, target_width))
custom_scaled = cv2.merge((b_custom, g_custom, r_custom))


# Zapis wyników
cv2.imwrite("zestawZadan1/scaled_nearest.png", nearest_scaled)
cv2.imwrite("zestawZadan1/scaled_extended_nearest.png", extended_nearest_scaled)
cv2.imwrite("zestawZadan1/scaled_bilinear.png", bilinear_scaled)
cv2.imwrite("zestawZadan1/scaled_custom.png", custom_scaled)

print("Wszystkie obrazy zostały przeskalowane i zapisane.")
