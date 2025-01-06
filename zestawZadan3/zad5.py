import cv2
import matplotlib.pyplot as plt
import numpy as np

# Wczytaj obraz w skali szarości
image = cv2.imread("zdjecia/zdjeciaPakiet3/czaszka.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Błąd: Obraz nie został wczytany. Sprawdź ścieżkę do pliku.")
    exit()

# (a) Wyrównanie histogramu
equalized_image = cv2.equalizeHist(image)

# Zapisanie wyrównanego obrazu
cv2.imwrite("czaszka_equalized.png", equalized_image)


# (b) Hiperbolizacja histogramu z parametrem α = -1/3
def hyperbolize_histogram(img, alpha):
    img_normalized = img / 255.0  # Normalizacja pikseli do zakresu [0, 1]
    hyperbolized = np.tanh(alpha * img_normalized)  # Funkcja hiperboliczna
    hyperbolized_normalized = (
        (hyperbolized - hyperbolized.min()) / (hyperbolized.max() - hyperbolized.min())
    ) * 255  # Skalowanie
    return hyperbolized_normalized.astype(np.uint8)


alpha = -1 / 3
hyperbolized_image = hyperbolize_histogram(equalized_image, alpha)

# Zapisanie hiperbolizowanego obrazu
cv2.imwrite("czaszka_hyperbolized.png", hyperbolized_image)


# Obliczanie histogramów
def compute_histograms(img):
    # Histogram znormalizowany Hn(g)
    hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
    normalized_hist = hist / hist.sum()

    # Histogram skumulowany Hs(g)
    cumulative_hist = np.cumsum(normalized_hist)
    return normalized_hist, cumulative_hist


# Histogramy dla oryginalnego, wyrównanego i hiperbolizowanego obrazu
hist_original, cum_hist_original = compute_histograms(image)
hist_equalized, cum_hist_equalized = compute_histograms(equalized_image)
hist_hyperbolized, cum_hist_hyperbolized = compute_histograms(hyperbolized_image)


# Wizualizacja
def plot_histograms(img, hist, cum_hist, title, filename):
    plt.figure(figsize=(12, 6))

    # Obraz
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title(f"{title} - Obraz")
    plt.axis("off")

    # Histogram znormalizowany
    plt.subplot(1, 3, 2)
    plt.plot(hist, color="blue")
    plt.title(f"{title} - Histogram Hn(g)")
    plt.xlabel("Wartości szarości")
    plt.ylabel("Częstość występowania")

    # Histogram skumulowany
    plt.subplot(1, 3, 3)
    plt.plot(cum_hist, color="green")
    plt.title(f"{title} - Skumulowany Histogram Hs(g)")
    plt.xlabel("Wartości szarości")
    plt.ylabel("Skumulowana częstość")

    # Zapisanie wykresów
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# Generowanie wykresów
plot_histograms(
    image, hist_original, cum_hist_original, "Oryginalny", "original_histograms.png"
)
plot_histograms(
    equalized_image,
    hist_equalized,
    cum_hist_equalized,
    "Wyrównany",
    "equalized_histograms.png",
)
plot_histograms(
    hyperbolized_image,
    hist_hyperbolized,
    cum_hist_hyperbolized,
    "Hiperbolizowany",
    "hyperbolized_histograms.png",
)
