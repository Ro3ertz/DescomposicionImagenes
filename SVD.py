import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import cv2 # type: ignore
from skimage.color import rgb2gray # type: ignore

def load_image(image_path):
    """Cargar la imagen y la convierte a escala de grises"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = rgb2gray(img) #Conversion a escala de grises
    return gray

def compress_image(img, k):
    """Aplica la Descomposición en Valores Singulares (SVD) para comprimir la imagen."""
    U, S, Vt = np.linalg.svd(img, full_matrices=False)
    
    # Reducir la cantidad de valores singulares
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]
    
    # Reconstrucción de la imagen comprimida
    compressed_img = np.dot(U_k, np.dot(S_k, Vt_k))
    return compressed_img

def plot_compression_results(original, compressed_images, ks):
    """Muestra la imagen original y las comprimidas con diferentes valores de k."""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, len(ks) + 1, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original")
    plt.axis('off')
    
    for i, (img, k) in enumerate(zip(compressed_images, ks)):
        plt.subplot(1, len(ks) + 1, i + 2)
        plt.imshow(img, cmap='gray')
        plt.title(f"k = {k}")
        plt.axis('off')
    
    plt.show()

# Cargar imagen
gray_image = load_image(r"D:\SVD\example.jpg")

# Aplicar compresión con diferentes valores de k
ks = [5, 20, 50]
compressed_images = [compress_image(gray_image, k) for k in ks]

# Mostrar los resultados
plot_compression_results(gray_image, compressed_images, ks)
