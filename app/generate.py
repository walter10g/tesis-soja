from PIL import Image, ImageEnhance, ImageFilter
import os
import random
import numpy as np

# Definir rangos de colores en RGB para cada categoría
CATEGORY_COLOR_RANGES = {
    "bad": [(110, 65, 50), (160, 100, 70)],           # Marrón oscuro (planta muerta)
    "slightly_bad": [(210, 180, 100), (240, 210, 130)],  # Amarillo claro
    "slightly_good": [(180, 240, 180), (210, 250, 210)], # Verde claro
    "good": [(30, 90, 30), (50, 120, 50)]             # Verde oscuro
}

# Mapear categorías a valores de máscara
CATEGORY_MASK_VALUES = {
    "bad": 0,
    "slightly_bad": 85,
    "slightly_good": 170,
    "good": 255
}

# Carpeta base para guardar las imágenes generadas
output_dir = "datasets/colors"
images_dir = os.path.join(output_dir, "images")
masks_dir = os.path.join(output_dir, "masks")

os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

# Número de imágenes a generar por categoría
images_per_category = 500  # Cambiado a 500
image_size = (224, 224)  # Tamaño de las imágenes

def generate_random_color(color_range):
    """
    Genera un color aleatorio dentro del rango definido.
    """
    return tuple(random.randint(low, high) for low, high in zip(color_range[0], color_range[1]))

def add_brightness_variation(img, factor_range=(0.8, 1.2)):
    """
    Agrega una variación de brillo aleatoria a una imagen.
    """
    enhancer = ImageEnhance.Brightness(img)
    factor = random.uniform(*factor_range)  # Factor de brillo entre 80% y 120%
    return enhancer.enhance(factor)

def add_contrast_variation(img, factor_range=(0.8, 1.2)):
    """
    Agrega una variación de contraste aleatoria a una imagen.
    """
    enhancer = ImageEnhance.Contrast(img)
    factor = random.uniform(*factor_range)  # Factor de contraste entre 80% y 120%
    return enhancer.enhance(factor)

def add_noise(img, noise_factor=0.05):
    """
    Agrega ruido aleatorio a una imagen.
    """
    arr = np.array(img)
    noise = np.random.normal(0, noise_factor * 255, arr.shape).astype(np.int16)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_arr)

def add_blur(img, max_radius=2):
    """
    Aplica un desenfoque aleatorio a la imagen.
    """
    if random.random() < 0.5:  # 50% de probabilidad de aplicar blur
        radius = random.uniform(0, max_radius)
        return img.filter(ImageFilter.GaussianBlur(radius))
    return img

def create_image_with_variation(color):
    """
    Crea una imagen con color sólido y agrega variaciones realistas.
    """
    img = Image.new("RGB", image_size, color)  # Imagen base
    img = add_brightness_variation(img)
    img = add_contrast_variation(img)
    img = add_noise(img)
    img = add_blur(img)
    return img

def create_mask(value):
    """
    Crea una máscara sólida con un valor único.
    """
    return Image.new("L", image_size, value)

# Generar las imágenes y máscaras para cada categoría
for category, color_range in CATEGORY_COLOR_RANGES.items():
    for i in range(images_per_category):
        # Generar un color aleatorio dentro del rango de la categoría
        color = generate_random_color(color_range)
        mask_value = CATEGORY_MASK_VALUES[category]

        # Crear imagen con variaciones
        img = create_image_with_variation(color)
        # Crear máscara correspondiente
        mask = create_mask(mask_value)

        # Guardar en las carpetas correspondientes
        img.save(os.path.join(images_dir, f"{category}_{i+1}.jpg"))
        mask.save(os.path.join(masks_dir, f"{category}_{i+1}.png"))

print(f"{images_per_category} imágenes y máscaras generadas por categoría, guardadas en {output_dir}")
