from PIL import Image, ImageEnhance
import os
import random

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
images_per_category = 80
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

def create_image_with_variation(color, factor_range=(0.8, 1.2)):
    """
    Crea una imagen con color sólido y variación de brillo.
    """
    img = Image.new("RGB", image_size, color)  # Imagen base
    return add_brightness_variation(img, factor_range)

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

        # Crear imagen con variación de brillo
        img = create_image_with_variation(color)
        # Crear máscara correspondiente
        mask = create_mask(mask_value)

        # Guardar en las carpetas correspondientes
        img.save(os.path.join(images_dir, f"{category}_{i+1}.jpg"))
        mask.save(os.path.join(masks_dir, f"{category}_{i+1}.png"))

print(f"80 imágenes y máscaras generadas por categoría, guardadas en {output_dir}")
