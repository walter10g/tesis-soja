import os
from PIL import Image
import numpy as np

# Directorios de entrada y salida
input_dir = "datasets/colors/images/train"
output_dir = "datasets/masks"

# Mapear categorías a valores únicos para las máscaras
category_to_value = {
    "bad": 1,
    "good": 2,
    "slightly_bad": 3,
    "slightly_good": 4,
}

def generate_mask(image_path, category_value):
    """
    Genera una máscara para una imagen específica usando variaciones de intensidad.
    """
    # Cargar la imagen original
    image = Image.open(image_path).convert("L")  # Convertir a escala de grises
    image_array = np.array(image)

    # Normalizar los valores para generar variaciones
    normalized = (image_array / 255.0) * category_value
    mask = (normalized * 255 / category_value).astype(np.uint8)

    return Image.fromarray(mask)

def process_dataset(input_dir, output_dir):
    """
    Procesa el dataset para generar máscaras.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for category, value in category_to_value.items():
        category_input_path = os.path.join(input_dir, category)
        category_output_path = os.path.join(output_dir, category)

        if not os.path.exists(category_output_path):
            os.makedirs(category_output_path)

        for filename in os.listdir(category_input_path):
            if filename.endswith((".jpg", ".png")):
                input_path = os.path.join(category_input_path, filename)
                output_path = os.path.join(category_output_path, filename)

                # Generar máscara
                mask = generate_mask(input_path, value)
                mask.save(output_path)

                print(f"Generada máscara para {filename} en la categoría {category}")

# Ejecutar el procesamiento
process_dataset(input_dir, output_dir)
