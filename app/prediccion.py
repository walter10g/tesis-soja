from PIL import Image, ImageDraw
import os
import torch
from torchvision import transforms
import numpy as np
from model import load_model

# Mapeo de categorías a nombres y colores
CATEGORIES = {
    0: ("Identificado", (255, 0, 0, 100)),  # Rojo translúcido
    1: ("Desconocido", (200, 200, 200, 100)),  # Gris claro translúcido
}

# Umbral para clasificar un parche como identificado
IDENTIFIED_THRESHOLD = 0.5

# Transformaciones simples: solo convertir a tensor
transform = transforms.ToTensor()

def divide_image(image, patch_size=(224, 224)):
    """
    Divide la imagen en parches de tamaño definido.
    """
    width, height = image.size
    patches = []
    coordinates = []

    for top in range(0, height, patch_size[1]):
        for left in range(0, width, patch_size[0]):
            box = (left, top, min(left + patch_size[0], width), min(top + patch_size[1], height))
            patch = image.crop(box)
            patch = patch.resize((224, 224))  # Asegurar tamaño adecuado para el modelo
            patches.append(patch)
            coordinates.append(box)

    return patches, coordinates

def predict_patches(patches, model, device):
    """
    Realiza predicciones para una lista de parches de imagen.
    """
    results = []
    model.eval()
    for patch in patches:
        patch_tensor = transform(patch).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(patch_tensor)
            probability = torch.sigmoid(output).item()  # Clasificación binaria
            if probability >= IDENTIFIED_THRESHOLD:
                results.append(0)  # Identificado
            else:
                results.append(1)  # Desconocido
    return results

def reconstruct_image(image, predictions, coordinates):
    """
    Reconstruye la imagen coloreada a partir de las predicciones.
    """
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for prediction, (left, top, right, bottom) in zip(predictions, coordinates):
        color = CATEGORIES.get(prediction, ("Desconocido", (200, 200, 200, 100)))[1]
        draw.rectangle([left, top, right, bottom], fill=color)

    # Combinar la imagen original con la superposición
    result = Image.alpha_composite(image.convert("RGBA"), overlay)
    return result.convert("RGB")

def calculate_category_percentages(predictions):
    """
    Calcula el porcentaje de parches para las categorías.
    """
    total_patches = len(predictions)
    identified_count = predictions.count(0)
    unknown_count = predictions.count(1)

    percentages = {
        "Identificado": (identified_count / total_patches) * 100,
        "Desconocido": (unknown_count / total_patches) * 100
    }
    return percentages

if __name__ == "__main__":
    # Configuración
    model_path = "resnet18_soja_model.pth"
    input_dir = "test_images"
    output_dir = "test_images_segmented"

    os.makedirs(output_dir, exist_ok=True)

    # Cargar el modelo entrenado
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path).to(device)

    # Procesar todas las imágenes de la carpeta de entrada
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        if not os.path.isfile(input_path):
            continue

        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_segmented.jpg")

        # Cargar y dividir la imagen en parches
        original_image = Image.open(input_path).convert("RGB")
        patches, coordinates = divide_image(original_image)

        # Realizar predicción para cada parche
        predictions = predict_patches(patches, model, device)

        # Calcular porcentajes de cada categoría
        percentages = calculate_category_percentages(predictions)

        # Imprimir porcentajes
        print(f"Porcentajes por categoría para {filename}:")
        for category, percentage in percentages.items():
            print(f"  {category}: {percentage:.2f}%")

        # Reconstruir la imagen segmentada sombreada
        shaded_image = reconstruct_image(original_image, predictions, coordinates)

        # Guardar la imagen sombreada
        shaded_image.save(output_path)
        print(f"Imagen segmentada y sombreada guardada en '{output_path}'")
