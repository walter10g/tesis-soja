import torch
from PIL import Image, ImageDraw
import os
import numpy as np
from model import SimpleCNN

# Colores para las categorías
IDENTIFIED_COLOR = (255, 0, 0, 100)  # Rojo translúcido
UNKNOWN_COLOR = (200, 200, 200, 100)  # Gris claro translúcido

# Umbral para considerar una predicción como "identificada"
IDENTIFIED_THRESHOLD = 0.5

def divide_image(image, patch_size=(8, 8)):
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
            patch = patch.resize(patch_size)  # Redimensionar al tamaño esperado por el modelo
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
        # Convertir el parche a tensor
        patch_tensor = torch.tensor(np.array(patch)).float().permute(2, 0, 1) / 255.0  # Normalización implícita a [0, 1]
        patch_tensor = patch_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(patch_tensor)
            probability = torch.sigmoid(output).item()

            # Mostrar probabilidad para debugging
            print(f"Probabilidad del parche: {probability:.4f}")

            # Clasificar como identificada o desconocida según el umbral
            if probability >= IDENTIFIED_THRESHOLD:
                results.append("identified")
            else:
                results.append("unknown")

    return results

def reconstruct_image(image, predictions, coordinates):
    """
    Reconstruye la imagen coloreada a partir de las predicciones.
    """
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for prediction, (left, top, right, bottom) in zip(predictions, coordinates):
        color = IDENTIFIED_COLOR if prediction == "identified" else UNKNOWN_COLOR
        draw.rectangle([left, top, right, bottom], fill=color)

    # Combinar la imagen original con la superposición
    result = Image.alpha_composite(image.convert("RGBA"), overlay)
    return result.convert("RGB")

def calculate_category_percentages(predictions):
    """
    Calcula el porcentaje de parches para las categorías identificada y desconocida.
    """
    total_patches = len(predictions)
    identified_count = predictions.count("identified")
    unknown_count = predictions.count("unknown")

    percentages = {
        "Identificada": (identified_count / total_patches) * 100,
        "Desconocida": (unknown_count / total_patches) * 100
    }
    return percentages

if __name__ == "__main__":
    # Configuración
    model_path = "simple_cnn_good_model.pth"
    input_dir = "test_images"
    output_dir = "test_images_segmented"

    os.makedirs(output_dir, exist_ok=True)

    # Cargar el modelo entrenado
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Procesar todas las imágenes en la carpeta de entrada
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        if not os.path.isfile(input_path):
            continue

        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_segmented.jpg")

        # Cargar y dividir la imagen en parches
        original_image = Image.open(input_path).convert("RGB")
        patches, coordinates = divide_image(original_image, patch_size=(8, 8))

        # Realizar predicción para cada parche
        print(f"Procesando {filename}...")
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
