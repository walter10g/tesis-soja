from PIL import Image, ImageDraw, ImageEnhance
import os
import torch
from torchvision import transforms
import numpy as np
from model import load_model

# Mapeo de categorías a nombres y colores
CATEGORIES = {
    0: ("Malo", (255, 0, 0, 100)),  # Rojo translúcido
    1: ("Ligeramente Malo", (255, 255, 0, 100)),  # Amarillo translúcido
    2: ("Ligeramente Bueno", (144, 238, 144, 100)),  # Verde claro translúcido
    3: ("Bueno", (0, 128, 0, 100)),  # Verde oscuro translúcido
    4: ("Desconocido", (128, 128, 128, 100)),  # Gris translúcido
}

# Umbral para considerar una predicción como "desconocida"
UNKNOWN_THRESHOLD = 0.5

# Transformaciones para imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def divide_image(image, patch_size=(32, 32)):
    """
    Divide la imagen en parches de tamaño definido.
    """
    width, height = image.size
    patches = []
    coordinates = []

    for top in range(0, height, patch_size[1]):
        for left in range(0, width, patch_size[0]):
            box = (left, top, left + patch_size[0], top + patch_size[1])
            patch = image.crop(box)
            patches.append(patch)
            coordinates.append(box)

    return patches, coordinates

def predict_patches(patches, model, device):
    """
    Realiza predicciones para una lista de parches de imagen.
    """
    results = []
    for patch in patches:
        image_tensor = transform(patch).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
            
            # Clasificar como "Desconocido" si la probabilidad más alta es menor al umbral
            if probabilities[predicted_class] < UNKNOWN_THRESHOLD:
                results.append(4)  # Categoría "Desconocido"
            else:
                results.append(predicted_class)
    return results

def reconstruct_image(image, predictions, coordinates, patch_size=(32, 32)):
    """
    Reconstruye la imagen coloreada a partir de las predicciones.
    """
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for prediction, (left, top, right, bottom) in zip(predictions, coordinates):
        color = CATEGORIES.get(prediction, ("Desconocido", (128, 128, 128, 100)))[1]
        draw.rectangle([left,top, right, bottom], fill=color)

    # Combinar la imagen original con la superposición
    result = Image.alpha_composite(image.convert("RGBA"), overlay)
    return result.convert("RGB")

def calculate_category_percentages(predictions):
    """
    Calcula el porcentaje de píxeles para cada categoría.
    """
    total_patches = len(predictions)
    category_counts = {key: 0 for key in CATEGORIES.keys()}

    for prediction in predictions:
        category_counts[prediction] += 1

    percentages = {CATEGORIES[key][0]: (count / total_patches) * 100 for key, count in category_counts.items()}
    return percentages

if __name__ == "__main__":
    # Configuración
    model_path = "resnet18_soja_model.pth"
    input_dir = "test_images"
    output_dir = "test_images_segmented"

    os.makedirs(output_dir, exist_ok=True)

    # Cargar el modelo entrenado
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, num_classes=4).to(device)

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

