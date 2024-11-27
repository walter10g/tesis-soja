import os
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

# Colores para cada categoría (categorías numeradas como Cat 1, Cat 2, etc.)
CATEGORY_COLORS = {
    0: (128, 128, 128),  # Unknown/Background (gris)
    1: (255, 0, 0),      # bad (rojo)
    2: (0, 255, 0),      # good (verde)
    3: (255, 255, 0),    # slightly_bad (amarillo)
    4: (0, 0, 255),      # slightly_good (azul)
}

def analyze_image(image_path, model, output_dir, patch_size=(16, 16), device="cpu"):
    """
    Analiza una imagen dividiéndola en parches de 16x16, segmenta cada parche y genera una imagen sombreada.

    Args:
        image_path (str): Ruta de la imagen de entrada.
        model (torch.nn.Module): Modelo de segmentación entrenado.
        output_dir (str): Carpeta para guardar las imágenes segmentadas.
        patch_size (tuple): Tamaño de los parches (ancho, alto).
        device (str): Dispositivo ("cpu" o "cuda").
    """
    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Cargar la imagen original
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # Guardar tamaño original (ancho, alto)

    # Preparar transformaciones para redimensionar los parches
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Redimensionar cada parche
        transforms.ToTensor(),         # Convertir a tensor
    ])

    # Convertir la imagen en un array NumPy para manipular parches
    image_array = np.array(image)

    # Calcular cuántos parches caben en la imagen
    width, height = original_size
    patch_width, patch_height = patch_size
    num_patches_x = width // patch_width
    num_patches_y = height // patch_height

    # Crear una máscara de colores con el mismo tamaño que la imagen original
    color_mask = np.zeros_like(image_array, dtype=np.uint8)

    # Contador de píxeles para cada categoría
    pixel_counts = {category: 0 for category in CATEGORY_COLORS.keys()}

    # Recorrer cada parche de la imagen
    for y in range(num_patches_y):
        for x in range(num_patches_x):
            # Coordenadas del parche actual
            x_start = x * patch_width
            x_end = x_start + patch_width
            y_start = y * patch_height
            y_end = y_start + patch_height

            # Extraer el parche de la imagen
            patch = image_array[y_start:y_end, x_start:x_end]
            patch_image = Image.fromarray(patch)

            # Transformar el parche para el modelo
            input_tensor = preprocess(patch_image).unsqueeze(0).to(device)

            # Realizar la predicción con el modelo
            with torch.no_grad():
                output = model(input_tensor)["out"]
                predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            # Redimensionar la máscara predicha al tamaño original del parche
            predicted_mask_resized = np.array(
                Image.fromarray(predicted_mask.astype(np.uint8)).resize((patch_width, patch_height), resample=Image.NEAREST)
            )

            # Actualizar el conteo de píxeles para cada categoría
            unique, counts = np.unique(predicted_mask_resized, return_counts=True)
            for category, count in zip(unique, counts):
                pixel_counts[category] += count

            # Crear la máscara de colores para el parche
            patch_color_mask = np.zeros((patch_height, patch_width, 3), dtype=np.uint8)
            for category, color in CATEGORY_COLORS.items():
                patch_color_mask[predicted_mask_resized == category] = color

            # Insertar el parche coloreado en la máscara completa
            color_mask[y_start:y_end, x_start:x_end] = patch_color_mask

    # Crear una nueva imagen a partir de la máscara de colores
    color_mask_image = Image.fromarray(color_mask)

    # Superponer la máscara sobre la imagen original
    segmented_image = Image.blend(image, color_mask_image, alpha=0.5)

    # Guardar la imagen segmentada
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    segmented_image.save(output_path)
    print(f"Imagen segmentada guardada en {output_path}")

    # Calcular el porcentaje de píxeles por categoría
    total_pixels = sum(pixel_counts.values())
    category_percentages = {
        f"Cat {category}": (count / total_pixels) * 100 for category, count in pixel_counts.items()
    }

    # Imprimir los porcentajes finales por categoría
    print(f"Resultados para {os.path.basename(image_path)}:")
    for category, percentage in category_percentages.items():
        print(f"{category}: {percentage:.2f}%")

def main():
    # Configuración
    model_path = "deeplab_complete_model.pth"
    input_dir = "test_images"
    output_dir = "test_images_segmented"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Cargar el modelo entrenado
    model = torch.load(model_path, map_location=device)
    model.eval()

    # Procesar cada imagen en el directorio de entrada
    for image_filename in os.listdir(input_dir):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, image_filename)
            print(f"Procesando imagen: {image_filename}")
            analyze_image(image_path, model, output_dir, patch_size=(16, 16), device=device)

if __name__ == "__main__":
    main()
