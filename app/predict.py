import os
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from model import initialize_model

# Configuración
MODEL_PATH = "model.pth"  # Ruta al modelo guardado
TEST_IMAGES_DIR = "test_images"  # Carpeta de imágenes a analizar
SEGMENTED_IMAGES_DIR = "test_images_segmented"  # Carpeta para guardar imágenes segmentadas
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BLOCK_SIZE = 16  # Tamaño del bloque (32x32)

# Colores para sombrear las categorías
CATEGORY_COLORS = {
    0: (0, 255, 0, 100),    # Verde translúcido (good)
    1: (128, 0, 128, 100),  # Lila translúcido (slightly_good)
    2: (255, 255, 0, 100),  # Amarillo translúcido (slightly_bad)
    3: (255, 0, 0, 100),    # Rojo translúcido (bad)
    "unknown": (128, 128, 128, 100)  # Gris translúcido (desconocido)
}

# Transformaciones para preprocesar las imágenes
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Igual que en el entrenamiento
])

# Cargar modelo completo
def load_model(model_path, device):
    # Cargar modelo completo
    model = torch.load(model_path, map_location=device)
    model.eval()  # Poner el modelo en modo evaluación
    return model

# Procesar una imagen
def process_image(image_path, model, device):
    # Cargar imagen
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Crear una nueva imagen para la segmentación
    segmented_image = image.copy()
    overlay = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(overlay)

    # Contadores para estadísticas
    category_counts = {key: 0 for key in CATEGORY_COLORS.keys()}

    # Dividir en bloques 32x32 y clasificar
    for y in range(0, height, BLOCK_SIZE):
        for x in range(0, width, BLOCK_SIZE):
            # Recortar bloque
            block = image.crop((x, y, x + BLOCK_SIZE, y + BLOCK_SIZE))
            
            # Redimensionar si el bloque es menor a BLOCK_SIZE (en los bordes)
            block = block.resize((BLOCK_SIZE, BLOCK_SIZE), Image.Resampling.BICUBIC)

            # Preprocesar bloque
            block_tensor = transform(block).unsqueeze(0).to(device)

            # Predecir categoría
            with torch.no_grad():
                outputs = model(block_tensor)
                _, predicted = torch.max(outputs, 1)
                category = predicted.item()

            # Determinar color del bloque
            if category in CATEGORY_COLORS:
                color = CATEGORY_COLORS[category]
                category_counts[category] += 1
            else:
                color = CATEGORY_COLORS["unknown"]
                category_counts["unknown"] += 1

            # Dibujar el bloque con sombreado translúcido
            draw.rectangle([x, y, x + BLOCK_SIZE, y + BLOCK_SIZE], fill=color)

    # Combinar imagen original con la capa de sombreado
    segmented_image = Image.alpha_composite(segmented_image.convert("RGBA"), overlay)

    # Convertir a RGB para guardar en formatos como JPEG
    segmented_image = segmented_image.convert("RGB")

    # Estadísticas de porcentaje
    total_blocks = sum(category_counts.values())
    print(f"Resultados para {os.path.basename(image_path)}:")
    for category, count in category_counts.items():
        percentage = (count / total_blocks) * 100 if total_blocks > 0 else 0
        label = "Desconocido" if category == "unknown" else f"Categoría {category}"
        print(f"  {label}: {percentage:.2f}%")

    return segmented_image

# Procesar todas las imágenes de la carpeta
def process_images(input_dir, output_dir, model, device):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Procesando {image_name}...")
            segmented_image = process_image(image_path, model, device)
            
            # Convertir a RGB antes de guardar
            segmented_image.save(os.path.join(output_dir, image_name))
            
            print(f"Imagen segmentada guardada: {image_name}")

# Main
if __name__ == "__main__":
    # Cargar modelo
    print("Cargando modelo...")
    model = load_model(MODEL_PATH, DEVICE)
    print("Modelo cargado.")

    # Procesar imágenes
    print("Procesando imágenes...")
    process_images(TEST_IMAGES_DIR, SEGMENTED_IMAGES_DIR, model, DEVICE)
    print("Todas las imágenes han sido procesadas.")
