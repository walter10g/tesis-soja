import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
from model import load_model

# Colores para cada clase
CLASS_COLORS = {
    0: [153, 51, 51],      # Marrón rojizo (suelo muerto)
    1: [255, 204, 102],    # Amarillo (ligeramente malo)
    2: [102, 204, 102],    # Verde claro (moderadamente bueno)
    3: [0, 102, 51],       # Verde oscuro (muy bueno)
}

# Transformaciones para imágenes
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    """
    Carga y preprocesa una imagen.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"La imagen {image_path} no existe.")

    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image_tensor = transform(image).unsqueeze(0)

    print(f"Tamaño del tensor preprocesado para {image_path}: {image_tensor.shape}")
    print(f"Valores máximos y mínimos del tensor: {image_tensor.max()}, {image_tensor.min()}")
    return image_tensor, original_size

def save_colored_mask(predictions, output_path, original_size):
    """
    Crea y guarda una máscara segmentada coloreada.
    """
    mask = np.zeros((predictions.shape[0], predictions.shape[1], 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        mask[predictions == class_id] = color

    mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(output_path, mask_resized)
    print(f"Máscara segmentada guardada en {output_path}")


def save_raw_predictions(predictions, output_path):
    """
    Guarda las predicciones crudas en escala de grises para depuración.
    """
    try:
        if np.max(predictions) == 0:
            raise ValueError("Predicciones contienen solo ceros. Verifica el modelo y los datos.")
        
        # Normalizar las predicciones a 0-255
        normalized_pred = (predictions / np.max(predictions)) * 255.0
        normalized_pred = normalized_pred.astype(np.uint8)

        # Guardar como imagen
        cv2.imwrite(output_path, normalized_pred)
        print(f"Predicciones crudas guardadas en {output_path}")
    except Exception as e:
        print(f"Error guardando predicciones crudas: {e}")


def segment_large_image(model, image_path, output_path, debug_path, block_size=224):
    """
    Divide imágenes grandes en bloques, segmenta cada bloque y los combina.
    """
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    width, height = original_size

    predictions = np.zeros((height, width), dtype=np.uint8)

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Recortar el bloque
            block = image.crop((x, y, min(x + block_size, width), min(y + block_size, height)))

            # Preprocesar y predecir
            block_tensor = transform(block).unsqueeze(0)
            with torch.no_grad():
                output = model(block_tensor)['out']
                pred_block = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            # Agregar al canvas de predicciones
            predictions[y:y + pred_block.shape[0], x:x + pred_block.shape[1]] = pred_block

   # Añadir el print aquí para depurar las predicciones
    print(f"Valores únicos en predicciones: {np.unique(predictions)}")

    # Guardar predicciones crudas para depuración
    save_raw_predictions(predictions, debug_path)

    # Crear y guardar la máscara segmentada
    save_colored_mask(predictions, output_path, original_size)

def process_folder(folder_path, output_folder, debug_folder, model):
    """
    Procesa todas las imágenes de una carpeta.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"La carpeta {folder_path} no existe.")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(debug_folder, exist_ok=True)

    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Archivo {file_name} ignorado (no es una imagen).")
            continue

        image_path = os.path.join(folder_path, file_name)
        output_path = os.path.join(output_folder, f"segmented_{file_name}")
        debug_path = os.path.join(debug_folder, f"raw_predictions_{file_name}")
        print(f"Procesando {file_name}...")

        try:
            segment_large_image(model, image_path, output_path, debug_path)
        except Exception as e:
            print(f"Error procesando {file_name}: {e}")

if __name__ == "__main__":
    # Cargar el modelo entrenado
    model = load_model("modelo_soja_colores.pth")
    model.eval()

    # Directorios
    input_folder = "test_images"
    output_folder = "test_images_segmented"
    debug_folder = "debug_predictions"

    # Procesar todas las imágenes en la carpeta
    process_folder(input_folder, output_folder, debug_folder, model)
