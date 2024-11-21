import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
from model import load_model

# Colores para cada clase
CLASS_COLORS = {
    0: [255, 0, 0],      # Rojo (bad)
    1: [255, 255, 0],    # Amarillo (slightly_bad)
    2: [144, 238, 144],  # Verde claro (slightly_good)
    3: [0, 128, 0],      # Verde oscuro (good)
}

# Transformaciones para imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionar para el modelo
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    """
    Preprocesa una imagen para ser entrada al modelo.
    """
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # Guardar el tamaño original
    image = transform(image).unsqueeze(0)  # Redimensionar y convertir a tensor
    return image, original_size

def segment_image(model, image_path, output_path):
    """
    Realiza la predicción y genera una imagen segmentada con el tamaño original.
    """
    # Preprocesar la imagen
    image_tensor, original_size = preprocess_image(image_path)
    
    # Realizar la predicción
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predictions = torch.max(outputs, 1)
    
    # Convertir la predicción en una máscara de colores
    predictions = predictions.squeeze().numpy()  # Convertir a numpy array
    mask = np.zeros((224, 224, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        mask[predictions == class_id] = color

    # Redimensionar la máscara al tamaño original
    mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)

    # Superponer la máscara en la imagen original
    original_image = cv2.imread(image_path)
    blended = cv2.addWeighted(original_image, 0.7, mask_resized, 0.3, 0)
    
    # Guardar la imagen segmentada
    cv2.imwrite(output_path, blended)
    print(f"Imagen segmentada guardada en {output_path}")

def process_folder(folder_path, output_folder, model):
    """
    Procesa todas las imágenes de una carpeta y guarda las segmentadas.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Recorrer todas las imágenes en la carpeta
    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)
        output_path = os.path.join(output_folder, f"segmented_{file_name}")
        
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Procesando {file_name}...")
            segment_image(model, image_path, output_path)
        else:
            print(f"Archivo {file_name} ignorado (no es una imagen).")

if __name__ == "__main__":
    # Cargar el modelo entrenado
    model = load_model("modelo_soja_adaptado.pth")

    # Directorios
    input_folder = "test_images"  # Carpeta con las imágenes de prueba
    output_folder = "test_images_segmented"  # Carpeta para las imágenes segmentadas

    # Procesar todas las imágenes en la carpeta
    process_folder(input_folder, output_folder, model)
