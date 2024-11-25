import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from model import SegmentationModel
from torchvision.datasets import VisionDataset
from sklearn.metrics import jaccard_score
import numpy as np
import os
from PIL import Image

# Configuración general
num_epochs = 30
batch_size = 16
learning_rate = 0.00001  # Learning rate reducido
weight_decay = 1e-4      # Regularización
num_classes = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Uso de GPU si está disponible
model_path = "modelo_soja_colores.pth"

# Verificar si se está utilizando la GPU
if device.type == "cuda":
    print(f"Entrenamiento utilizando: {torch.cuda.get_device_name(0)}")
else:
    print("Entrenamiento utilizando CPU.")

# Transformaciones con data augmentation
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mask_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.NEAREST),
    transforms.ToTensor(),
])

# Dataset personalizado
class SegmentationDataset(VisionDataset):
    def __init__(self, images_dir, masks_dir, image_transform=None, mask_transform=None):
        super(SegmentationDataset, self).__init__(images_dir)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = (mask.squeeze(0) * (num_classes - 1) / 255).long()  # Normalización de máscara

        return image, mask

# Directorios
images_dir = "datasets/colors/images"
masks_dir = "datasets/colors/masks"

# Cargar el dataset
dataset = SegmentationDataset(images_dir, masks_dir, image_transform, mask_transform)

# Dividir en conjuntos de entrenamiento y prueba
torch.manual_seed(42)  # Reproducibilidad
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Inicializar modelo
model = SegmentationModel(num_classes=num_classes).to(device)

# Cargar modelo existente
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print("Modelo previamente entrenado cargado.")
    except RuntimeError as e:
        print(f"Error al cargar pesos previos: {e}")
        print("Entrenamiento desde cero.")
else:
    print("Entrenamiento desde cero.")

# Configuración de entrenamiento
weights = torch.tensor([0.5, 1.0, 1.0, 1.0]).to(device)  # Ajustar pesos según distribución de clases
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Entrenamiento
print("Iniciando entrenamiento...")
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Evaluación intermedia cada 5 épocas
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            model.eval()
            print(f"Evaluación intermedia después de {epoch + 1} épocas:")
            for i, (images, masks) in enumerate(test_loader):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
                print(f"Valores únicos en predicciones: {torch.unique(predictions)}")
                
                # Guardar algunas predicciones como imágenes para visualización
                if i < 5:  # Guardar solo las primeras 5
                    for j in range(images.size(0)):
                        pred_img = predictions[j].cpu().numpy()
                        mask_img = masks[j].cpu().numpy()
                        
                        # Guardar predicciones y máscaras
                        Image.fromarray((pred_img * 85).astype(np.uint8)).save(f"output_pred_epoch_{epoch+1}_{i}_{j}.png")
                        Image.fromarray((mask_img * 85).astype(np.uint8)).save(f"output_mask_epoch_{epoch+1}_{i}_{j}.png")
            model.train()

# Guardar modelo entrenado
torch.save(model.state_dict(), model_path)
print(f"Modelo guardado en '{model_path}'")

# Evaluación final
model.eval()
correct = 0
total = 0
ious = []

with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == masks).sum().item()
        total += masks.numel()

        # Calcular IoU por lote
        ious.append([
            jaccard_score((predictions == cls).cpu().numpy().flatten(),
                          (masks == cls).cpu().numpy().flatten(),
                          average='binary', zero_division=0)
            for cls in range(num_classes)
        ])

# Calcular promedio de IoU por clase
ious = np.array(ious)
iou_per_class = ious.mean(axis=0)

# Reporte de métricas
print(f"Precisión en el conjunto de prueba: {100 * correct / total:.2f}%")
for cls, iou in enumerate(iou_per_class):
    print(f"IoU para la clase {cls}: {iou:.4f}")
print(f"IoU promedio: {iou_per_class.mean():.4f}")
