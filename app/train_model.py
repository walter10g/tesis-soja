import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from model import SegmentationModel
from torchvision.datasets import VisionDataset
from sklearn.metrics import jaccard_score
import numpy as np  # Importar numpy
import os
from PIL import Image

# Configuración general
num_epochs = 30
batch_size = 8
learning_rate = 0.001
num_classes = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "modelo_soja_colores.pth"

# Transformaciones con data augmentation
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
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
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
        print("Modelo previamente entrenado cargado.")
    except RuntimeError as e:
        print(f"Error al cargar pesos previos: {e}")
        print("Entrenamiento desde cero.")
else:
    print("Entrenamiento desde cero.")

# Configuración de entrenamiento
weights = torch.tensor([1.0, 1.0, 1.0, 1.0]).to(device)  # Ajusta los pesos según distribución de clases si es necesario
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

# Guardar modelo entrenado
torch.save(model.state_dict(), model_path)
print(f"Modelo guardado en '{model_path}'")

# Evaluación con métricas avanzadas
def compute_iou(predictions, masks, num_classes):
    iou_per_class = []
    for cls in range(num_classes):
        pred_cls = (predictions == cls).cpu().numpy().flatten()
        mask_cls = (masks == cls).cpu().numpy().flatten()
        iou = jaccard_score(mask_cls, pred_cls, average='binary', zero_division=0)
        iou_per_class.append(iou)
    return iou_per_class

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
        batch_ious = compute_iou(predictions, masks, num_classes)
        ious.append(batch_ious)

# Calcular promedio de IoU por clase
ious = np.array(ious)
iou_per_class = ious.mean(axis=0)

# Reporte de métricas
print(f"Precisión en el conjunto de prueba: {100 * correct / total:.2f}%")
for cls, iou in enumerate(iou_per_class):
    print(f"IoU para la clase {cls}: {iou:.4f}")
print(f"IoU promedio: {iou_per_class.mean():.4f}")
