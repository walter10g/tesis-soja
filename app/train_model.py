import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from model import SegmentationModel
from torchvision.datasets import VisionDataset
import os
from PIL import Image

# Configuración general
num_epochs = 10
batch_size = 4
learning_rate = 0.001
num_classes=4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "modelo_soja_colores.pth"

# Transformaciones
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
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
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Inicializar modelo
model = SegmentationModel(num_classes=4).to(device)

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
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Entrenamiento
print("Iniciando entrenamiento...")
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)  # Salida del modelo ya ajustada
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Guardar modelo entrenado
torch.save(model.state_dict(), model_path)
print(f"Modelo guardado en '{model_path}'")

# Evaluación
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == masks).sum().item()
        total += masks.numel()

print(f"Precisión en el conjunto de prueba: {100 * correct / total:.2f}%")
