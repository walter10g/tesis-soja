import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import os

# Configuración general
num_epochs = 10
batch_size = 4
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rutas del dataset
aerial_images_path = "datasets/aerial"  # Imágenes capturadas por dron
ground_images_path = "datasets/ground"  # Imágenes cercanas (nivel suelo)

# Transformaciones para imágenes cercanas (simulación aérea)
ground_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Reducción de resolución
    transforms.GaussianBlur(kernel_size=(5, 5)),  # Simular desenfoque de vista aérea
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Variaciones de color
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Transformaciones para imágenes aéreas
aerial_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ajustar tamaño
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset personalizado para combinar imágenes aéreas y cercanas
class CombinedDataset(Dataset):
    def __init__(self, aerial_path, ground_path, aerial_transform, ground_transform):
        self.aerial_images = datasets.ImageFolder(root=aerial_path, transform=aerial_transform)
        self.ground_images = datasets.ImageFolder(root=ground_path, transform=ground_transform)

        # Combinar ambas listas de imágenes
        self.data = self.aerial_images.samples + self.ground_images.samples
        self.classes = self.aerial_images.classes  # Las mismas clases para ambos datasets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert("RGB")

        # Aplicar la transformación correcta
        if "aerial" in image_path:
            image = aerial_transform(image)
        else:
            image = ground_transform(image)

        return image, label

# Cargar dataset combinado
dataset = CombinedDataset(
    aerial_path=aerial_images_path,
    ground_path=ground_images_path,
    aerial_transform=aerial_transform,
    ground_transform=ground_transform
)

# Dividir en entrenamiento y prueba (80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Modelo ResNet preentrenado
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 4)  # 4 clases: bad, slightly_bad, slightly_good, good
model = model.to(device)

# Función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Entrenamiento del modelo
print("Iniciando entrenamiento...")
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Guardar el modelo entrenado
torch.save(model.state_dict(), "modelo_soja_adaptado.pth")
print("Modelo guardado en 'modelo_soja_adaptado.pth'")

# Evaluación del modelo
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Precisión en el conjunto de prueba: {100 * correct / total:.2f}%")
