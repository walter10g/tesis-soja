import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import os

# Configuración
DATASET_DIR = "datasets/colors/images"  # Ruta al dataset
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
MODEL_SAVE_PATH = "model.pth"  # Ruta para guardar el modelo

# Semilla para reproducibilidad
torch.manual_seed(SEED)
if DEVICE == 'cuda':
    torch.cuda.manual_seed_all(SEED)

# Clase para agregar ruido gaussiano
class AddGaussianNoise(object):
    """Agrega ruido gaussiano a una imagen tensorial."""
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)

# Transformaciones de las imágenes
transform = transforms.Compose([
    transforms.ToTensor(),  # Convertir a tensor
    AddGaussianNoise(mean=0.0, std=0.05),  # Añadir ruido gaussiano
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizar
])

# Dataset y DataLoader
train_dataset = datasets.ImageFolder(root=os.path.join(DATASET_DIR, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(DATASET_DIR, 'val'), transform=transform)

# Reducir num_workers a 0 en Windows si hay problemas con multiprocessing
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Inicializar el modelo
def initialize_model(num_classes=4, device=DEVICE):
    # Cargar ResNet-18 con pesos preentrenados
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Reemplazar la capa final para clasificación
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(device)
    return model

model = initialize_model(num_classes=4, device=DEVICE)

# Configurar pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Entrenamiento del modelo
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, save_path):
    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        # Validación
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader.dataset):.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.4f}")

    # Guardar el modelo completo entrenado
    torch.save(model, save_path)
    print(f"Modelo completo guardado en {save_path}")

# Validación del modelo
def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    val_loss /= len(val_loader.dataset)
    accuracy = correct / total
    return val_loss, accuracy

# Ejecutar entrenamiento
if __name__ == "__main__":
    train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, DEVICE, MODEL_SAVE_PATH)
