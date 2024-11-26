import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from model import SimpleCNN
import os
import random
import matplotlib.pyplot as plt
import numpy as np

# Configuración general
num_epochs = 20  # Ajustado para memorizar rápidamente
batch_size = 16
learning_rate = 0.001  # Learning rate optimizado
weight_decay = 1e-4  # Regularización L2
early_stopping_patience = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "simple_cnn_good_model.pth"

# Directorio único para la categoría "good"
images_dir = "datasets/colors/images/"

# Transformaciones: Redimensionar a 8x8 y convertir a tensor
transform = transforms.Compose([
    transforms.Resize((8, 8)),  # Redimensionar a 8x8
    transforms.ToTensor()
])

# Función principal
def main():
    # Dataset para una sola categoría
    dataset = datasets.ImageFolder(root=images_dir, transform=transform)

    # Dividir el dataset en entrenamiento y prueba
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, dataset_size))

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Cargar modelo
    model = SimpleCNN().to(device)

    if os.path.exists(model_path):
        print(f"Cargando modelo preexistente desde {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Entrenamiento desde cero.")

    # Función de entrenamiento
    def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, num_epochs, best_model_path):
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            # Entrenamiento
            model.train()
            running_loss = 0.0
            for images, _ in train_loader:
                images = images.to(device)
                labels = torch.zeros(images.size(0), 1).to(device)  # Todas las etiquetas son 0

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {running_loss / len(train_loader):.4f}")

            # Validación
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, _ in test_loader:
                    images = images.to(device)
                    labels = torch.zeros(images.size(0), 1).to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(test_loader)
            print(f"Validation Loss: {val_loss:.4f}")

            # Guardar el mejor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Mejor modelo guardado en la época {epoch + 1}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping activado.")
                break

        return model

    # Configurar criterio y optimizador
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print("Fase 1: Entrenamiento inicial")
    model = train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, num_epochs, model_path)

    print(f"Modelo final guardado en {model_path}")

# Proteger la ejecución del script principal
if __name__ == "__main__":
    main()
