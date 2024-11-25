import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.metrics import f1_score
from model import SimpleCNN  # Importamos el modelo SimpleCNN
import numpy as np
import os
from collections import Counter
import multiprocessing

# Configuración general
num_epochs = 50
batch_size = 16
learning_rate = 0.0001
weight_decay = 5e-4
early_stopping_patience = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "simple_cnn_soja_model.pth"

# Forzar método de inicio seguro para Windows
multiprocessing.set_start_method("spawn", force=True)

# Transformaciones con data augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Directorios base
images_dir = "datasets/colors/images"

# Función principal
def main():
    # Dataset y DataLoader
    dataset = datasets.ImageFolder(images_dir, transform=transform)

    # Dividir el dataset en entrenamiento y prueba
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, dataset_size))

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    # Calcular pesos para el sampler
    class_counts = Counter(dataset.targets)
    class_weights = [1.0 / class_counts[label] for label in dataset.targets]

    # Ajustar los pesos para el subconjunto de entrenamiento
    train_targets = [dataset.targets[i] for i in train_indices]
    train_class_counts = Counter(train_targets)
    train_weights = [1.0 / train_class_counts[label] for label in train_targets]
    train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

    # DataLoaders con `num_workers=4` para mejor rendimiento
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Cargar modelo
    model = SimpleCNN(num_classes=4).to(device)

    if os.path.exists(model_path):
        print(f"Cargando modelo preexistente desde {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Entrenamiento desde cero.")

    # Función de entrenamiento
    def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, scheduler, num_epochs, best_model_path):
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            # Entrenamiento
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {running_loss / len(train_loader):.4f}")

            # Validación
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            all_labels = []
            all_predictions = []
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

            val_loss /= len(test_loader)
            accuracy = 100 * correct / total
            f1 = f1_score(all_labels, all_predictions, average="weighted")
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%, F1-Score: {f1:.4f}")

            # Guardar el mejor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Mejor modelo guardado en la época {epoch + 1} con precisión de {accuracy:.2f}%")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            scheduler.step()

            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping activado.")
                break

        return model

    # Configurar criterio, optimizador y scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=num_epochs)

    print("Fase 1: Entrenamiento inicial")
    model = train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, scheduler, num_epochs, model_path)

    print(f"Modelo final guardado en {model_path}")

# Proteger la ejecución del script principal
if __name__ == "__main__":
    main()
