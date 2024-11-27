import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import create_deeplab_model  # Define tu modelo en este archivo
from dataset import SegmentationDataset  # Define el dataset personalizado en este archivo
import os

def train_model(model, dataloaders, criterion, optimizer, num_epochs=10, device="cpu"):
    """
    Función para entrenar un modelo de segmentación.
    """
    best_model_wts = model.state_dict()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)["out"]  # Salida de DeepLabV3
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f}")

            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()

    print(f"Best val Loss: {best_loss:.4f}")
    model.load_state_dict(best_model_wts)
    return model

def main():
    # Configuración
    image_dir = "datasets/colors/images/train"
    mask_dir = "datasets/masks/train"
    val_image_dir = "datasets/colors/images/val"
    val_mask_dir = "datasets/masks/val"
    batch_size = 16
    num_epochs = 50
    learning_rate = 0.001
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Validar que las carpetas de imágenes y máscaras no estén vacías
    for dir_path, dir_type in zip(
        [image_dir, mask_dir, val_image_dir, val_mask_dir],
        ["train images", "train masks", "val images", "val masks"]
    ):
        if not os.path.exists(dir_path) or len(os.listdir(dir_path)) == 0:
            raise FileNotFoundError(f"La carpeta '{dir_type}' está vacía o no existe: {dir_path}")

    # Transformaciones
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    mask_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        lambda x: (x * 255).long()  # Convertir máscara a índices enteros
    ])

    # Datasets personalizados
    train_dataset = SegmentationDataset(
        image_dir=image_dir, mask_dir=mask_dir,
        transform=data_transforms, mask_transform=mask_transforms
    )
    val_dataset = SegmentationDataset(
        image_dir=val_image_dir, mask_dir=val_mask_dir,
        transform=data_transforms, mask_transform=mask_transforms
    )

    # Mostrar cantidad de datos cargados
    print(f"Cantidad de datos de entrenamiento: {len(train_dataset)}")
    print(f"Cantidad de datos de validación: {len(val_dataset)}")

    # DataLoaders
    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),  # num_workers=0 para evitar errores en Windows
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
    }

    # Crear modelo DeepLab
    num_classes = 5  # Clases: fondo + (bad, good, slightly_bad, slightly_good)
    model = create_deeplab_model(num_classes=num_classes)
    model = model.to(device)

    # Definir función de pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Entrenar modelo
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs, device=device)

    # Guardar modelo completo
    torch.save(model, "deeplab_complete_model.pth")
    print("Modelo completo guardado como 'deeplab_complete_model.pth'.")

if __name__ == "__main__":
    main()
