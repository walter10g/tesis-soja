import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

def create_deeplab_model(num_classes):
    """
    Crea un modelo DeepLabV3 basado en ResNet-50 con PyTorch.
    """
    # Cargar el modelo preentrenado
    model = deeplabv3_resnet50(pretrained=True)

    # Modificar la última capa de clasificación para ajustarse al número de clases
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    # Inicializar pesos de la nueva capa
    nn.init.xavier_normal_(model.classifier[4].weight)
    if model.classifier[4].bias is not None:
        nn.init.zeros_(model.classifier[4].bias)

    return model


if __name__ == "__main__":
    # Configuración de prueba
    num_classes = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Crear el modelo
    model = create_deeplab_model(num_classes=num_classes)
    model = model.to(device)

    # Resumen del modelo (mostrar número de parámetros)
    print("Modelo DeepLabV3 configurado para PyTorch:")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total de parámetros entrenables: {total_params}")
