import torch
import torch.nn as nn
from torchvision import models


class ResNet18SegmentationModel(nn.Module):
    def __init__(self, num_classes=4, freeze_backbone=True):
        """
        Modelo de segmentación basado en ResNet-18 para clasificar imágenes en múltiples categorías.

        Args:
            num_classes (int): Número de categorías de salida. (Default: 4)
            freeze_backbone (bool): Si True, congela las capas del backbone preentrenado. (Default: True)
        """
        super(ResNet18SegmentationModel, self).__init__()
        # Cargar ResNet-18 preentrenado
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Extraer el backbone sin la capa de clasificación
        self.backbone = nn.Sequential(*list(resnet18.children())[:-1])  # Hasta la capa de pooling global
        
        # Congelar el backbone si se especifica
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Crear un clasificador personalizado
        in_features = resnet18.fc.in_features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),  # Reducir dimensionalidad
            nn.ReLU(),
            nn.Dropout(0.5),             # Regularización
            nn.Linear(256, num_classes)  # Salida final para las clases (4 categorías)
        )

    def forward(self, x):
        """
        Realiza el paso forward del modelo.
        
        Args:
            x (torch.Tensor): Tensor de entrada con tamaño [batch_size, channels, height, width].
        
        Returns:
            torch.Tensor: Tensor de salida con las probabilidades de cada clase.
        """
        # Pasar los datos a través del backbone y luego al clasificador
        x = self.backbone(x)
        x = self.classifier(x)
        return x


def load_model(model_path, num_classes=4, freeze_backbone=True):
    """
    Carga un modelo ajustado para clasificación multicategoría con ResNet-18.

    Args:
        model_path (str): Ruta al archivo de los pesos del modelo.
        num_classes (int): Número de categorías de salida. (Default: 4)
        freeze_backbone (bool): Si True, congela las capas del backbone preentrenado. (Default: True)
    
    Returns:
        ResNet18SegmentationModel: Modelo cargado con los pesos entrenados.
    """
    # Inicializar modelo
    model = ResNet18SegmentationModel(num_classes=num_classes, freeze_backbone=freeze_backbone)
    
    # Cargar pesos desde el archivo
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    # Ajustar claves si el checkpoint incluye prefijos adicionales
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    print("Pesos del modelo cargados correctamente.")
    return model
