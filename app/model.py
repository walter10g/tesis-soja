import torch
from torchvision import models
import torch.nn as nn
from torchvision.models.segmentation import FCN_ResNet50_Weights

class SegmentationModel(nn.Module):
    def __init__(self, num_classes=4):
        super(SegmentationModel, self).__init__()
        # Usar FCN con ResNet-50 como backbone
        weights = FCN_ResNet50_Weights.DEFAULT
        self.model = models.segmentation.fcn_resnet50(weights=weights)
        # Ajustar la salida del modelo para el número de clases
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, dict) and 'out' in output:
            return output['out']
        return output  # Si devuelve directamente un tensor


def load_model(model_path, num_classes=4):
    """
    Carga un modelo de segmentación con los pesos entrenados.
    """
    # Cargar modelo preentrenado con pesos predeterminados
    weights = FCN_ResNet50_Weights.DEFAULT
    model = models.segmentation.fcn_resnet50(weights=weights)
    
    # Ajustar el clasificador para que coincida con el número de clases
    model.classifier[4] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1))
    
    # Cargar los pesos ajustados
    state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    
    # Ajustar claves si es necesario
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    
    # Cargar los pesos al modelo
    model.load_state_dict(state_dict, strict=False)
    print("Pesos del modelo cargados correctamente.")
    return model
