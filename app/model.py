import torch
from torchvision import models
import torch.nn as nn

def load_model(model_path, num_classes=4):
    """
    Carga el modelo entrenado para inferencia.
    """
    # Cargar el modelo ResNet-18
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Cargar los pesos entrenados
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # Modo evaluación
    return model
