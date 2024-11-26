import torch
import torch.nn as nn
from torchvision import models

class CropClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(CropClassifier, self).__init__()
        # Cargamos ResNet-18 preentrenado
        self.resnet = models.resnet18(pretrained=True)
        
        # Reemplazamos la capa completamente conectada final para que coincida con nuestras clases
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
        # Activación final (Softmax o CrossEntropy se maneja en la pérdida)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.resnet(x)
        return self.softmax(x)

# Función para inicializar el modelo
def initialize_model(num_classes=4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = CropClassifier(num_classes=num_classes)
    model = model.to(device)
    return model
