import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        """
        Modelo de red neuronal convolucional simple para clasificación binaria.
        """
        super(SimpleCNN, self).__init__()
        
        # Capas convolucionales
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Batch normalization para estabilizar el entrenamiento
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Capas totalmente conectadas
        self.fc1 = nn.Linear(128 * 1 * 1, 256)  # Compatible con imágenes de 8x8
        self.fc2 = nn.Linear(256, 1)  # Salida binaria

        # Dropout para evitar sobreajuste
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Paso forward a través de las capas convolucionales
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten para pasar a las capas fully connected
        x = torch.flatten(x, 1)

        # Paso por las capas fully connected
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def load_model(model_path):
    """
    Carga un modelo SimpleCNN preentrenado si se proporciona un archivo de pesos.
    """
    model = SimpleCNN()
    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        print("Pesos del modelo cargados correctamente desde:", model_path)
    return model
