import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        """
        Modelo de red neuronal convolucional simple para clasificación de imágenes.
        
        Args:
            num_classes (int): Número de categorías de salida.
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
        
        # Capa fully connected
        self.fc1 = nn.Linear(128 * 28 * 28, 256)  # Asegúrate de que las dimensiones coincidan con el tamaño de entrada
        self.fc2 = nn.Linear(256, num_classes)

        # Dropout para evitar sobreajuste
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Paso forward a través de las capas convolucionales
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten para pasar a la capa fully connected
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def load_model(model_path, num_classes=4):
    """
    Carga un modelo SimpleCNN preentrenado si se proporciona un archivo de pesos.
    
    Args:
        model_path (str): Ruta al archivo de los pesos del modelo.
        num_classes (int): Número de categorías de salida.
    
    Returns:
        SimpleCNN: Modelo cargado.
    """
    model = SimpleCNN(num_classes=num_classes)
    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        print("Pesos del modelo cargados correctamente desde:", model_path)
    return model
