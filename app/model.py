import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

# Cargar modelo entrenado
class SimpleResNet(nn.Module):
    def __init__(self):
        super(SimpleResNet, self).__init__()
        self.fc = nn.Linear(512, 2)  # Ajusta según tu modelo

    def forward(self, x):
        return self.fc(x)

# Cargar el modelo
model = SimpleResNet()
model.load_state_dict(torch.load("modelo_soja.pth", map_location=torch.device("cpu")))
model.eval()

# Transformaciones para imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_color(image_data: bytes):
    image = Image.open(io.BytesIO(image_data))
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return "Good" if predicted.item() == 0 else "Bad"
