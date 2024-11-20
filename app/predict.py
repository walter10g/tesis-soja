import torch
from torchvision import transforms, models
from PIL import Image

# Cargar el modelo entrenado
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Dos clases: Good y Bad
model.load_state_dict(torch.load("modelo_soja.pth", map_location=torch.device("cpu")))
model.eval()

# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Función para realizar predicción
def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Añade la dimensión de batch
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return "Good" if predicted.item() == 0 else "Bad"

# Prueba con una imagen
if __name__ == "__main__":
    # Cambia "test_good.jpg" por el nombre de tu imagen de prueba
    image_path = "app/test_images/test_good.jpg"
    result = predict_image(image_path)
    print(f"La predicción para {image_path} es: {result}")
