import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# Hiperparámetros
num_epochs = 10
batch_size = 2
learning_rate = 0.001

# Transformaciones enfocadas en color
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ajustar tamaño
    transforms.ToTensor(),
    # Solo normalizamos para colores RGB
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Cargar datasets
dataset = datasets.ImageFolder(root='app/datasets/train', transform=transform)

# Dividir datos en entrenamiento y prueba (80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Modelo ResNet preentrenado
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# Cambia el modelo para que tenga 4 salidas
model.fc = nn.Linear(model.fc.in_features, 4)  # 4 clases: bad, slightly_bad, slightly_good, good

# Ajusta la pérdida para trabajar con 4 clases
criterion = nn.CrossEntropyLoss()

# Opcional: imprimir las clases detectadas por ImageFolder
print(f"Clases detectadas: {dataset.classes}")  # ['bad', 'good', 'slightly_bad', 'slightly_good']

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Entrenamiento
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Guardar el modelo entrenado
torch.save(model.state_dict(), "modelo_soja.pth")
print("Modelo guardado en 'modelo_soja.pth'")

# Evaluación
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Precisión en el conjunto de prueba: {100 * correct / total:.2f}%")
