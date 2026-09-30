"""Thesis experiment: train a frozen ResNet-18 feature extractor."""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets", "final_dataset")
MODEL_PATH = os.path.join(BASE_DIR, "models", "resnet18_final_best.pth")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "final_run")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 30
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

full_dataset = datasets.ImageFolder(DATA_DIR)
class_names = full_dataset.classes
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
generator = torch.Generator().manual_seed(42)
train_indices, val_indices = random_split(
    range(len(full_dataset)), [train_size, val_size], generator=generator
)
torch.save(
    {"train_indices": train_indices.indices,
     "val_indices": val_indices.indices,
     "class_names": class_names},
    os.path.join(RESULTS_DIR, "split_indices.pt"),
)

train_data = datasets.ImageFolder(DATA_DIR, transform=train_transform)
val_data = datasets.ImageFolder(DATA_DIR, transform=val_transform)
train_loader = DataLoader(
    torch.utils.data.Subset(train_data, train_indices.indices),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
)
val_loader = DataLoader(
    torch.utils.data.Subset(val_data, val_indices.indices),
    batch_size=BATCH_SIZE, num_workers=0,
)

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
for parameter in model.parameters():
    parameter.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_correct = train_total = 0
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    model.eval()
    val_correct = val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            val_correct += (model(images).argmax(1) == labels).sum().item()
            val_total += labels.size(0)
    val_acc = 100 * val_correct / val_total
    print(
        f"Epoch {epoch + 1}/{EPOCHS} | Loss: {train_loss:.4f} | "
        f"Train Acc: {100 * train_correct / train_total:.2f}% | "
        f"Val Acc: {val_acc:.2f}%"
    )
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)

print("Best Validation Accuracy:", best_acc, "%")
