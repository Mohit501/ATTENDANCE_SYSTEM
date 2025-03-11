import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import os
import numpy as np

# Define the Vision Transformer-based Face Recognition model
class FullViTModel(nn.Module):
    def __init__(self, num_classes=5):
        super(FullViTModel, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", trust_remote_code=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),  # Dropout layer
            nn.Linear(self.vit.config.hidden_size, num_classes)  # Fully connected layer
        )

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        cls_token_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.classifier(cls_token_output)
        return logits

# Data transformations for training and validation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224)),  # Random cropping
    transforms.RandomHorizontalFlip(),        # Random horizontal flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Color jitter
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load dataset
data_path = "dataset"  # Update with your dataset path
train_dataset = datasets.ImageFolder(os.path.join(data_path, "train"), transform=train_transform)
valid_dataset = datasets.ImageFolder(os.path.join(data_path, "valid"), transform=valid_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
y_train = np.array(train_dataset.targets)
if len(y_train) > 0:
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
else:
    class_weights = np.ones(len(train_dataset.classes))  # Default equal weights if empty

# Convert to tensor
class_weights = torch.tensor(class_weights, dtype=torch.float).to("cpu")

# Initialize model, criterion, and optimizer
num_classes = len(train_dataset.classes)
print(num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FullViTModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW([
    {"params": model.vit.parameters(), "lr": 1e-5},  # Fine-tune ViT
    {"params": model.classifier.parameters(), "lr": 1e-4}  # Train classifier
])

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Learning rate scheduler

# Mixed precision training setup
scaler = torch.cuda.amp.GradScaler()

# Training loop
def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=30):
    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100. * correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        valid_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        valid_accuracy = 100. * correct / total
        valid_losses.append(valid_loss / len(valid_loader))
        valid_accuracies.append(valid_accuracy)

        scheduler.step()

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_losses[-1]:.4f} | Train Accuracy: {train_accuracies[-1]:.2f}%")
        print(f"Valid Loss: {valid_losses[-1]:.4f} | Valid Accuracy: {valid_accuracies[-1]:.2f}%\n")

    # Plot training curves
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Valid Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(valid_accuracies, label="Valid Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.show()

# Train the model
train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=20)
print("Training completed")

# Save the trained model
torch.save(model.state_dict(), "vit_full_model.pth")
