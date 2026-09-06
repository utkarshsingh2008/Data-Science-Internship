# 1. Dataset Loading & Initial Exploration

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

print("PyTorch Version:", torch.__version__)

transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

print("\nDataset Information:")
print("Training samples:", len(train_dataset))
print("Testing samples:", len(test_dataset))
print("Number of classes:", len(train_dataset.classes))
print("Classes:", train_dataset.classes)

# Display sample images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    image, label = train_dataset[i]
    ax.imshow(image.squeeze(), cmap="gray")
    ax.set_title(f"Label: {label}")
    ax.axis("off")

plt.tight_layout()
#plt.show()

# 2. Class Distribution Analysis

from collections import Counter

train_labels = [label for _, label in train_dataset]

class_counts = Counter(train_labels)

print("\nClass Distribution:")

for digit in range(10):
    print(f"Digit {digit}: {class_counts[digit]} samples")

# Plot class distribution
plt.figure(figsize=(8, 5))

plt.bar(
    class_counts.keys(),
    class_counts.values()
)

plt.xlabel("Digit Class")
plt.ylabel("Number of Samples")
plt.title("MNIST Training Dataset - Class Distribution")
plt.xticks(range(10))

#plt.show()

# 3. Image & Pixel Data Analysis

image, label = train_dataset[0]

print("\nImage Information:")
print("Image Shape:", image.shape)
print("Minimum Pixel Value:", image.min().item())
print("Maximum Pixel Value:", image.max().item())
print("Average Pixel Value:", image.mean().item())

# Display one image with pixel values
plt.figure(figsize=(5, 5))

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Sample Image - Label: {label}")
plt.colorbar(label="Pixel Intensity")
plt.axis("off")

#plt.show()

# 4. Data Preprocessing

print("\nData Preprocessing Information:")

print("Training Image Shape:", train_dataset.data.shape)
print("Testing Image Shape:", test_dataset.data.shape)

image, label = train_dataset[0]

print("Tensor Shape:", image.shape)
print("Pixel Value Range:", image.min().item(), "to", image.max().item())


# 5. Train/Validation Split

from torch.utils.data import random_split

train_size = int(0.8 * len(train_dataset))
validation_size = len(train_dataset) - train_size

train_data, validation_data = random_split(
    train_dataset,
    [train_size, validation_size],
    generator=torch.Generator().manual_seed(42)
)

print("\nDataset Split:")
print("Training samples:", len(train_data))
print("Validation samples:", len(validation_data))
print("Testing samples:", len(test_dataset))

# 6. Create DataLoaders

from torch.utils.data import DataLoader

batch_size = 64

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)

validation_loader = DataLoader(
    validation_data,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

print("\nDataLoader Information:")
print("Batch Size:", batch_size)
print("Training Batches:", len(train_loader))
print("Validation Batches:", len(validation_loader))
print("Testing Batches:", len(test_loader))

# 7. CNN Architecture Design

import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = CNNModel()

print(model)

# 8. Model Summary & Parameter Count

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("\nModel Information:")
print("Total Trainable Parameters:", total_parameters)

# 9. Define Loss Function & Optimizer

import torch.optim as optim

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

print("\nTraining Configuration:")
print("Loss Function: CrossEntropyLoss")
print("Optimizer: Adam")
print("Learning Rate: 0.001")

# 10. Model Training


num_epochs = 10

train_losses = []
train_accuracies = []
validation_losses = []
validation_accuracies = []

for epoch in range(num_epochs):

    # Training
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    # Validation
    model.eval()

    validation_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            outputs = model(images)
            loss = criterion(outputs, labels)

            validation_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    validation_loss = validation_loss / len(validation_loader)
    validation_accuracy = 100 * correct / total

    # Store results
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    validation_losses.append(validation_loss)
    validation_accuracies.append(validation_accuracy)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f}, "
        f"Train Accuracy: {train_accuracy:.2f}%, "
        f"Validation Loss: {validation_loss:.4f}, "
        f"Validation Accuracy: {validation_accuracy:.2f}%"
    )
    
torch.save(model.state_dict(), "mnist_cnn_model.pth")

print("\nTrained model saved successfully.")

    # 11. Training & Validation Performance

plt.figure(figsize=(10, 5))

plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(validation_accuracies, label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)

plt.show()

# 12. Training & Validation Loss

plt.figure(figsize=(10, 5))

plt.plot(train_losses, label="Training Loss")
plt.plot(validation_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)

plt.show()

# 13. Test Set Evaluation

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total

print("\nTest Set Evaluation:")
print("Correct Predictions:", correct)
print("Total Test Samples:", total)
print(f"Test Accuracy: {test_accuracy:.2f}%")

# 14. Confusion Matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns

model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted.numpy())

cm = confusion_matrix(all_labels, all_predictions)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=range(10),
    yticklabels=range(10)
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("MNIST Confusion Matrix")

plt.show()

# 15. Error Analysis

model.eval()

incorrect_images = []
incorrect_labels = []
incorrect_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        for i in range(len(labels)):

            if predicted[i] != labels[i]:

                incorrect_images.append(images[i])
                incorrect_labels.append(labels[i].item())
                incorrect_predictions.append(predicted[i].item())

print("\nError Analysis:")
print("Total Incorrect Predictions:", len(incorrect_images))

# Display first 10 incorrect predictions
num_images = min(10, len(incorrect_images))

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i in range(num_images):

    axes.flat[i].imshow(
        incorrect_images[i].squeeze(),
        cmap="gray"
    )

    axes.flat[i].set_title(
        f"Actual: {incorrect_labels[i]}\n"
        f"Predicted: {incorrect_predictions[i]}"
    )

    axes.flat[i].axis("off")

plt.tight_layout()
plt.show()

# 16. Overfitting Analysis

final_train_accuracy = train_accuracies[-1]
final_validation_accuracy = validation_accuracies[-1]

accuracy_gap = final_train_accuracy - final_validation_accuracy

print("\nOverfitting Analysis:")
print(f"Final Training Accuracy: {final_train_accuracy:.2f}%")
print(f"Final Validation Accuracy: {final_validation_accuracy:.2f}%")
print(f"Accuracy Gap: {accuracy_gap:.2f}%")

if accuracy_gap > 5:
    print("Observation: The model shows signs of overfitting.")
else:
    print("Observation: No significant overfitting is observed.")

# 17. Improved CNN Model

class ImprovedCNNModel(nn.Module):
    def __init__(self):
        super(ImprovedCNNModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


improved_model = ImprovedCNNModel()

print("\nImproved CNN Architecture:")
print(improved_model)

# 18. Improved Model Training & Comparison

improved_criterion = nn.CrossEntropyLoss()

improved_optimizer = optim.Adam(
    improved_model.parameters(),
    lr=0.001
)

improved_epochs = 5

for epoch in range(improved_epochs):

    improved_model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        improved_optimizer.zero_grad()

        outputs = improved_model(images)
        loss = improved_criterion(outputs, labels)

        loss.backward()
        improved_optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{improved_epochs}] "
        f"Training Accuracy: {train_accuracy:.2f}%"
    )


# Evaluate improved model
improved_model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = improved_model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

improved_test_accuracy = 100 * correct / total

print("\nModel Comparison:")
print(f"Original CNN Accuracy: {test_accuracy:.2f}%")
print(f"Improved CNN Accuracy: {improved_test_accuracy:.2f}%")

# 19. Final Evaluation Metrics

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

improved_model.eval()

final_labels = []
final_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        outputs = improved_model(images)
        _, predicted = torch.max(outputs, 1)

        final_labels.extend(labels.numpy())
        final_predictions.extend(predicted.numpy())

accuracy = accuracy_score(final_labels, final_predictions)

precision = precision_score(
    final_labels,
    final_predictions,
    average="weighted"
)

recall = recall_score(
    final_labels,
    final_predictions,
    average="weighted"
)

f1 = f1_score(
    final_labels,
    final_predictions,
    average="weighted"
)

print("\nFinal Model Evaluation:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-Score: {f1 * 100:.2f}%")

# 20. Classification Report

from sklearn.metrics import classification_report

print("\nClassification Report:")

print(
    classification_report(
        final_labels,
        final_predictions,
        target_names=[str(i) for i in range(10)]
    )
)

# 21. Final Confusion Matrix

final_cm = confusion_matrix(
    final_labels,
    final_predictions
)

print("\nFinal Confusion Matrix:")
print(final_cm)

plt.figure(figsize=(10, 8))

sns.heatmap(
    final_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=range(10),
    yticklabels=range(10)
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Final CNN Confusion Matrix")

plt.show()

# 22. Final Error Analysis

incorrect_images = []
incorrect_labels = []
incorrect_predictions = []

for i in range(len(final_labels)):

    if final_labels[i] != final_predictions[i]:

        image, _ = test_dataset[i]

        incorrect_images.append(image)
        incorrect_labels.append(final_labels[i])
        incorrect_predictions.append(final_predictions[i])

print("\nFinal Error Analysis:")
print("Total Incorrect Predictions:", len(incorrect_images))

# Display incorrect predictions
num_images = min(10, len(incorrect_images))

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i in range(num_images):

    axes.flat[i].imshow(
        incorrect_images[i].squeeze(),
        cmap="gray"
    )

    axes.flat[i].set_title(
        f"Actual: {incorrect_labels[i]}\n"
        f"Predicted: {incorrect_predictions[i]}"
    )

    axes.flat[i].axis("off")

plt.tight_layout()
plt.show()