import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50

# Define paths
image_folder = 'visionline/'
labels_file_path = 'labels_optical_new.xlsx'

# Load labels
labels_df = pd.read_excel(labels_file_path)
labels_df['Class'] = labels_df['Class'].apply(lambda x: 1 if x == 'OK' else 0)

# Filter out entries with missing files
existing_files = labels_df['Part ID'].apply(lambda x: os.path.isfile(os.path.join(image_folder, x)))
labels_df = labels_df[existing_files]

if labels_df.empty:
    raise ValueError("No valid images found. Please check your dataset.")

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, labels_df, image_folder, transform=None):
        self.labels_df = labels_df.reset_index(drop=True)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.labels_df.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.labels_df.iloc[idx, 4]
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Split data into training and validation sets
train_df, val_df = train_test_split(labels_df, test_size=0.2, random_state=42)

# Create datasets
train_dataset = CustomDataset(train_df, image_folder, transform=transform)
val_dataset = CustomDataset(val_df, image_folder, transform=transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the ResNet model without pre-trained weights
model = resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            predicted = (torch.sigmoid(outputs) > 0.5).int()
            total += labels.size(0)
            correct += (predicted == labels.int()).sum().item()

    val_accuracy = correct / total
    print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
        outputs = model(inputs)
        predicted = (torch.sigmoid(outputs) > 0.5).int()
        total += labels.size(0)
        correct += (predicted == labels.int()).sum().item()

val_accuracy = correct / total
print(f'Final Validation Accuracy: {val_accuracy * 100:.2f}%')
