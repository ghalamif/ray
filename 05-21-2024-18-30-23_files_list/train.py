import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
from model import ResNet18

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 0.001
batch_size = 32
num_epochs = 10
num_classes = 10  # Adjust based on your dataset

# Load Data
annotations_file = 'labels_optical_new.xlsx'
img_dir = 'visionline'
dataloader = get_dataloader(annotations_file, img_dir, batch_size)

# Initialize network
model = ResNet18(num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.to(device)
        targets = targets.to(device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'resnet_model.pth')
