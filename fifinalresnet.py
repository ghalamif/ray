import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_excel(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = self.annotations.iloc[index, 0]
        img_path = os.path.join(self.root_dir, img_name)
        img_path = os.path.normpath(img_path)
        
        print(f"Attempting to open image at path: {img_path}")
        
        if not os.path.isfile(img_path):
            print(f"File not found: {img_path}")
            raise FileNotFoundError(f"File not found: {img_path}")

        image = Image.open(img_path).convert('RGB')
        y_label = 1 if self.annotations.iloc[index, 2] == 'NOK' else 0

        if self.transform:
            image = self.transform(image)

        return image, y_label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = CustomDataset(
    csv_file='labels.xlsx',
    root_dir='visionline',
    transform=transform
)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)

num_classes = 2
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model.eval()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Training finished.')
