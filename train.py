import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import CustomDataset
from model import ResNet18
import ray
from ray import tune
from ray import train

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset
dataset = CustomDataset(
    csv_file=os.path.abspath('labels.xlsx'),  
    root_dir=os.path.abspath('visionline/'),  
    transform=transform
)

def get_data_loader():
    return DataLoader(dataset, batch_size=32, shuffle=True)

def validate(model, data_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / total
    accuracy = correct / total
    model.train()
    return avg_loss, accuracy

def simpleTrain():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Instantiate the model
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
   
    data_loader = get_data_loader()
    num_epochs = 10

    for epoch in range(num_epochs):
        total_correct = 0
        total_samples = 0
        running_loss = 0.0
        
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total_samples
        epoch_accuracy = total_correct / total_samples
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

    print('Training finished.')    


def rayTuneTrain(config):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the model
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Training loop
    num_epochs = config["num_epochs"]
    data_loader = get_data_loader()

    for epoch in range(num_epochs):
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss, val_accuracy = validate(model, data_loader, device)
        train.report({"val_loss": val_loss,"val_accuracy": val_accuracy})
  
'''
a=ray.init( )
print(a)



# Define search space
search_space = {
    "lr": tune.grid_search([0.001, 0.0001]),
    "num_epochs": 10
}

trainable_with_resources = tune.with_resources(rayTuneTrain,
    resources= {"cpu": 1, "gpu": 1})

tunner = tune.Tuner(
    trainable_with_resources,
    param_space=search_space,
    
)

results = tunner.fit() #returns an ResultGrid object which has methods you can use for analyzing your training.
best_result = results.get_best_result(metric="val_loss", mode="min").config

'''
simpleTrain()






