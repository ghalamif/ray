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
#from ray.air.config import ScalingConfig

@ray.remote(num_gpus=1)
class Trainable:
    def __init__(self, model_class, dataset_class, csv_file, root_dir, batch_size=32, lr=0.0001, num_epochs=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_class().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.dataset = dataset_class(csv_file=csv_file, root_dir=root_dir, transform=self.transform)
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def train_epoch(self):
        self.model.train()
        total_correct = 0
        total_samples = 0
        running_loss = 0.0

        for images, labels in self.data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_accuracy = total_correct / total_samples

        return epoch_loss, epoch_accuracy

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_loss = total_loss / total
        accuracy = correct / total
        self.model.train()
        return avg_loss, accuracy

    def simple_train(self):
        for epoch in range(self.num_epochs):
            epoch_loss, epoch_accuracy = self.train_epoch()
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
        print('Training finished.')

    def ray_tune_train(self, config):
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["lr"])
        num_epochs = config["num_epochs"]
        for epoch in range(num_epochs):
            self.train_epoch()
            val_loss, val_accuracy = self.validate()
            train.report({"val_loss": val_loss, "val_accuracy": val_accuracy})

    def bestTune(self):
        
        # Define search space
        search_space = {
            "lr": tune.grid_search([0.001, 0.0001]),
            "num_epochs": 10
        }

        trainable_with_resources = tune.with_resources(
            self.ray_tune_train,
            resources= {"cpu": 1, "gpu": 1})

        tunner = tune.Tuner(
            trainable_with_resources,
            param_space=search_space,
            
        )

        results = tunner.fit() #returns an ResultGrid object which has methods you can use for analyzing your training.
        best_result = results.get_best_result(metric="val_loss", mode="min").config
        return best_result, results
'''
csv_file = os.path.abspath('labels.xlsx')
root_dir = os.path.abspath('visionline/')

trainable = Trainable(ResNet18, CustomDataset, csv_file, root_dir)

context = ray.init()
print(context.dashboard_url)

ray.get(trainable.simple_train.remote())


# Define search space
search_space = {
    "lr": tune.grid_search([0.001, 0.0001]),
    "num_epochs": 10
}

trainable_with_resources = tune.with_resources(trainable.ray_tune_train,
    resources= {"cpu": 1, "gpu": 1})

tunner = tune.Tuner(
    trainable_with_resources,
    param_space=search_space,
    
)

results = tunner.fit() #returns an ResultGrid object which has methods you can use for analyzing your training.
best_result = results.get_best_result(metric="val_loss", mode="min").config
print(best_result)

 '''
csv_file = '/mnt/data/labels.xlsx'
root_dir = '/mnt/data/visionline/'

ray.init(ignore_reinit_error=True, address='auto')

'''
from ray.autoscaler import ScalingConfig
scaling_config = ScalingConfig(
    num_workers=4,
    use_gpu=True,
    #num_cpus_per_worker=1,
    #num_gpus_per_worker=1
)
'''

print(ray.is_initialized())
print(ray.available_resources())
print (ray.cluster_resources())
print(ray.cluster_resources()['CPU'])
print(ray.cluster_resources()['GPU'])



trainable = Trainable.remote(ResNet18, CustomDataset, csv_file, root_dir)
print(ray.get(trainable.bestTune.remote()))
ray.shutdown()