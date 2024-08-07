import os
import tempfile
import pandas as pd
from PIL import Image

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from torchvision.transforms import ToTensor, Normalize, Compose

import ray
from ray import train
from ray.train.torch import TorchTrainer

# Custom dataset class
class VisionlineDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = 1 if self.labels.iloc[idx, 2] == "OK" else 0 
        if self.transform:
            image = self.transform(image)
        return image, label

def train_func():
    model = resnet18(num_classes=2)  
    # Model preparation for distributed training.
    model = train.torch.prepare_model(model)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Data
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data_dir = os.path.abspath('visionline/') 
    labels_csv = os.path.abspath('labels.csv')

    print('#######################')
    print('#######################')
    print('#######################')
    print('#######################')
    print(data_dir)
    print(labels_csv)
    print('#######################')
    print('#######################')
    print('#######################')
    print('#######################')

    

    train_data = VisionlineDataset(csv_file=labels_csv, root_dir=data_dir, transform=transform)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    # DataLoader
    train_loader = train.torch.prepare_data_loader(train_loader)

    # Training
    for epoch in range(10):
        if train.get_context().get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)

        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Report metrics and checkpoint
        metrics = {"loss": loss.item(), "epoch": epoch}
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(
                model.module.state_dict(),
                os.path.join(temp_checkpoint_dir, "model.pt")
            )
            train.report(
                metrics,
                checkpoint=train.Checkpoint.from_directory(temp_checkpoint_dir),
            )
        if train.get_context().get_world_rank() == 0:
            print(metrics)

runtime_env = {
    "pip": ["torch", "torchvision", "pandas"],
    "working_dir": "/srv/nfs/kube-ray",
}

ray.init(
    "ray://192.168.209.37:10001",
    runtime_env=runtime_env,
)

# Scaling config
scaling_config = train.ScalingConfig(num_workers=1, use_gpu=True)

# Launch distributed training job
trainer = TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    run_config=train.RunConfig(
        storage_path="/srv/nfs/kube-ray",
        name="faps",
       # failure_config=train.FailureConfig(-1) #for unlmited retries
    )
)
result = trainer.fit()

# Load the trained model
with result.checkpoint.as_directory() as checkpoint_dir:
    model_state_dict = torch.load(os.path.join(checkpoint_dir, "model.pt"))
    model = resnet18(num_classes=2)
    model.load_state_dict(model_state_dict)
