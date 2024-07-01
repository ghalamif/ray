import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset

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
        
        if not os.path.isfile(img_path):
            print(f"File not found: {img_path}") 
            raise FileNotFoundError(f"File not found: {img_path}") 
           

        image = Image.open(img_path).convert('RGB') 
        y_label = 1 if self.annotations.iloc[index, 2] == 'NOK' else 0

        if self.transform:
            image = self.transform(image)

        return image, y_label
