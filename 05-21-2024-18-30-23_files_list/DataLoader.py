import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class DataLoader:
    def __init__(self, image_dir, label_file_paths):
        self.image_dir = image_dir
        self.label_file_paths = label_file_paths
        self.labels_df = self._load_labels()
    
    def _load_labels(self):
        labels_df_list = [pd.read_excel(file) for file in self.label_file_paths]
        labels_df = pd.concat(labels_df_list, ignore_index=True)
        return labels_df
    
    def load_and_preprocess_images(self):
        images = []
        labels = []
        for index, row in self.labels_df.iterrows():
            image_path = os.path.join(self.image_dir, row['filename'])
            image = img_to_array(load_img(image_path, color_mode='grayscale'))
            image = image / 255.0
            images.append(image)
            labels.append(row['label'])
        return np.array(images), np.array(labels)

# Usage
image_dir = '/mnt/data/'
label_file_paths = ['/mnt/data/labels_optical_new.xlsx', '/mnt/data/labels_optical.xlsx']
data_loader = DataLoader(image_dir, label_file_paths)
images, labels = data_loader.load_and_preprocess_images()

print(f'Images shape: {images.shape}, Labels shape: {labels.shape}')
