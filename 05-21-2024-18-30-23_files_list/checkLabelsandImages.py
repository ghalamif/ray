import os
import pandas as pd

# File paths
annotations_file = 'labels_optical_new.xlsx'
img_dir = 'visionline'

# Load the Excel file
img_labels = pd.read_excel(annotations_file)

# Get the list of images from the directory
available_images = set(os.listdir(img_dir))

# Get the list of images from the Excel file
listed_images = set(img_labels.iloc[:, 0])

# Find missing images
missing_images = listed_images - available_images

# Output the result
if missing_images:
    print("Missing images:")
    for img in missing_images:
        print(img)
else:
    print("All images are available.")
