import os
import pandas as pd

# Define the paths
excel_path = 'labels_optical.xlsx'  # Path to your Excel file
images_directory = 'visionline'  # Path to your images directory
output_excel_path = 'labels.xlsx'  # Path for the output Excel file

# Read the Excel file
df = pd.read_excel(excel_path)

# List all images in the directory
image_files = os.listdir(images_directory)

# Create a DataFrame from the image files list
images_df = pd.DataFrame(image_files, columns=['Image'])
images_df['Part ID'] = images_df['Image']  # Assume 'Part ID' includes the file extension


# Inner join with the original DataFrame on 'Part ID'
merged_df = pd.merge(df, images_df, on='Part ID', how='inner')
print(merged_df)
# Save the updated DataFrame to the new Excel file
merged_df.to_excel(output_excel_path, index=False)

print(f"Joined data has been written to {output_excel_path}")
