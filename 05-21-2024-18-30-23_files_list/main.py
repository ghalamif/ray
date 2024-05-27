from CNNModel import CNNModel
from DataLoader import DataLoader


def main():
    image_dir = 'visionline'
    label_file_paths = ['labels_optical_new.xlsx', 'labels_optical.xlsx']

    # Load and preprocess data
    data_loader = DataLoader(image_dir, label_file_paths)
    images, labels = data_loader.load_and_preprocess_images()

    # Define input shape for the CNN
    input_shape = (images.shape[1], images.shape[2], 1)

    # Initialize and compile the CNN model
    cnn_model = CNNModel(input_shape)
    cnn_model.compile_model()

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Train the model
    cnn_model.train_model(X_train, y_train, X_val, y_val)

    # Evaluate the model
    cnn_model.evaluate_model(X_val, y_val)

    # Plot training history
    cnn_model.plot_training_history()

if __name__ == "__main__":
    main()
