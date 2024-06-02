import os
import torch
import cv2

# Assuming you have a function to load your model
def load_model(model_path):
    try:
        # Load the model
        model = torch.load(model_path)
        # Ensure that the loaded object is a PyTorch model
        if not isinstance(model, torch.nn.Module):
            raise TypeError("The loaded object is not a PyTorch model.")
        # Set model to evaluation mode
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Assuming you have a function to load your data
def load_data(data_folder):
    try:
        images = []
        labels = []
        for filename in os.listdir(data_folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(data_folder, filename)
                images.append(cv2.imread(image_path))
                label_path = os.path.join(data_folder, filename.replace(".jpg", ".txt").replace(".png", ".txt"))
                with open(label_path, "r") as file:
                    label = file.read().strip()  # Assuming the label is in the text file
                    labels.append(label)
        return images, labels
    except Exception as e:
        print(f"Error loading data: {e}")
        return [], []

def calculate_accuracy(model, images, labels):
    try:
        correct = 0
        total = 0
        for image, label in zip(images, labels):
            # Convert image to PyTorch tensor and perform necessary preprocessing
            # (e.g., normalization, resizing) depending on the model's input requirements
            # Make predictions using the model
            predicted_label = model(image)
            # Compare predicted label with ground truth label
            if predicted_label == label:
                correct += 1
            total += 1
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    except Exception as e:
        print(f"Error calculating accuracy: {e}")
        return 0.0

def main():
    try:
        # Load the model
        model_path = "best.pt"
        model = load_model(model_path)
        if model is None:
            print("Failed to load model. Exiting.")
            return

        # Load images and labels
        valid_folder = r"C:\Users\Jahnavi\Documents\TDL\valid"
        images, labels = load_data(valid_folder)

        if not images or not labels:
            print("No data loaded. Exiting.")
            return

        # Calculate accuracy
        accuracy = calculate_accuracy(model, images, labels)
        print(f"Accuracy: {accuracy}")
    except Exception as e:
        print(f"Error in main function: {e}")

if __name__ == "__main__":
    main()
