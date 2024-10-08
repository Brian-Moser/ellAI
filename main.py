import os
import cv2
import torch
import argparse
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from torchvision import datasets, transforms
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import nn, optim
from PIL import Image

def get_last_conv_layer(model):
    layers = list(model.children())
    while layers:
        layer = layers.pop()
        if isinstance(layer, nn.Conv2d):
            return layer
        elif isinstance(layer, nn.Sequential):
            layers.extend(layer.children())
    raise ValueError("No convolutional layer found in the model.")


def generate_CAM(model, test_image, class_idx):
    # Set up hooks to capture gradients and feature maps
    gradients = []
    feature_maps = []

    def save_gradient(grad):
        gradients.append(grad)

    def save_feature_map(module, input, output):
        feature_maps.append(output)

    # Register hooks on the last convolutional layer
    last_conv_layer = get_last_conv_layer(model)
    last_conv_layer.register_forward_hook(save_feature_map)
    last_conv_layer.register_backward_hook(lambda module, grad_in, grad_out: save_gradient(grad_out[0]))

    # Ensure input requires gradients
    test_image.requires_grad = True

    # Forward pass
    model.eval()  # Set model to eval mode
    outputs = model(test_image)
    
    # Create one-hot output for the predicted class
    one_hot_output = torch.zeros(outputs.size()).to(test_image.device)
    one_hot_output[0][class_idx] = 1
    
    # Zero gradients and backward pass
    model.zero_grad()
    outputs.backward(gradient=one_hot_output)

    # Get the gradients and feature map
    gradients = gradients[0]  # Get the captured gradient
    feature_map = feature_maps[0]  # Get the captured feature map

    # Calculate weights and CAM
    weights = gradients.mean(dim=[1, 2], keepdim=True)
    cam = (weights * feature_map).sum(dim=1).squeeze()

    # Process the CAM
    cam = F.relu(cam)  # ReLU
    cam /= cam.max()  # Normalize
    cam = cam.cpu().detach().numpy()
    cam = cv2.resize(cam, (256, 256))  # Resize to match input size
    cam = (cam * 255).astype(np.uint8)

    return cam



def save_CAM(cam, original_image_path, output_path):
    cam_colored = cv2.applyColorMap(cam, cv2.COLORMAP_JET)  # Apply color map
    original_image = cv2.imread(original_image_path)
    original_image = cv2.resize(original_image, (256, 256))  # Resize to match CAM

    # Combine original image and CAM
    combined = cv2.addWeighted(original_image, 0.5, cam_colored, 0.5, 0)
    
    output_file_name = output_path.replace(".jpg", "_finetuning_CAM.jpg")
    cv2.imwrite(output_file_name, combined)
    print(f"Class Activation Map saved to {output_file_name}")


def get_regular_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

def load_dataset(data_dir):
    return datasets.ImageFolder(root=data_dir, transform=get_regular_transforms())

def load_image(image_path):
    return get_regular_transforms()(Image.open(image_path))

def train_knn(features, labels, n_neighbors=3):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(features, labels)
    return knn

def predict_knn(knn, test_features):
    prediction = knn.predict([test_features])
    probabilities = knn.predict_proba([test_features])
    return prediction, probabilities

def get_pretrained_features(dataset, model):
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the final layer
    model.eval()

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    features = []
    labels = []

    with torch.no_grad():
        for images, label in data_loader:
            output = model(images)
            output = output.flatten(start_dim=1).numpy()  # Flatten and move to CPU
            features.extend(output)
            labels.extend(label.numpy())

    return zip(features, labels)

def replace_last_layer(model, num_classes):
    if hasattr(model, 'fc'):  # For ResNet and similar architectures
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):  # For MobileNet, EfficientNet
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Model architecture not supported for fine-tuning: {type(model)}")
    return model

def train_finetuned_model(dataset, model, device, epochs=5):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Set model to training mode
    model.train()
    model.to(device)

    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")
    
    return model

def save_model(model, dataset_path, model_name):
    os.makedirs("model_weights", exist_ok=True)
    model_save_path = os.path.join("model_weights", f"{os.path.basename(dataset_path)}_{model_name}.pth")
    torch.save(model.cpu().state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

def load_model(model, dataset_path, model_name):
    model_load_path = os.path.join("model_weights", f"{os.path.basename(dataset_path)}_{model_name}.pth")
    if os.path.exists(model_load_path):
        model.load_state_dict(torch.load(model_load_path))
        print(f"Model loaded from {model_load_path}")
        return True
    return False

def save_log(dataset_path, model_name, image_path, method, predicted_class, probabilities, dataset_classes):
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    log_file_name = os.path.join(output_folder, f"log_{os.path.basename(dataset_path)}_{model_name}_{os.path.basename(image_path)}_{method}.txt")

    with open(log_file_name, 'w') as log_file:
        log_file.write(f"Dataset Path: {dataset_path}\n")
        log_file.write(f"Model: {model_name}\n")
        log_file.write(f"Image Path: {image_path}\n")
        log_file.write(f"Method: {method}\n")
        log_file.write(f"Predicted Class: {predicted_class}\n")
        log_file.write("Class Probabilities:\n")
        for i, prob in enumerate(probabilities[0]):
            log_file.write(f"{dataset_classes[i]}: {prob*100:.2f}%\n")
    
    print(f"Log saved to {log_file_name}")

def main(args):
    dataset = load_dataset(args.dataset_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(models, args.model)(pretrained=True)

    if args.use_finetuning:
        # Modify the final layer to match the number of classes
        model = replace_last_layer(model, len(dataset.classes))
        model_exists = load_model(model, args.dataset_path, args.model)

        if model_exists and not args.retrain:
            print("Loaded existing model. Skipping fine-tuning.")
        else:
            # Fine-tune the model and save it
            print("Fine-tuning the model...")
            model = train_finetuned_model(dataset, model, device, epochs=5)
            save_model(model, args.dataset_path, args.model)

        if (args.image_path is None):
            print("Image path not provided. Exiting...")
            return

        # Load and process the image for classification
        test_image = load_image(args.image_path).unsqueeze(0).to(device)
        model.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(test_image)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            predicted_class = dataset.classes[predicted.item()]

            class_idx = predicted.item()

        if (args.use_cam):
            # Generate and save the CAM
            cam = generate_CAM(model, test_image, class_idx)
            save_CAM(cam, args.image_path, os.path.join("output", f"log_{os.path.basename(args.dataset_path)}_{args.model}_{os.path.basename(args.image_path)}_cam.jpg"))

        print(f"Predicted class for the image {args.image_path}: {predicted_class}")
        save_log(args.dataset_path, args.model, args.image_path, "finetuning", predicted_class, probabilities.cpu().numpy(), dataset.classes)
    
    else:
        # KNN logic
        print("Using KNN for classification...")
        dataset_and_features = get_pretrained_features(dataset, model)
        features, labels = zip(*dataset_and_features) 

        knn = train_knn(features, labels, n_neighbors=len(dataset.classes))

        # Load and process the image for classification
        test_image = load_image(args.image_path)
        test_features = get_pretrained_features([(test_image.squeeze(0), 0)], model)
        test_features = next(test_features)[0]
        prediction, probabilities = predict_knn(knn, test_features)
        predicted_class = dataset.classes[prediction[0]]

        print(f"Predicted class for the image {args.image_path}: {predicted_class}")
        save_log(args.dataset_path, args.model, args.image_path, "knn", predicted_class, probabilities, dataset.classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Classification on a Custom Dataset")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the dataset directory within 'datasets/'")
    parser.add_argument("--image_path", type=str, required=False,
                        help="Path to the image you want to classify")
    parser.add_argument("--model", type=str, default='mobilenet_v2',
                        help="Pretrained model to use for feature extraction (default: 'mobilenet_v2')")
    parser.add_argument("--use_finetuning", action='store_true',
                        help="Flag to use fine-tuning instead of KNN for classification")
    parser.add_argument("--retrain", action='store_true',
                        help="Flag to retrain and overwrite the model if it already exists in the model_weights folder")
    parser.add_argument("--use_cam", action='store_true',
                        help="Flag to also calculate the class activation map (CAM) for the image")

    args = parser.parse_args()
    main(args)
