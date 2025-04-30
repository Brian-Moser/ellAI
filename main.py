import os
import cv2
import torch
import argparse
import numpy as np
import pickle

from sklearn.neighbors import KNeighborsClassifier
from torchvision import datasets, transforms
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import nn, optim
from PIL import Image
from torch.utils.data.dataset import random_split

def get_last_conv_layer(model):
    layers = list(model.children())
    while layers:
        layer = layers.pop()
        if isinstance(layer, nn.Conv2d):
            return layer
        elif isinstance(layer, nn.Sequential):
            layers.extend(layer.children())
    raise ValueError("No convolutional layer found in the model.")

def to_heatmap(tensor, size=224):
    """
    tensor: 2-D or 3-D torch.Tensor on *any* device
    returns: uint8 numpy array shape (size, size)
    """
    tensor = tensor.detach()
    if tensor.dim() == 3:          # shape (C,H,W) → spatial map
        tensor = tensor.sum(0)
    tensor = torch.relu(tensor)
    if torch.all(tensor == 0):
        tensor += 1e-12            # avoid divide-by-zero
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-12)
    tensor = tensor.cpu().numpy()
    tensor = cv2.resize(tensor, (size, size))
    tensor = np.uint8(tensor * 255)
    return tensor

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
    last_conv_layer.register_full_backward_hook(
        lambda module, grad_in, grad_out: gradients.append(grad_out[0])
    )

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
    cam = to_heatmap(cam, 224)
    return cam

def save_CAM(cam, original_image_path, output_path, dataset_path, model_name, image_path):
    output_folder = os.path.join(output_path, "test_results")
    os.makedirs(output_folder, exist_ok=True)
    
    # Read and resize original image
    original_image = cv2.imread(original_image_path)
    original_image = cv2.resize(original_image, (224, 224))  # Resize to match model input size
    
    # Resize CAM to match original image size
    cam = cv2.resize(cam, (224, 224))
    cam_colored = cv2.applyColorMap(cam, cv2.COLORMAP_JET)  # Apply color map

    # Combine original image and CAM
    combined = cv2.addWeighted(original_image, 0.5, cam_colored, 0.5, 0)
    
    output_file_name = os.path.join(output_folder, f"cam_{os.path.basename(dataset_path)}_{model_name}_{os.path.basename(image_path)}.jpg")
    cv2.imwrite(output_file_name, combined)
    print(f"Class Activation Map saved to {output_file_name}")


def get_train_transforms():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

def get_test_transforms():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

def load_dataset(data_dir):
    return datasets.ImageFolder(root=data_dir, transform=get_train_transforms())

def load_image(image_path):
    return get_test_transforms()(Image.open(image_path))

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

def train_finetuned_model(dataset, model, device, epochs=10, batch_size=32):
    # Split dataset into train and validation
    n_val = int(0.2 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Freeze all layers initially
    for p in model.parameters():
        p.requires_grad = False
    
    # Create separate parameter groups for head and backbone
    head_params = []
    backbone_params = []
    for n, p in model.named_parameters():
        if n.startswith('fc') or n.startswith('classifier.'):
            head_params.append(p)
            p.requires_grad = True  # Unfreeze head
        else:
            backbone_params.append(p)
    
    # Set up optimizer with different learning rates
    optimizer = optim.Adam([
        {'params': head_params, 'lr': 3e-3},
        {'params': backbone_params, 'lr': 1e-4}
    ])
    
    criterion = nn.CrossEntropyLoss()
    model.train()
    model.to(device)
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        avg_loss = running_loss / len(train_loader)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        # Unfreeze backbone after 5 epochs
        if epoch == 5:
            for p in backbone_params:
                p.requires_grad = True
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def save_model(model, output_path, dataset_path, model_name):
    model_weights_dir = os.path.join(output_path, "model_weights")
    os.makedirs(model_weights_dir, exist_ok=True)
    model_save_path = os.path.join(model_weights_dir, f"{os.path.basename(dataset_path)}_{model_name}.pth")
    torch.save(model.cpu().state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

def load_model(model, output_path, dataset_path, model_name):
    model_load_path = os.path.join(output_path, "model_weights", f"{os.path.basename(dataset_path)}_{model_name}.pth")
    if os.path.exists(model_load_path):
        model.load_state_dict(torch.load(model_load_path))
        print(f"Model loaded from {model_load_path}")
        return True
    return False

def save_log(output_path, dataset_path, model_name, image_path, method, predicted_class, probabilities, dataset_classes):
    output_folder = os.path.join(output_path, "test_results")
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

def generate_gradcam_pp(model, img_tensor, class_idx):
    gradients = []
    feature_maps = []

    def save_gradient(grad):
        gradients.append(grad)

    def save_feature_map(module, input, output):
        feature_maps.append(output)

    # Register hooks on the last convolutional layer
    last_conv_layer = get_last_conv_layer(model)
    last_conv_layer.register_forward_hook(save_feature_map)
    last_conv_layer.register_full_backward_hook(
        lambda module, grad_in, grad_out: gradients.append(grad_out[0])
    )

    # Ensure input requires gradients
    img_tensor.requires_grad = True

    # Forward pass
    model.eval()
    outputs = model(img_tensor)
    
    # Create one-hot output for the predicted class
    one_hot_output = torch.zeros(outputs.size()).to(img_tensor.device)
    one_hot_output[0][class_idx] = 1
    
    # Zero gradients and backward pass
    model.zero_grad()
    outputs.backward(gradient=one_hot_output)

    # Get the gradients and feature map
    grad = gradients[0]      # dY/dF
    fmap = feature_maps[0]   # F

    # Grad-CAM++ computation
    alpha = (grad.pow(2) / (2*grad.pow(2) + (fmap*grad.pow(3)).sum(dim=[2,3], keepdim=True))).clamp(min=1e-7)
    weights = (alpha * grad.relu()).sum(dim=[2,3], keepdim=True)
    cam = (weights * fmap).sum(1).relu()

    # Process the CAM
    cam = to_heatmap(cam, 224)
    return cam

def generate_saliency(model, img_tensor, class_idx):
    img_tensor = img_tensor.clone().detach().requires_grad_(True)
    model.zero_grad()
    out = model(img_tensor)
    score = out[0, class_idx]
    score.backward()

    saliency = img_tensor.grad.data.abs().max(dim=1, keepdim=True)[0]   # shape (1,1,H,W)
    saliency = saliency.squeeze().cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    saliency = (saliency * 255).astype(np.uint8)
    saliency = cv2.resize(saliency, (224, 224))
    return saliency

def save_visualization(vis_map, original_image_path, output_path, dataset_path, model_name, image_path, vis_type):
    output_folder = os.path.join(output_path, "test_results")
    os.makedirs(output_folder, exist_ok=True)
    
    # Read and resize original image
    original_image = cv2.imread(original_image_path)
    original_image = cv2.resize(original_image, (224, 224))
    
    # Apply appropriate color map based on visualization type
    if vis_type == "saliency":
        vis_colored = cv2.applyColorMap(vis_map, cv2.COLORMAP_BONE)
    else:  # CAM or Grad-CAM++
        vis_colored = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)

    # Combine original image and visualization
    combined = cv2.addWeighted(original_image, 0.5, vis_colored, 0.5, 0)
    
    output_file_name = os.path.join(output_folder, f"{vis_type}_{os.path.basename(dataset_path)}_{model_name}_{os.path.basename(image_path)}.jpg")
    cv2.imwrite(output_file_name, combined)
    print(f"{vis_type.upper()} visualization saved to {output_file_name}")

def main(args):
    dataset = load_dataset(args.dataset_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(models, args.model)(pretrained=True)

    if args.mode == "train":
        if args.use_finetuning:
            # Modify the final layer to match the number of classes
            model = replace_last_layer(model, len(dataset.classes))
            model_exists = load_model(model, args.output_path, args.dataset_path, args.model)

            if model_exists and not args.retrain:
                print("Loaded existing model. Skipping fine-tuning.")
            else:
                # Fine-tune the model and save it
                print("Fine-tuning the model...")
                model = train_finetuned_model(dataset, model, device, epochs=args.epochs, batch_size=args.batch_size)
                save_model(model, args.output_path, args.dataset_path, args.model)
        else:
            # KNN training logic
            print("Using KNN for classification...")
            dataset_and_features = get_pretrained_features(dataset, model)
            features, labels = zip(*dataset_and_features) 

            knn = train_knn(features, labels, n_neighbors=len(dataset.classes))
            # Save KNN model
            knn_save_path = os.path.join(args.output_path, "model_weights", f"{os.path.basename(args.dataset_path)}_{args.model}_knn.pkl")
            os.makedirs(os.path.dirname(knn_save_path), exist_ok=True)
            with open(knn_save_path, 'wb') as f:
                pickle.dump(knn, f)
            print(f"KNN model saved to {knn_save_path}")

    elif args.mode == "test":
        if not args.image_path:
            print("Error: --image_path is required in test mode")
            return

        if args.use_finetuning:
            # Load the fine-tuned model
            model = replace_last_layer(model, len(dataset.classes))
            if not load_model(model, args.output_path, args.dataset_path, args.model):
                print(f"Error: Model not found at {os.path.join(args.output_path, 'model_weights', f'{os.path.basename(args.dataset_path)}_{args.model}.pth')}")
                print("Did you forget to train first?")
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

            if args.vis:
                if args.vis == "vanilla":
                    vis_map = generate_saliency(model, test_image, class_idx)
                    vis_type = "saliency"
                elif args.vis == "cam":
                    vis_map = generate_CAM(model, test_image, class_idx)
                    vis_type = "cam"
                elif args.vis == "gradcam_pp":
                    vis_map = generate_gradcam_pp(model, test_image, class_idx)
                    vis_type = "gradcam_pp"
                
                save_visualization(vis_map, args.image_path, args.output_path, args.dataset_path, args.model, args.image_path, vis_type)

            print(f"Predicted class for the image {args.image_path}: {predicted_class}")
            save_log(args.output_path, args.dataset_path, args.model, args.image_path, "finetuning", predicted_class, probabilities.cpu().numpy(), dataset.classes)
        
        else:
            # KNN testing logic
            knn_path = os.path.join(args.output_path, "model_weights", f"{os.path.basename(args.dataset_path)}_{args.model}_knn.pkl")
            if not os.path.exists(knn_path):
                print(f"Error: KNN model not found at {knn_path}")
                print("Did you forget to train first?")
                return

            with open(knn_path, 'rb') as f:
                knn = pickle.load(f)

            # Load and process the image for classification
            test_image = load_image(args.image_path)
            test_features = get_pretrained_features([(test_image.squeeze(0), 0)], model)
            test_features = next(test_features)[0]
            prediction, probabilities = predict_knn(knn, test_features)
            predicted_class = dataset.classes[prediction[0]]

            print(f"Predicted class for the image {args.image_path}: {predicted_class}")
            save_log(args.output_path, args.dataset_path, args.model, args.image_path, "knn", predicted_class, probabilities, dataset.classes)


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
    parser.add_argument("--vis", choices=["vanilla", "cam", "gradcam_pp"], required=False,
                        help="Type of visualization to generate: vanilla saliency map, CAM, or Grad-CAM++")
    parser.add_argument("--output_path", type=str, default=".",
                        help="Base directory for model weights and test results")
    parser.add_argument("--mode", choices=["train", "test"], required=True,
                        help="Whether to train or test the model")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs for training (default: 10)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training (default: 32)")

    args = parser.parse_args()

    # Validate arguments
    if args.mode == "test" and not args.image_path:
        parser.error("--image_path is required in test mode")

    main(args)
