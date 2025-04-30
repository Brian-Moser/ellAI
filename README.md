# ellAI

## Installation

```
conda create --name ellAI python=3.7
conda activate ellAI
pip install -r requirements.txt
```

## Image Classification on a Custom Dataset

This script performs image classification on a custom dataset using either a fine-tuned pretrained model or a K-Nearest Neighbors (KNN) classifier. The script supports two distinct modes: training and testing, which can be executed independently.

## Features

### Training Improvements
- **Validation Split**: 80/20 train/validation split for better model evaluation
- **Layer-wise Training**: 
  - Initially trains only the head with frozen backbone
  - Gradually unfreezes backbone after 5 epochs
- **Optimized Learning Rates**:
  - Head: 3e-3
  - Backbone: 1e-4
- **Data Augmentation**:
  - Resize to 256x256
  - Random crop to 224x224 with padding
  - Random horizontal flip
  - ImageNet normalization
- **Model Checkpointing**: Saves the best model based on validation accuracy
- **Configurable Training**:
  - Adjustable number of epochs
  - Configurable batch size
  - Automatic device selection (CPU/GPU)

### Visualization Options
- **Saliency Maps**:
  - Vanilla Saliency: Shows which pixels the model is most sensitive to for classification
  - Class Activation Maps (CAM): Highlights regions important for class prediction
  - Grad-CAM++: Improved localization, better for multiple instances
- **Output Visualization**:
  - Classification results
  - Heatmap overlays (BONE colormap for saliency, JET for CAM)
  - Probability distributions

## Arguments

### Required Arguments

#### `--dataset_path`
- **Type:** [`str`]
- **Required:** [`True`]
- **Description:** Path to the dataset directory. This directory should contain subdirectories for each class/category, with images inside those subdirectories.

#### `--mode`
- **Type:** [`str`]
- **Required:** [`True`]
- **Choices:** `train`, `test`
- **Description:** Selects whether to train the model or test it. In test mode, `--image_path` is required.

### Optional Arguments

#### `--image_path`
- **Type:** [`str`]
- **Required:** [`False`]
- **Description:** Path to the image you want to classify. Required when `--mode test` is used.

#### `--model`
- **Type:** [`str`]
- **Default:** `'mobilenet_v2'`
- **Description:** Pretrained model to use for feature extraction. The default model is [`mobilenet_v2`]. You can specify other models available in [`torchvision.models`].

#### `--use_finetuning`
- **Type:** `flag`
- **Required:** [`False`]
- **Description:** Flag to use fine-tuning instead of KNN for classification. If this flag is set, the script will fine-tune the specified pretrained model on the provided dataset.

#### `--retrain`
- **Type:** `flag`
- **Required:** [`False`]
- **Description:** Flag to retrain and overwrite the model if it already exists. If this flag is set, the script will retrain the model even if a saved model already exists.

#### `--use_cam`
- **Type:** `flag`
- **Required:** [`False`]
- **Description:** Flag to also calculate the class activation map (CAM) for the image. Only works with fine-tuned models.

#### `--use_gradcam_pp`
- **Type:** `flag`
- **Required:** [`False`]
- **Description:** Flag to use Grad-CAM++ instead of regular CAM. Provides better localization, especially for images with multiple instances of the target class.

#### `--output_path`
- **Type:** [`str`]
- **Default:** `"."`
- **Description:** Base directory where models and test results will be saved. Models are saved in `output_path/model_weights/` and test results in `output_path/test_results/`.

#### `--vis`
- **Type:** [`str`]
- **Required:** [`False`]
- **Choices:** `vanilla`, `cam`, `gradcam_pp`
- **Description:** Type of visualization to generate. Vanilla saliency map shows pixel sensitivity, CAM highlights important regions, and Grad-CAM++ provides improved localization.

#### `--epochs`
- **Type:** [`int`]
- **Default:** `10`
- **Description:** Number of epochs for training the model. Each epoch represents a complete pass through the training dataset.

#### `--batch_size`
- **Type:** [`int`]
- **Default:** `32`
- **Description:** Number of samples per batch during training. Larger batch sizes can speed up training but require more memory.

## Example Usage

### Training a Model
```sh
# Train with fine-tuning (custom epochs and batch size)
python main.py --dataset_path datasets/my_dataset \
               --mode train \
               --model mobilenet_v2 \
               --use_finetuning \
               --epochs 20 \
               --batch_size 64 \
               --output_path /path/to/save

# Train with fine-tuning (default parameters)
python main.py --dataset_path datasets/my_dataset \
               --mode train \
               --model mobilenet_v2 \
               --use_finetuning \
               --output_path /path/to/save

# Train with KNN
python main.py --dataset_path datasets/my_dataset \
               --mode train \
               --output_path /path/to/save
```

### Testing a Model
```sh
# Test with fine-tuned model (vanilla saliency)
python main.py --dataset_path datasets/my_dataset \
               --mode test \
               --image_path images/test_image.jpg \
               --model mobilenet_v2 \
               --use_finetuning \
               --vis vanilla \
               --output_path /path/to/save

# Test with fine-tuned model (regular CAM)
python main.py --dataset_path datasets/my_dataset \
               --mode test \
               --image_path images/test_image.jpg \
               --model mobilenet_v2 \
               --use_finetuning \
               --vis cam \
               --output_path /path/to/save

# Test with fine-tuned model (Grad-CAM++)
python main.py --dataset_path datasets/my_dataset \
               --mode test \
               --image_path images/test_image.jpg \
               --model mobilenet_v2 \
               --use_finetuning \
               --vis gradcam_pp \
               --output_path /path/to/save

# Test with KNN
python main.py --dataset_path datasets/my_dataset \
               --mode test \
               --image_path images/test_image.jpg \
               --output_path /path/to/save
```

### Retraining a Model
```sh
python main.py --dataset_path datasets/my_dataset \
               --mode train \
               --model mobilenet_v2 \
               --use_finetuning \
               --retrain \
               --output_path /path/to/save
```

## Notes
- Ensure that the dataset directory structure is correct, with subdirectories for each class containing the respective images.
- In training mode:
  - Fine-tuned models are saved as `.pth` files in `output_path/model_weights/`
  - KNN models are saved as `.pkl` files in `output_path/model_weights/`
  - Training progress shows both loss and validation accuracy
  - Best model is saved based on validation accuracy
  - Adjust batch size based on available GPU memory
  - More epochs may improve accuracy but increase training time
- In testing mode:
  - Classification logs are saved in `output_path/test_results/`
  - Visualization outputs (if enabled) are saved in `output_path/test_results/`
  - Choose between vanilla saliency, CAM, and Grad-CAM++ based on your needs
  - Vanilla saliency uses BONE colormap for better visualization of sensitivity patterns
  - CAM and Grad-CAM++ use JET colormap for highlighting important regions
- The script will check for required files in test mode and provide helpful error messages if they are missing
- Visualizations are only available when using fine-tuned models (KNN does not support visualizations)