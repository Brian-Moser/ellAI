# ellAI

## Installation

```
conda create --name ellAI python=3.7
conda activate ellAI
pip install -r requirements.txt
```

## Arguments

# Image Classification on a Custom Dataset

This script performs image classification on a custom dataset using either a fine-tuned pretrained model or a K-Nearest Neighbors (KNN) classifier. Below are the details of the arguments that can be passed to the script.

## Arguments

### `--dataset_path`
- **Type:** [`str`]
- **Required:** [`True`]
- **Description:** Path to the dataset directory within the [`datasets/`] folder. This directory should contain subdirectories for each class/category, with images inside those subdirectories.

### `--image_path`
- **Type:** [`str`]
- **Required:** [`False`]
- **Description:** Path to the image you want to classify. If not provided, the script will only train the model or KNN classifier without performing classification on a specific image.

### `--model`
- **Type:** [`str`]
- **Default:** `'mobilenet_v2'`
- **Description:** Pretrained model to use for feature extraction. The default model is [`mobilenet_v2`]. You can specify other models available in [`torchvision.models`].

### `--use_finetuning`
- **Type:** `flag`
- **Required:** [`False`]
- **Description:** Flag to use fine-tuning instead of KNN for classification. If this flag is set, the script will fine-tune the specified pretrained model on the provided dataset.

### `--retrain`
- **Type:** `flag`
- **Required:** [`False`]
- **Description:** Flag to retrain and overwrite the model if it already exists in the [`model_weights`] folder. If this flag is set, the script will retrain the model even if a saved model already exists.

## Example Usage

### Fine-tuning a different model than MobileNet
```sh
python main.py --dataset_path datasets/my_dataset --model resnet18 --use_finetuning
```

### Using KNN for Classification
```sh
python main.py --dataset_path datasets/my_dataset --image_path images/test_image.jpg
```

### Retraining a Model
```sh
python main.py --dataset_path datasets/my_dataset --model resnet18 --use_finetuning --retrain
```

## Notes
- Ensure that the dataset directory structure is correct, with subdirectories for each class containing the respective images.
- If using fine-tuning, the script will save the fine-tuned model in the [`model_weights`] folder.
- Logs of the classification results will be saved in the [`output`] folder. The naming of the log is depending on the [`dataset_path`], [`model`], [`image_path`] and the [`use_finetuning`] (either `finetuning` or `knn`). An example log name would be `log_leaves_healthy_mobilenet_v2_hainbuche_healthy.jpg_finetuning.txt`